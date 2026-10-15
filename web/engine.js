// The audiobook playback engine - the timeline is the book's text: it is split into units (sentences, long ones at
// semicolons and colons, and paragraphs at blank lines), which a Web Worker synthesizes on demand a few minutes ahead of
// the position. Playback schedules each unit's streamed blocks with Web Audio, and only the audio near the position is
// kept, so a whole book costs no more than a chapter.

import {encodeWav} from '../lib/wav.js';


// Timing (s) - the playback lead, and how far ahead to schedule blocks
const leadSeconds = 0.06;
const scheduleAheadSeconds = 6;


// The synthesis window - synthesize at most this many units (and characters) ahead of the position, keep this many units
// of audio behind it, and split sentences longer than this many characters at semicolons and colons (phrase boundaries)
const windowUnits = 40;
const windowChars = 2500;
const keepBehindUnits = 8;
const longSentenceChars = 200;


// The initial speaking rate estimate (characters per second at rate 1), refined by measurement
const defaultCharsPerSecond = 14;


// Abbreviations that don't end sentences
const abbreviations = new Set([
    'mr', 'mrs', 'ms', 'dr', 'st', 'vs', 'etc', 'jr', 'sr', 'prof', 'mt', 'no', 'vol', 'fig', 'ch', 'esq', 'rev', 'capt', 'col', 'gen', 'lt'
]);


// Split text into units (sentences, long sentences at semicolons and colons, and paragraphs at blank lines) - [{index,
// start, end, text, paragraphStart}]. The worker pauses longer before a paragraph's first unit than between sentences.
export function splitUnits(text) {
    const units = [];
    let start = 0;
    let paragraphStart = true;
    const addUnit = (end, endsParagraph) => {
        const unitText = text.slice(start, end);
        if (/[A-Za-z0-9]/.test(unitText)) {
            const leading = unitText.length - unitText.trimStart().length;
            units.push({
                'index': units.length, 'start': start + leading, 'end': start + unitText.trimEnd().length,
                'text': unitText.trim().replace(/\s+/g, ' '), paragraphStart
            });
            paragraphStart = false;
        }
        if (endsParagraph) {
            paragraphStart = true;
        }
        start = end;
    };
    for (let ix = 0; ix < text.length; ix++) {
        const char = text[ix];
        if (char === '\n' && /^\n[ \t\r]*\n/.test(text.slice(ix, ix + 40))) {
            addUnit(ix, true);
        } else if ((char === ';' || char === ':') && ix - start > longSentenceChars && /\s/.test(text[ix + 1] || ' ')) {
            addUnit(ix + 1, false);
        } else if ('.!?'.includes(char)) {
            let end = ix + 1;
            while (end < text.length && '.!?"\')]”’'.includes(text[end])) {
                end++;
            }
            if (end < text.length && !/\s/.test(text[end])) {
                continue;
            }
            const word = text.slice(Math.max(start, ix - 20), ix).match(/([A-Za-z]+)$/);
            if (char === '.' && word !== null && (abbreviations.has(word[1].toLowerCase()) || /^[A-Z]$/.test(word[1]))) {
                continue;
            }
            addUnit(end, false);
            ix = end - 1;
        }
    }
    addUnit(text.length, true);
    return units;
}


// Headings numbered with Roman numerals ('Chapter IV') - the label and the numeral
const headingLabels = [
    'Chapter', 'CHAPTER', 'Book', 'BOOK', 'Part', 'PART', 'Volume', 'VOLUME', 'Act', 'ACT', 'Scene', 'SCENE', 'Stave', 'STAVE',
    'Letter', 'LETTER', 'Canto', 'CANTO', 'Section', 'SECTION'
];
const romanHeadingPattern = new RegExp(`\\b(${headingLabels.join('|')})\\s+([IVXLCDM]+)\\b`, 'g');


// A Roman numeral's value
function romanValue(roman) {
    const values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000};
    let total = 0;
    for (let ix = 0; ix < roman.length; ix++) {
        const value = values[roman[ix]];
        total += ix + 1 < roman.length && values[roman[ix + 1]] > value ? -value : value;
    }
    return total;
}


// A unit's text for speaking - the display text keeps its typography, but the synthesizer gets dashes as pauses,
// ellipses as periods, no invisible formatting characters, and Roman-numeral headings as numbers ('Chapter IV' ->
// 'Chapter 4')
export function speechText(text) {
    return text.
        replace(/[\u2060\u00ad\u200b\ufeff]/g, '').
        replace(/[\u200a\u2009\u202f\u00a0]/g, ' ').
        replace(/\s*[\u2014\u2013]+\s*/g, ', ').
        replace(/\u2026/g, '...').
        replace(romanHeadingPattern, (match, label, roman) => `${label} ${romanValue(roman)}`).
        replace(/^([IVXLCDM]+)\.$/, (match, roman) => `${romanValue(roman)}.`);
}


// The engine - load a book, then play, pause, skip, and seek. onChange(state) is called when the state changes (playing,
// paused, buffering, ended, or a new unit), and the caller may poll state() for the position.
export class Engine {
    constructor({onChange = () => {
        // No listener
    }} = {}) {
        this.onChange = onChange;
        this.audioContext = null;
        this.output = null;
        this.mediaElement = null;
        this.worker = null;
        this.workerReady = false;
        this.workerStarts = 0;
        this.nextRunId = 1;
        this.session = null;
        this.playing = false;
        this.unit = 0;
        this.time = 0;
        this.chain = [];
        this.sources = new Set();
        this.ended = false;
        this.lastUnit = -1;
        this.lastMaintenance = 0;
        this.measuredChars = 0;
        this.measuredSeconds = 0;
        this.startWorker();
        this.timer = setInterval(() => this.tick(), 100);
    }

    // Start the synthesis worker - one that fails before it's ready (a dropped module request, say) gets one fresh start
    startWorker() {
        this.workerStarts += 1;
        this.worker = new Worker(new URL('worker.js', import.meta.url), {'type': 'module'});
        this.worker.onmessage = (event) => this.onWorkerMessage(event.data);
        this.worker.onerror = (event) => {
            event.preventDefault();
            if (!this.workerReady && this.workerStarts < 2) {
                this.worker.terminate();
                this.startWorker();
                this.restartRuns();
                return;
            }
            this.error = `The speech synthesizer failed to start: ${event.message || 'serve the app over HTTP'}`;
            this.notify();
        };
    }

    // After a worker restart - forget the old worker's runs (their units are asked of the new one) and prepare the voice
    restartRuns() {
        const {session} = this;
        if (session !== null) {
            session.run = null;
            for (const cell of session.cache) {
                if (!cell.done) {
                    Object.assign(cell, {'blocks': [], 'length': 0, 'run': null});
                }
            }
            this.worker.postMessage({'type': 'prepare', 'voice': session.voice});
        }
    }

    // Is a book loaded?
    get loaded() {
        return this.session !== null;
    }

    // Load a book's text, paused at a character offset (snapped to its sentence) - with the voice and rate
    load(text, {voice = 'neural-male', rate = 1} = {}, startChar = 0) {
        this.stopSources();
        this.cancelRun();
        this.playing = false;
        this.ended = false;
        // The speaking rate is measured afresh for each book, voice, and rate
        this.measuredChars = 0;
        this.measuredSeconds = 0;
        const units = splitUnits(text);
        this.session = {
            text, units, voice, rate, 'run': null, 'runIds': new Set(),
            'cache': units.map(() => ({'blocks': [], 'length': 0, 'done': false, 'run': null}))
        };
        this.unit = units.length === 0 ? 0 : this.unitAtChar(startChar);
        this.time = 0;
        // Have the worker load the voice now, so the first play starts quickly
        this.worker.postMessage({'type': 'prepare', voice});
        this.notify();
    }

    // Change the voice or rate - continuing from the current sentence
    setVoice({voice, rate}) {
        if (this.session === null || (voice === this.session.voice && rate === this.session.rate)) {
            return;
        }
        const wasPlaying = this.playing;
        const {unit} = this.position();
        this.load(this.session.text, {voice, rate}, this.session.units[unit].start);
        if (wasPlaying) {
            this.play();
        }
    }

    // Play from the position (from the start at the end) - call from a user gesture (it unlocks audio)
    play() {
        if (this.session === null || this.session.units.length === 0) {
            return;
        }
        this.ensureAudio();
        if (this.ended) {
            this.ended = false;
            this.unit = 0;
            this.time = 0;
        }
        this.startPlayback(this.unit, this.time);
    }

    // Pause, keeping the position
    pause() {
        if (this.playing) {
            const current = this.position();
            this.stopSources();
            this.playing = false;
            this.unit = current.unit;
            this.time = current.time;
            if (this.mediaElement !== null) {
                this.mediaElement.pause();
            }
            this.notify();
        }
    }

    // Toggle play and pause
    toggle() {
        if (this.playing) {
            this.pause();
        } else {
            this.play();
        }
    }

    // Skip about a number of seconds (negative back) - to a sentence start: ahead at least to the next sentence; back to
    // this sentence's start if more than two seconds into it, else to an earlier one
    skip(seconds) {
        if (this.session === null || this.session.units.length === 0) {
            return;
        }
        const current = this.position();
        const target = this.positionChar(current) + seconds * this.charsPerSecond();
        let unit = this.unitAtChar(Math.max(0, Math.min(this.textLength() - 1, target)));
        if (seconds > 0 && unit <= current.unit) {
            unit = current.unit + 1;
        } else if (seconds < 0 && unit >= current.unit) {
            unit = current.time > 2 ? current.unit : current.unit - 1;
        }
        if (unit >= this.session.units.length) {
            this.finish();
            return;
        }
        this.moveTo(Math.max(0, unit));
    }

    // Seek to a character offset's sentence
    seekChar(char) {
        if (this.session !== null && this.session.units.length !== 0) {
            this.ended = false;
            this.moveTo(this.unitAtChar(char));
        }
    }

    // The state - {status ('loading', 'paused', 'buffering', 'playing', 'ended'), char (the position, interpolated
    // through its sentence), sentenceChar (its sentence's start), percent, sentence (its text), remainingSeconds}
    state() {
        if (this.session === null || this.session.units.length === 0) {
            return {
                'status': this.workerReady ? 'paused' : 'loading', 'char': 0, 'sentenceChar': 0, 'percent': 0, 'sentence': '',
                'remainingSeconds': 0
            };
        }
        const current = this.position();
        const char = this.positionChar(current);
        let status = 'paused';
        if (this.ended) {
            status = 'ended';
        } else if (this.playing) {
            const now = this.audioContext.currentTime;
            const started = this.chain.some((entry) => entry.startTime !== null && entry.startTime <= now);
            status = started ? 'playing' : 'buffering';
        } else if (!this.workerReady) {
            status = 'loading';
        }
        return {
            status, char, 'sentenceChar': this.session.units[current.unit].start, 'percent': 100 * char / this.textLength(),
            'sentence': this.session.units[current.unit].text,
            'remainingSeconds': (this.textLength() - char) / this.charsPerSecond()
        };
    }

    // The speaking rate (characters per second) - measured from synthesized units, at the session's rate
    charsPerSecond() {
        // The measurement is blended with the default, weighted as a minute of audio, so the estimate settles smoothly
        const rate = this.session === null ? 1 : this.session.rate;
        const priorSeconds = 60;
        return (this.measuredChars + defaultCharsPerSecond * rate * priorSeconds) / (this.measuredSeconds + priorSeconds);
    }

    // Notify the caller of a state change
    notify() {
        this.onChange(this.state());
    }

    // Create the audio context and output (on a user gesture). A silent looping media element keeps the page's audio
    // session in playback mode (on iOS, Web Audio alone follows the ringer switch) and enables the lock-screen controls.
    ensureAudio() {
        if (this.audioContext === null) {
            if (navigator.audioSession) {
                navigator.audioSession.type = 'playback';
            }
            this.audioContext = new AudioContext({'latencyHint': 'playback'});
            this.output = this.audioContext.createGain();
            this.output.connect(this.audioContext.destination);
            this.mediaElement = document.createElement('audio');
            this.mediaElement.loop = true;
            this.mediaElement.setAttribute('playsinline', '');
            this.mediaElement.src = URL.createObjectURL(new Blob([encodeWav(new Float32Array(8000), 8000)], {'type': 'audio/wav'}));
        }
        this.audioContext.resume();
        this.mediaElement.play().catch(() => {
            // Autoplay restrictions - Web Audio still plays after the gesture
        });
    }

    // Handle a worker message - the worker's readiness, and unit audio (the first run to reach a unit owns it)
    onWorkerMessage(message) {
        if (message.type === 'ready') {
            this.workerReady = true;
            this.notify();
            return;
        }
        if (message.type === 'error') {
            // A worker that couldn't load its models gets one fresh start, as a worker that failed to load
            if (!this.workerReady && this.workerStarts < 2) {
                this.worker.terminate();
                this.startWorker();
                this.restartRuns();
                return;
            }
            this.error = message.message;
            this.notify();
            return;
        }
        const {session} = this;
        if (session === null || !session.runIds.has(message.id)) {
            return;
        }
        if (message.type === 'runDone') {
            if (session.run !== null && session.run.id === message.id) {
                session.run.active = false;
            }
            return;
        }
        const cell = session.cache[message.unit];
        if (cell.done || (cell.run !== null && cell.run !== message.id)) {
            return;
        }
        cell.run = message.id;
        if (message.type === 'unitBlock') {
            // Blocks keep their samples - audio buffers are made when they're scheduled (the audio context may not
            // exist yet, before the first play)
            const duration = message.samples.length / message.sampleRate;
            cell.blocks.push({'samples': message.samples, 'sampleRate': message.sampleRate, 'start': cell.length, duration});
            cell.length += duration;
        } else if (message.type === 'unitDone') {
            cell.done = true;
            this.measuredChars += session.units[message.unit].text.length;
            this.measuredSeconds += cell.length;
        }
        this.pump();
    }

    // Unit queries - is a unit ready to start (it has audio), complete, and its length (s)
    unitReady(unit) {
        const cell = this.session.cache[unit];
        return cell.blocks.length !== 0 || cell.done;
    }

    unitComplete(unit) {
        return this.session.cache[unit].done;
    }

    unitLength(unit) {
        return this.session.cache[unit].length;
    }

    // The text's length for positions (to the end of its last unit)
    textLength() {
        return this.session.units[this.session.units.length - 1].end;
    }

    // The unit at a character offset - the first unit ending after it (a binary search)
    unitAtChar(char) {
        const {units} = this.session;
        let low = 0;
        let high = units.length - 1;
        while (low < high) {
            const middle = (low + high) >> 1;
            if (units[middle].end > char) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        return low;
    }

    // The position - {unit, time}
    position() {
        if (!this.playing) {
            return {'unit': this.unit, 'time': this.time};
        }
        const now = this.audioContext.currentTime;
        for (let ix = this.chain.length - 1; ix >= 0; ix--) {
            const entry = this.chain[ix];
            if (entry.startTime !== null && entry.startTime <= now) {
                let time = now - entry.startTime;
                if (ix === 0) {
                    time = Math.max(time, entry.from);
                }
                if (this.unitComplete(entry.unit)) {
                    time = Math.min(time, this.unitLength(entry.unit));
                }
                return {'unit': entry.unit, time};
            }
        }
        return {'unit': this.chain[0].unit, 'time': this.chain[0].from};
    }

    // A position's character offset - interpolated through its unit by time
    positionChar({unit, time}) {
        const target = this.session.units[unit];
        const complete = this.unitComplete(unit);
        const length = complete ? this.unitLength(unit) : Math.max(0.001, target.text.length / this.charsPerSecond());
        const fraction = Math.max(0, Math.min(complete ? 1 : 0.95, time / length));
        return target.start + fraction * (target.end - target.start);
    }

    // Make sure the units in the window from a unit on are being synthesized - keeping the current run if it reaches the
    // first missing unit next
    ensureSynthesis(from) {
        const {session} = this;
        const remaining = [];
        const charLimit = session.units[from].start + windowChars;
        for (let unit = from; unit < session.units.length && unit < from + windowUnits; unit++) {
            if (unit > from + 1 && session.units[unit].start > charLimit) {
                break;
            }
            if (!this.unitComplete(unit)) {
                remaining.push(unit);
            }
        }
        if (remaining.length === 0) {
            return;
        }
        const {run} = session;
        if (run !== null && run.active) {
            const index = run.order.indexOf(remaining[0]);
            if (index !== -1 && run.order.slice(0, index).every((unit) => this.unitComplete(unit))) {
                return;
            }
        }
        const id = this.nextRunId++;
        session.run = {id, 'order': remaining, 'active': true};
        session.runIds.add(id);
        this.worker.postMessage({
            'type': 'units', id, 'voice': session.voice, 'rate': session.rate,
            'units': remaining.map((unit) => {
                const {index, text, paragraphStart} = session.units[unit];
                return {index, 'text': speechText(text), paragraphStart};
            })
        });
    }

    // Cancel the worker's unit run
    cancelRun() {
        if (this.worker !== null) {
            this.worker.postMessage({'type': 'cancel'});
        }
        if (this.session !== null && this.session.run !== null) {
            this.session.run.active = false;
        }
    }

    // Drop the cached audio of units outside the window around a unit (and not playing)
    evictCache(center) {
        const {session} = this;
        const playing = new Set(this.playing ? this.chain.map((entry) => entry.unit) : []);
        const charLimit = session.units[center].start + windowChars;
        for (let unit = 0; unit < session.cache.length; unit++) {
            const cell = session.cache[unit];
            if (cell.blocks.length === 0 && !cell.done) {
                continue;
            }
            const inWindow = unit >= center - keepBehindUnits && unit < center + windowUnits &&
                (unit <= center + 1 || session.units[unit].start <= charLimit);
            if (!inWindow && !playing.has(unit)) {
                session.cache[unit] = {'blocks': [], 'length': 0, 'done': false, 'run': null};
            }
        }
    }

    // Schedule a block of a chain entry's unit - skipping what has already passed
    scheduleBlock(entry, block) {
        let when = entry.startTime + block.start;
        let offset = 0;
        const earliest = this.audioContext.currentTime + 0.005;
        if (when + block.duration <= earliest) {
            return;
        }
        if (when < earliest) {
            offset = earliest - when;
            when = earliest;
        }
        const buffer = this.audioContext.createBuffer(1, block.samples.length, block.sampleRate);
        buffer.copyToChannel(block.samples, 0);
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.output);
        source.start(when, offset);
        source.onended = () => this.sources.delete(source);
        this.sources.add(source);
    }

    // Stop the scheduled sources
    stopSources() {
        for (const source of this.sources) {
            source.onended = null;
            source.stop();
        }
        this.sources.clear();
    }

    // Play from a position
    startPlayback(unit, time) {
        this.stopSources();
        this.playing = true;
        this.chain = [{unit, 'from': time, 'startTime': null, 'cursor': 0}];
        this.ensureSynthesis(unit);
        this.pump();
        this.notify();
    }

    // Move to a unit's start - playing from there if playing, else preparing its audio
    moveTo(unit) {
        if (this.playing) {
            this.startPlayback(unit, 0);
        } else {
            this.unit = unit;
            this.time = 0;
            this.ensureSynthesis(unit);
            this.notify();
        }
    }

    // Stop at the end of the book
    finish() {
        this.stopSources();
        this.playing = false;
        this.ended = true;
        this.unit = this.session.units.length - 1;
        this.time = this.unitComplete(this.unit) ? this.unitLength(this.unit) : 0;
        if (this.mediaElement !== null) {
            this.mediaElement.pause();
        }
        this.notify();
    }

    // Advance playback - start chain units when they can start, schedule their blocks ahead, extend the chain with the
    // next unit, and finish at the end of the book
    pump() {
        if (!this.playing) {
            return;
        }
        const now = this.audioContext.currentTime;
        for (let ix = 0; ix < this.chain.length; ix++) {
            const entry = this.chain[ix];
            if (entry.startTime === null) {
                const previous = this.chain[ix - 1];
                if (ix === 0 && this.unitReady(entry.unit)) {
                    entry.startTime = now + leadSeconds - entry.from;
                } else if (ix !== 0 && previous.startTime !== null && this.unitComplete(previous.unit) && this.unitReady(entry.unit)) {
                    entry.startTime = Math.max(previous.startTime + this.unitLength(previous.unit), now + 0.03);
                } else {
                    break;
                }
            }
            const cell = this.session.cache[entry.unit];
            const horizon = now + scheduleAheadSeconds;
            while (entry.cursor < cell.blocks.length && entry.startTime + cell.blocks[entry.cursor].start < horizon) {
                this.scheduleBlock(entry, cell.blocks[entry.cursor]);
                entry.cursor++;
            }
            if (ix === this.chain.length - 1 && this.unitComplete(entry.unit) && entry.unit + 1 < this.session.units.length) {
                this.chain.push({'unit': entry.unit + 1, 'from': 0, 'startTime': null, 'cursor': 0});
            }
        }

        // Drop finished units from the front of the chain
        while (this.chain.length > 1 && this.chain[1].startTime !== null && this.chain[1].startTime < now - 1) {
            this.chain.shift();
        }

        // The end of the book
        const last = this.chain[this.chain.length - 1];
        if (last.unit === this.session.units.length - 1 && last.startTime !== null && this.unitComplete(last.unit) &&
            now >= last.startTime + this.unitLength(last.unit) + 0.05) {
            this.finish();
        }
    }

    // The clock - advance playback, keep synthesis and the cache around the position, and report unit changes
    tick() {
        if (this.session === null || this.session.units.length === 0) {
            return;
        }
        this.pump();
        const {unit} = this.position();
        if (this.playing) {
            this.ensureSynthesis(unit);
        }
        const now = performance.now();
        if (now - this.lastMaintenance > 2000) {
            this.lastMaintenance = now;
            this.evictCache(unit);
        }
        if (unit !== this.lastUnit) {
            this.lastUnit = unit;
            this.notify();
        }
    }
}
