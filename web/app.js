// The browser front end - text, voice, rate, room, and crowd controls; a player whose timeline is the text (sentences
// are synthesized on demand in Web Workers, one per crowd member, and cached); play/pause, skipping by 1% or 5% of the
// text, a position slider, and bookmarks (per text, in browser storage); playback with Web Audio (each member panned,
// the room ambience by a convolver); and WAV download

import {encodeWav} from '../lib/wav.js';
import {convolve, roomImpulse} from '../lib/room.js';
import {normalize} from '../lib/klatt.js';


// Elements
const textElement = document.getElementById('text');
const rateElement = document.getElementById('rate');
const rateValueElement = document.getElementById('rateValue');
const roomElement = document.getElementById('room');
const crowdElement = document.getElementById('crowd');
const speakButton = document.getElementById('speak');
const stopButton = document.getElementById('stop');
const downloadButton = document.getElementById('download');
const skipButtons = [...document.querySelectorAll('button[data-skip]')];
const playPauseButton = document.getElementById('playPause');
const bookmarkButton = document.getElementById('bookmark');
const seekElement = document.getElementById('seek');
const percentElement = document.getElementById('percent');
const nowElement = document.getElementById('now');
const statusElement = document.getElementById('status');
const bookmarksSection = document.getElementById('bookmarksSection');
const bookmarksElement = document.getElementById('bookmarks');


// The crowd - the lead (the chosen voice, centered) and five other speakers, each with its own voice, pitch, vocal
// tract length, rate, seed, start offset (s), stereo position, and level
const crowdMembers = [
    {'voice': null, 'pitchScale': 1, 'tractScale': 1, 'rate': 1, 'seed': 1, 'offset': 0, 'pan': 0, 'gain': 1},
    {'voice': 'male', 'pitchScale': 0.88, 'tractScale': 1.05, 'rate': 0.95, 'seed': 11, 'offset': 0.11, 'pan': -0.6, 'gain': 0.72},
    {'voice': 'female', 'pitchScale': 1.08, 'tractScale': 0.97, 'rate': 1.04, 'seed': 12, 'offset': 0.05, 'pan': 0.5, 'gain': 0.72},
    {'voice': 'male', 'pitchScale': 1.12, 'tractScale': 0.96, 'rate': 1.06, 'seed': 13, 'offset': 0.2, 'pan': 0.3, 'gain': 0.66},
    {'voice': 'female', 'pitchScale': 0.9, 'tractScale': 1.03, 'rate': 0.97, 'seed': 14, 'offset': 0.16, 'pan': -0.35, 'gain': 0.66},
    {'voice': 'male', 'pitchScale': 0.95, 'tractScale': 1, 'rate': 1.02, 'seed': 15, 'offset': 0.27, 'pan': 0.8, 'gain': 0.58}
];


// Timing (s) - the playback lead, how far ahead to schedule blocks, and the dropped silence at a mid-paragraph unit's start
const leadSeconds = 0.06;
const scheduleAheadSeconds = 4;
const unitTrimSeconds = 0.12;


// The synthesis window - synthesize at most this many units (and characters) ahead of the position, keep this many units
// of audio behind it (so long texts, like books, need only a few minutes of audio in memory), and split sentences longer
// than this many characters at semicolons and colons (phrase boundaries)
const windowUnits = 40;
const windowChars = 2500;
const keepBehindUnits = 8;
const longSentenceChars = 200;


// State - the workers ([{worker, ready}]), the audio graph, the download request, and the session: the text's units
// (sentences) and their synthesized audio - cache[unit][member] = {blocks: [{buffer, start}], length, done, run}
let workers = [];
let audioContext = null;
let output = null;
let nextRunId = 1;
let download = null;
let session = null;


// The player - its position is a unit and a time within it. While playing, the chain is the units being played
// ([{unit, from, startTime (audio context time of the unit's start, or null until it can start), cursors}]).
const player = {'playing': false, 'unit': 0, 'time': 0, 'chain': [], 'sources': new Set(), 'dragging': false};


// Show a status message
function setStatus(message, isError = false) {
    statusElement.textContent = message;
    statusElement.classList.toggle('error', isError);
}


// The current settings, and keys identifying the synthesized audio (the room is applied at playback)
function getSettings() {
    const voice = document.querySelector('input[name="voice"]:checked').value;
    const settings = {
        'text': textElement.value, voice, 'rate': Number(rateElement.value), 'room': roomElement.checked, 'crowd': crowdElement.checked
    };
    const key = JSON.stringify({'text': settings.text, voice, 'rate': settings.rate, 'crowd': settings.crowd});
    return {settings, key};
}


// The speakers for settings - the crowd, or just the chosen voice
function getMembers(settings) {
    const members = settings.crowd ? crowdMembers : crowdMembers.slice(0, 1);
    return members.map((member) => ({...member, 'voice': member.voice === null ? settings.voice : member.voice}));
}


// A text's hash (FNV-1a) - bookmarks belong to a text
function hashText(text) {
    let hash = 0x811c9dc5;
    for (let ix = 0; ix < text.length; ix++) {
        hash = Math.imul(hash ^ text.charCodeAt(ix), 0x01000193) >>> 0;
    }
    return hash.toString(16);
}


// Split text into units (sentences, long sentences at semicolons and colons, and paragraphs at blank lines) - [{index,
// start, end, text, trim}]. A unit after another in the same paragraph drops its leading silence (trim), so sentences
// flow at a natural pace.
const abbreviations = new Set(['mr', 'mrs', 'ms', 'dr', 'st', 'vs', 'etc', 'jr', 'sr', 'prof', 'mt', 'no', 'vol', 'fig']);
export function splitUnits(text) {
    const units = [];
    let start = 0;
    let paragraphStart = true;
    const addUnit = (end, endsParagraph) => {
        const unitText = text.slice(start, end);
        if (/[A-Za-z0-9]/.test(unitText)) {
            const leading = unitText.length - unitText.trimStart().length;
            units.push({
                'index': units.length, 'start': start + leading, 'end': start + unitText.trimEnd().length, 'text': unitText.trim(),
                'trim': paragraphStart ? 0 : unitTrimSeconds
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
            const word = text.slice(start, ix).match(/([A-Za-z]+)$/);
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


// Session audio queries - is a unit ready to start (every member has audio), complete (every member done), and its
// length (s, when complete - the longest member, with its offset)
function unitReady(unit) {
    return session.cache[unit].every((cell) => cell.blocks.length !== 0 || cell.done);
}

function unitComplete(unit) {
    return session.cache[unit].every((cell) => cell.done);
}

function unitLength(unit) {
    return Math.max(...session.cache[unit].map((cell, member) => session.members[member].offset + cell.length));
}


// The player's position - {unit, time}
function position() {
    if (!player.playing) {
        return {'unit': player.unit, 'time': player.time};
    }
    const now = audioContext.currentTime;
    for (let ix = player.chain.length - 1; ix >= 0; ix--) {
        const entry = player.chain[ix];
        if (entry.startTime !== null && entry.startTime <= now) {
            let time = now - entry.startTime;
            if (ix === 0) {
                time = Math.max(time, entry.from);
            }
            if (unitComplete(entry.unit)) {
                time = Math.min(time, unitLength(entry.unit));
            }
            return {'unit': entry.unit, time};
        }
    }
    return {'unit': player.chain[0].unit, 'time': player.chain[0].from};
}


// A position's character offset in the text - interpolated through its unit by time
function positionChar({unit, time}) {
    const target = session.units[unit];
    const length = unitComplete(unit) ? unitLength(unit) : Math.max(0.001, unitLength(unit));
    let fraction = Math.max(0, Math.min(1, time / length));
    if (!unitComplete(unit)) {
        fraction = Math.min(fraction, 0.95);
    }
    return target.start + fraction * (target.end - target.start);
}


// The text's length for positions (to the end of its last unit)
function textLength() {
    return session.units[session.units.length - 1].end;
}


// The unit at a character offset - the first unit ending after it (a binary search)
function unitAtChar(char) {
    let low = 0;
    let high = session.units.length - 1;
    while (low < high) {
        const middle = (low + high) >> 1;
        if (session.units[middle].end > char) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}


// Is the session synthesizing?
function isSynthesizing() {
    return session !== null && session.run !== null && session.run.active;
}


// Update the controls for the current state
function updateControls() {
    const hasSession = session !== null && session.units.length !== 0;
    stopButton.disabled = !player.playing && !isSynthesizing();
    playPauseButton.disabled = false;
    playPauseButton.innerHTML = player.playing ? '&#x23F8; Pause' : '&#x25B6; Play';
    playPauseButton.setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
    for (const button of [...skipButtons, bookmarkButton]) {
        button.disabled = !hasSession;
    }
    seekElement.disabled = !hasSession;
    if (!hasSession) {
        percentElement.textContent = '0%';
        nowElement.textContent = '';
        return;
    }
    const current = position();
    const percent = 100 * positionChar(current) / textLength();
    if (!player.dragging) {
        seekElement.value = Math.round(percent * 10);
        percentElement.textContent = `${Math.floor(percent)}%`;
        nowElement.textContent = session.units[current.unit].text;
    }
}


// Create a synthesis worker at an index of the pool
function createWorker(index) {
    const worker = new Worker(new URL('worker.js', import.meta.url), {'type': 'module'});
    const entry = {worker, 'ready': false};
    worker.onmessage = (event) => onWorkerMessage(entry, event.data);
    worker.onerror = (event) => {
        event.preventDefault();
        setStatus(`The synthesis worker failed: ${event.message || 'could not load (serve the page over HTTP)'}`, true);
    };
    workers[index] = entry;
}


// Make sure the pool has a count of workers
function ensureWorkers(count) {
    for (let ix = workers.length; ix < count; ix++) {
        createWorker(ix);
    }
}


// Cancel the workers' unit runs
function cancelRuns() {
    for (const entry of workers) {
        entry.worker.postMessage({'type': 'cancel'});
    }
    if (session !== null && session.run !== null) {
        session.run.active = false;
    }
}


// Handle a worker message
function onWorkerMessage(entry, message) {
    if (message.type === 'ready') {
        entry.ready = true;
        if (session === null && download === null) {
            setStatus('Ready.');
        }
    } else if (message.type === 'error' && message.id === null) {
        // A worker could not start (the dictionary failed to load)
        setStatus(message.message, true);
    } else if (download !== null && message.id === download.id) {
        onDownloadMessage(message);
    } else if (session !== null && session.runIds.has(message.id)) {
        onUnitMessage(message);
    }
    updateControls();
}


// Handle a unit-run message - add a block to its cell (the first run to reach a cell owns it), mark a cell done, or a
// run done
function onUnitMessage(message) {
    if (message.type === 'error') {
        setStatus(message.message, true);
        return;
    }
    if (message.type === 'runDone') {
        if (session.run !== null && session.run.id === message.id && ++session.run.membersDone === session.members.length) {
            session.run.active = false;
        }
        return;
    }
    const cell = session.cache[message.unit][message.member];
    if (cell.done || (cell.run !== null && cell.run !== message.id)) {
        return;
    }
    cell.run = message.id;
    if (message.type === 'unitBlock') {
        const buffer = audioContext.createBuffer(1, message.samples.length, message.sampleRate);
        buffer.copyToChannel(message.samples, 0);
        cell.blocks.push({buffer, 'start': cell.length});
        cell.length += buffer.duration;
    } else if (message.type === 'unitDone') {
        cell.done = true;
    }
    pump();
}


// Create the session for the current settings (keeping the current one if its settings are unchanged) - returns it, or
// null if there is nothing to say
function ensureSession() {
    const {settings, key} = getSettings();
    if (session !== null && session.key === key) {
        return session;
    }
    cancelRuns();
    const units = splitUnits(settings.text);
    if (units.length === 0) {
        session = null;
        setStatus('Nothing to say - type some text.');
        return null;
    }
    const members = getMembers(settings);
    ensureWorkers(members.length);
    session = {
        key, settings, members, units, 'hash': hashText(settings.text), 'run': null, 'runIds': new Set(), 'nodes': [],
        'cache': units.map(() => members.map(() => ({'blocks': [], 'length': 0, 'done': false, 'run': null})))
    };
    return session;
}


// Make sure the session's units in the window from a unit on are being synthesized - keeping the current run if it
// reaches the first missing unit next
function ensureSynthesis(from) {
    const remaining = [];
    const charLimit = session.units[from].start + windowChars;
    for (let unit = from; unit < session.units.length && unit < from + windowUnits; unit++) {
        if (unit > from + 1 && session.units[unit].start > charLimit) {
            break;
        }
        if (!unitComplete(unit)) {
            remaining.push(unit);
        }
    }
    if (remaining.length === 0) {
        return;
    }
    const run = session.run;
    if (run !== null && run.active) {
        const index = run.order.indexOf(remaining[0]);
        if (index !== -1 && run.order.slice(0, index).every((unit) => unitComplete(unit))) {
            return;
        }
    }
    const id = nextRunId++;
    session.run = {id, 'order': remaining, 'active': true, 'membersDone': 0};
    session.runIds.add(id);
    session.members.forEach((member, ix) => {
        const units = remaining.filter((unit) => !session.cache[unit][ix].done).map((unit) => {
            const {index, text, trim} = session.units[unit];
            return {index, text, trim};
        });
        workers[ix].worker.postMessage({
            'type': 'units', id, 'member': ix, units, 'voice': member.voice, 'rate': session.settings.rate * member.rate,
            'seed': member.seed, 'overrides': {'pitchScale': member.pitchScale, 'tractScale': member.tractScale}
        });
    });
}


// Drop the cached audio of units outside the window around a unit (and not playing)
function evictCache(center) {
    const playing = new Set(player.playing ? player.chain.map((entry) => entry.unit) : []);
    const charLimit = session.units[center].start + windowChars;
    session.cache.forEach((cells, unit) => {
        const inWindow = unit >= center - keepBehindUnits && (unit < center + windowUnits) &&
            (unit <= center + 1 || session.units[unit].start <= charLimit);
        if (!inWindow && !playing.has(unit) && cells.some((cell) => cell.blocks.length !== 0 || cell.done)) {
            session.cache[unit] = cells.map(() => ({'blocks': [], 'length': 0, 'done': false, 'run': null}));
        }
    });
}


// Keep synthesis and the cache around the position - called by the clock
let lastMaintenance = 0;
function maintainWindow() {
    if (session === null || session.units.length === 0) {
        return;
    }
    const {unit} = position();
    if (player.playing) {
        ensureSynthesis(unit);
    }
    const now = performance.now();
    if (now - lastMaintenance > 2000) {
        lastMaintenance = now;
        evictCache(unit);
    }
}


// Create the audio context (on a user gesture)
function ensureAudio() {
    if (audioContext === null) {
        audioContext = new AudioContext();
    }
    audioContext.resume();
}


// Connect the output for the room setting - the room ambience is a convolver - and the session members' playback nodes
// (each member's level and stereo position)
function connectOutput() {
    if (output !== null) {
        output.disconnect();
    }
    const room = roomElement.checked;
    // A crowd is louder - its members sum
    const master = audioContext.createGain();
    master.gain.value = session.members.length > 1 ? 0.42 : 1;
    master.connect(audioContext.destination);
    if (room) {
        // The convolver's impulse response must be at the context's sample rate
        const impulse = roomImpulse(audioContext.sampleRate);
        const buffer = audioContext.createBuffer(1, impulse.length, audioContext.sampleRate);
        buffer.copyToChannel(impulse, 0);
        const convolver = audioContext.createConvolver();
        convolver.normalize = false;
        convolver.buffer = buffer;
        const gain = audioContext.createGain();
        gain.gain.value = 0.85;
        convolver.connect(gain).connect(master);
        output = convolver;
    } else {
        output = master;
    }
    session.nodes = session.members.map((member) => {
        const gain = audioContext.createGain();
        gain.gain.value = member.gain;
        const panner = audioContext.createStereoPanner();
        panner.pan.value = member.pan;
        gain.connect(panner).connect(output);
        return gain;
    });
}


// Schedule a block of a chain entry's unit - skipping what has already passed
function scheduleBlock(entry, member, block) {
    let when = entry.startTime + session.members[member].offset + block.start;
    let offset = 0;
    const earliest = audioContext.currentTime + 0.005;
    if (when + block.buffer.duration <= earliest) {
        return;
    }
    if (when < earliest) {
        offset = earliest - when;
        when = earliest;
    }
    const source = audioContext.createBufferSource();
    source.buffer = block.buffer;
    source.connect(session.nodes[member]);
    source.start(when, offset);
    source.onended = () => player.sources.delete(source);
    player.sources.add(source);
}


// Stop the scheduled sources
function stopSources() {
    for (const source of player.sources) {
        source.onended = null;
        source.stop();
    }
    player.sources.clear();
}


// Advance playback - start chain units when they can start, schedule their blocks a few seconds ahead, extend the
// chain with the next unit, and finish at the end of the text
function pump() {
    if (!player.playing) {
        return;
    }
    const now = audioContext.currentTime;
    for (let ix = 0; ix < player.chain.length; ix++) {
        const entry = player.chain[ix];
        if (entry.startTime === null) {
            if (ix === 0 && unitReady(entry.unit)) {
                entry.startTime = now + leadSeconds - entry.from;
            } else if (ix !== 0 && player.chain[ix - 1].startTime !== null && unitComplete(player.chain[ix - 1].unit) && unitReady(entry.unit)) {
                const previous = player.chain[ix - 1];
                entry.startTime = Math.max(previous.startTime + unitLength(previous.unit), now + 0.03);
            } else {
                break;
            }
        }
        const horizon = now + scheduleAheadSeconds;
        session.cache[entry.unit].forEach((cell, member) => {
            const offset = session.members[member].offset;
            while (entry.cursors[member] < cell.blocks.length && entry.startTime + offset + cell.blocks[entry.cursors[member]].start < horizon) {
                scheduleBlock(entry, member, cell.blocks[entry.cursors[member]]);
                entry.cursors[member]++;
            }
        });

        // Extend the chain with the next unit
        if (ix === player.chain.length - 1 && unitComplete(entry.unit) && entry.unit + 1 < session.units.length) {
            player.chain.push({'unit': entry.unit + 1, 'from': 0, 'startTime': null, 'cursors': session.members.map(() => 0)});
        }
    }

    // Drop finished units from the front of the chain
    while (player.chain.length > 1 && player.chain[1].startTime !== null && player.chain[1].startTime < now - 1) {
        player.chain.shift();
    }

    // The end of the text
    const last = player.chain[player.chain.length - 1];
    if (last.unit === session.units.length - 1 && last.startTime !== null && unitComplete(last.unit) &&
        now >= last.startTime + unitLength(last.unit) + 0.05) {
        stopSources();
        player.playing = false;
        player.unit = last.unit;
        player.time = unitLength(last.unit);
        setStatus('Done.');
    } else if (player.chain.every((entry) => entry.startTime === null || entry.startTime > now)) {
        setStatus('Synthesizing...');
    } else {
        setStatus(session.members.length > 1 ? 'The crowd is speaking...' : 'Speaking...');
    }
}


// Play from a position
function startPlayback(unit, time) {
    stopSources();
    ensureAudio();
    connectOutput();
    player.playing = true;
    player.chain = [{unit, 'from': time, 'startTime': null, 'cursors': session.members.map(() => 0)}];
    ensureSynthesis(unit);
    pump();
    updateControls();
}


// Pause, keeping the position
function pausePlayback() {
    if (player.playing) {
        const current = position();
        stopSources();
        player.playing = false;
        player.unit = current.unit;
        player.time = current.time;
        setStatus('Paused.');
    }
    updateControls();
}


// Move to a unit's start - playing from there if playing, else just preparing its audio
function moveTo(unit) {
    if (player.playing) {
        startPlayback(unit, 0);
    } else {
        player.unit = unit;
        player.time = 0;
        ensureSynthesis(unit);
        updateControls();
    }
}


// Speak - play the text from the start (reusing its audio if the settings are unchanged)
function onSpeak() {
    saveSettings();
    ensureAudio();
    if (ensureSession() !== null) {
        startPlayback(0, 0);
    }
    updateControls();
}


// Stop - stop playback (back to the start) and stop synthesizing
function onStop() {
    stopSources();
    player.playing = false;
    player.unit = 0;
    player.time = 0;
    cancelRuns();
    setStatus('Stopped.');
    updateControls();
}


// Play/Pause - pause, or play from the position (from the start, at the end). If the settings changed, a new session
// starts - at the same sentence, if the text is the same.
function onPlayPause() {
    if (player.playing) {
        pausePlayback();
        return;
    }
    saveSettings();
    ensureAudio();
    const {settings, key} = getSettings();
    if (session === null || session.key !== key) {
        const sameText = session !== null && session.hash === hashText(settings.text);
        const unit = sameText ? player.unit : 0;
        if (ensureSession() !== null) {
            startPlayback(Math.min(unit, session.units.length - 1), 0);
        }
        return;
    }
    const last = session.units.length - 1;
    const atEnd = player.unit === last && unitComplete(last) && player.time >= unitLength(last) - 0.05;
    startPlayback(atEnd ? 0 : player.unit, atEnd ? 0 : player.time);
}


// Skip by a percent of the text - to a sentence start: ahead at least to the next sentence; back to this sentence's start
// if more than a second into it, else to an earlier one
function onSkip(percent) {
    if (session === null) {
        return;
    }
    const current = position();
    const target = positionChar(current) + percent / 100 * textLength();
    let unit = unitAtChar(Math.max(0, Math.min(textLength() - 1, target)));
    if (percent > 0 && unit <= current.unit) {
        unit = current.unit + 1;
    } else if (percent < 0 && unit >= current.unit) {
        unit = current.time > 1 ? current.unit : current.unit - 1;
    }
    if (unit >= session.units.length) {
        // Past the end - stop at the end
        stopSources();
        player.playing = false;
        player.unit = session.units.length - 1;
        player.time = unitComplete(player.unit) ? unitLength(player.unit) : 0;
        setStatus('At the end.');
        updateControls();
        return;
    }
    moveTo(Math.max(0, unit));
}


// The position slider - show the target while dragging, and move there on release
function onSeekInput() {
    if (session === null) {
        return;
    }
    player.dragging = true;
    const char = Number(seekElement.value) / 1000 * textLength();
    percentElement.textContent = `${Math.floor(Number(seekElement.value) / 10)}%`;
    nowElement.textContent = session.units[unitAtChar(char)].text;
}

function onSeekChange() {
    player.dragging = false;
    if (session !== null) {
        moveTo(unitAtChar(Number(seekElement.value) / 1000 * textLength()));
    }
}


// Bookmarks - [{hash, char (the sentence start), percent, snippet, created}] in browser storage (may be unavailable)
const bookmarksKey = 'voice-synth-bookmarks';
function loadBookmarks() {
    try {
        const bookmarks = JSON.parse(localStorage.getItem(bookmarksKey));
        return Array.isArray(bookmarks) ? bookmarks : [];
    } catch {
        return [];
    }
}

function saveBookmarks(bookmarks) {
    try {
        localStorage.setItem(bookmarksKey, JSON.stringify(bookmarks));
    } catch {
        setStatus('Bookmarks can\'t be saved in this browser.', true);
    }
}


// Bookmark the current sentence
function onBookmark() {
    if (session === null) {
        return;
    }
    const current = position();
    const unit = session.units[current.unit];
    const bookmarks = loadBookmarks();
    if (bookmarks.some((bookmark) => bookmark.hash === session.hash && bookmark.char === unit.start)) {
        setStatus('That sentence is already bookmarked.');
        return;
    }
    bookmarks.push({
        'hash': session.hash, 'char': unit.start, 'percent': Math.floor(100 * unit.start / textLength()),
        'snippet': unit.text.length > 90 ? `${unit.text.slice(0, 87)}...` : unit.text, 'created': new Date().toISOString()
    });
    saveBookmarks(bookmarks);
    renderBookmarks();
    setStatus(`Bookmarked at ${Math.floor(100 * unit.start / textLength())}%.`);
}


// Go to a bookmark - in the current text, at its sentence
function goToBookmark(bookmark) {
    ensureAudio();
    if (ensureSession() === null) {
        return;
    }
    const unit = unitAtChar(bookmark.char);
    moveTo(unit);
    if (!player.playing) {
        setStatus(`At the bookmark (${bookmark.percent}%) - press Play.`);
    }
    updateControls();
}


// Show the current text's bookmarks
function renderBookmarks() {
    const hash = hashText(textElement.value);
    const bookmarks = loadBookmarks().filter((bookmark) => bookmark.hash === hash).sort((first, second) => first.char - second.char);
    bookmarksElement.replaceChildren(...bookmarks.map((bookmark) => {
        const item = document.createElement('li');
        const go = document.createElement('button');
        go.type = 'button';
        go.className = 'go';
        go.textContent = `${bookmark.percent}% - ${bookmark.snippet}`;
        go.title = `Go to: ${bookmark.snippet}`;
        go.addEventListener('click', () => goToBookmark(bookmark));
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'remove';
        remove.textContent = '×';
        remove.setAttribute('aria-label', `Delete the bookmark at ${bookmark.percent}%`);
        remove.addEventListener('click', () => {
            saveBookmarks(loadBookmarks().filter((other) => !(other.hash === bookmark.hash && other.char === bookmark.char)));
            renderBookmarks();
        });
        item.append(go, remove);
        return item;
    }));
    bookmarksSection.hidden = bookmarks.length === 0;
}


// Download WAV - synthesize the whole text (each crowd member dry), then mix and save
function onDownload() {
    const {settings, key} = getSettings();
    const members = getMembers(settings);
    ensureWorkers(members.length);
    download = {
        'id': nextRunId++, key, settings, members, 'chunks': members.map(() => []), 'done': members.map(() => false),
        'sampleRate': 44100, 'count': 0
    };
    members.forEach((member, ix) => {
        workers[ix].worker.postMessage({
            'id': download.id, 'member': ix, 'text': settings.text, 'voice': member.voice, 'rate': settings.rate * member.rate,
            'seed': member.seed, 'room': settings.room && members.length === 1, 'stream': false,
            'overrides': {'pitchScale': member.pitchScale, 'tractScale': member.tractScale}
        });
    });
    setStatus('Synthesizing the download...');
}


// Handle a download message
function onDownloadMessage(message) {
    if (message.type === 'error') {
        setStatus(message.message, true);
        download = null;
    } else if (message.type === 'chunk') {
        download.chunks[message.member].push(message.samples);
        download.sampleRate = message.sampleRate;
    } else if (message.type === 'done') {
        download.done[message.member] = true;
        download.count = message.count;
        if (download.done.every((done) => done)) {
            if (download.count === 0) {
                setStatus('Nothing to say - type some text.');
            } else {
                saveWav(download);
                setStatus('Saved.');
            }
            download = null;
        }
    }
}


// Mix a finished download's members to mono - each at its offset and level, then the room ambience (a crowd's members
// are synthesized dry), normalized
function mixDownload(finished) {
    const sampleRate = finished.sampleRate;
    const tracks = finished.chunks.map((chunks) => {
        const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
        const track = new Float32Array(length);
        let offset = 0;
        for (const chunk of chunks) {
            track.set(chunk, offset);
            offset += chunk.length;
        }
        return track;
    });
    if (tracks.length === 1) {
        return tracks[0];
    }
    const starts = finished.members.map((member) => Math.round(member.offset * sampleRate));
    const length = Math.max(...tracks.map((track, ix) => starts[ix] + track.length));
    const mix = new Float32Array(length);
    tracks.forEach((track, ix) => {
        const gain = finished.members[ix].gain;
        for (let jx = 0; jx < track.length; jx++) {
            mix[starts[ix] + jx] += gain * track[jx];
        }
    });
    return normalize(finished.settings.room ? convolve(mix, roomImpulse(sampleRate)) : mix, 0.89);
}


// Save a finished download as a WAV file
function saveWav(finished) {
    const samples = mixDownload(finished);
    const blob = new Blob([encodeWav(samples, finished.sampleRate)], {'type': 'audio/wav'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `speech-${finished.settings.crowd ? 'crowd' : finished.settings.voice}.wav`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}


// Start the crowd's workers when the crowd is chosen, so they're ready (dictionary loaded) by the next Speak
function onCrowd() {
    if (crowdElement.checked) {
        ensureWorkers(crowdMembers.length);
    }
}


// Remember the settings per viewer (browser storage may be unavailable)
function saveSettings() {
    try {
        localStorage.setItem('voice-synth', JSON.stringify(getSettings().settings));
    } catch {
        // Settings are a convenience only
    }
}

function loadSettings() {
    try {
        const settings = JSON.parse(localStorage.getItem('voice-synth'));
        if (settings !== null) {
            textElement.value = settings.text;
            document.querySelector(`input[name="voice"][value="${settings.voice === 'female' ? 'female' : 'male'}"]`).checked = true;
            rateElement.value = settings.rate;
            roomElement.checked = settings.room;
            crowdElement.checked = settings.crowd === true;
        }
    } catch {
        // Settings are a convenience only
    }
}


// Show the rate
function showRate() {
    rateValueElement.textContent = `${Number(rateElement.value).toFixed(2)}x`;
}


// Start up
loadSettings();
showRate();
ensureWorkers(1);
onCrowd();
renderBookmarks();
speakButton.addEventListener('click', onSpeak);
stopButton.addEventListener('click', onStop);
downloadButton.addEventListener('click', onDownload);
playPauseButton.addEventListener('click', onPlayPause);
bookmarkButton.addEventListener('click', onBookmark);
for (const button of skipButtons) {
    button.addEventListener('click', () => onSkip(Number(button.dataset.skip)));
}
seekElement.addEventListener('input', onSeekInput);
seekElement.addEventListener('change', onSeekChange);
rateElement.addEventListener('input', showRate);
crowdElement.addEventListener('change', onCrowd);
textElement.addEventListener('input', renderBookmarks);
for (const element of [textElement, rateElement, roomElement, crowdElement, ...document.querySelectorAll('input[name="voice"]')]) {
    element.addEventListener('change', saveSettings);
}
textElement.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        onSpeak();
    }
});
// A debugging summary - the units, those with cached audio, and whether synthesis is running
window.voiceSynthDebug = () => ({
    'units': session === null ? 0 : session.units.length,
    'cachedUnits': session === null ? 0 : session.cache.filter((cells) => cells.some((cell) => cell.blocks.length !== 0)).length,
    'synthesizing': isSynthesizing(), 'position': session === null ? null : position()
});
speakButton.disabled = false;
downloadButton.disabled = false;
setInterval(() => {
    pump();
    maintainWindow();
    updateControls();
}, 100);
updateControls();
