// The synthesis Web Worker - loads the pronouncing dictionary and warms up the voices, then synthesizes text. In
// streaming mode, it posts small blocks of samples as they are synthesized so playback starts almost immediately (the
// page applies the room ambience); otherwise it posts each finished paragraph. The synthesis functions are exported so
// they also run under Node.
//
// Unit runs (the player): {'type': 'units', 'id', 'member', 'units': [{'index', 'text', 'trim'}], 'voice', 'rate', 'seed',
// 'overrides'} synthesizes the text's units (sentences) in order, posting {'type': 'unitBlock', 'id', 'member', 'unit',
// 'samples', 'sampleRate'} and {'type': 'unitDone', 'id', 'member', 'unit'}, then {'type': 'runDone', 'id', 'member'}.
// Each unit starts trim seconds into its audio (dropping redundant leading silence). A newer run or {'type': 'cancel'}
// supersedes a run between units.
//
// Whole-text requests (downloads): {'id', 'member', 'text', 'voice', 'rate', 'seed', 'room', 'stream', 'overrides'} - the
// member (a crowd member's index) is echoed back, and the overrides are voice parameters (e.g. pitchScale, tractScale)
// Messages out: {'type': 'ready'}, {'type': 'chunk', 'id', 'member', 'index', 'samples', 'sampleRate'},
// {'type': 'block', 'id', 'member', 'samples', 'sampleRate'}, {'type': 'done', 'id', 'member', 'count'},
// {'type': 'error', 'id', 'member', 'message'}

import {setDictionary, textToPhrases} from '../lib/lexicon.js';
import {speak} from '../lib/speak.js';
import {convolve, roomImpulse} from '../lib/room.js';
import {normalize} from '../lib/klatt.js';


// Split text into paragraphs (separated by blank lines), dropping those with nothing to say
export function splitParagraphs(text) {
    return text.split(/\n\s*\n/).map((paragraph) => paragraph.trim()).filter((paragraph) => textToPhrases(paragraph).length !== 0);
}


// Synthesize a passage - the same chain as tools/say.js: speak, then the optional room ambience
export function synthesizeText(text, {voice = 'male', rate = 1, seed = 1, room = false, overrides = {}} = {}) {
    const {samples: dry, sampleRate} = speak(text, voice, {seed, rate, overrides});
    const samples = room ? normalize(convolve(dry, roomImpulse(sampleRate)), 0.89) : dry;
    return {samples, sampleRate};
}


// Synthesize text paragraph by paragraph, calling onChunk(index, samples, sampleRate) for each - returns the paragraph count
export function synthesizeParagraphs(text, options, onChunk) {
    const paragraphs = splitParagraphs(text);
    paragraphs.forEach((paragraph, index) => {
        const {samples, sampleRate} = synthesizeText(paragraph, options);
        onChunk(index, samples, sampleRate);
    });
    return paragraphs.length;
}


// Stream text paragraph by paragraph, calling onBlock(samples, sampleRate) with each block as it is synthesized - returns
// the paragraph count
export function streamParagraphs(text, {voice = 'male', rate = 1, seed = 1, overrides = {}} = {}, onBlock) {
    const paragraphs = splitParagraphs(text);
    for (const paragraph of paragraphs) {
        speak(paragraph, voice, {seed, rate, overrides, 'onBlock': (samples) => onBlock(samples, 44100)});
    }
    return paragraphs.length;
}


// Stream a unit (sentence) of text, dropping its first trim seconds - calls onBlock(samples, sampleRate)
export function streamUnit(text, {voice = 'male', rate = 1, seed = 1, overrides = {}} = {}, trim, onBlock) {
    let toTrim = Math.round(trim * 44100);
    speak(text, voice, {seed, rate, overrides, 'onBlock': (samples) => {
        if (toTrim >= samples.length) {
            toTrim -= samples.length;
            return;
        }
        const kept = toTrim > 0 ? samples.slice(toTrim) : samples;
        toTrim = 0;
        onBlock(kept, 44100);
    }});
}


// Warm up the voices - the first synthesis with each voice calibrates its levels
export function warmUp() {
    for (const voice of ['male', 'female']) {
        speak('Hi.', voice);
    }
}


// The worker message loop (only when running as a worker)
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
    // Fetch the dictionary on startup
    const ready = fetch(new URL('../data/cmudict.dict', import.meta.url)).then((response) => {
        if (!response.ok) {
            throw new Error(`Could not load the pronouncing dictionary (${response.status})`);
        }
        return response.text();
    }).then((text) => {
        setDictionary(text);
        warmUp();
        self.postMessage({'type': 'ready'});
    });
    ready.catch((error) => {
        self.postMessage({'type': 'error', 'id': null, 'message': error.message});
    });

    // Unit runs - the latest run supersedes earlier ones between units
    let currentRun = 0;
    const runUnits = async ({id, member, units, voice, rate, seed, overrides = {}}) => {
        const run = ++currentRun;
        await ready;
        for (const unit of units) {
            if (run !== currentRun) {
                return;
            }
            streamUnit(unit.text, {voice, rate, 'seed': seed * 1000 + unit.index, overrides}, unit.trim, (samples, sampleRate) => {
                self.postMessage({'type': 'unitBlock', id, member, 'unit': unit.index, samples, sampleRate}, [samples.buffer]);
            });
            self.postMessage({'type': 'unitDone', id, member, 'unit': unit.index});
            // Yield, so a newer run or a cancel can arrive
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        if (run === currentRun) {
            self.postMessage({'type': 'runDone', id, member});
        }
    };

    // Synthesize each request, posting each paragraph's samples as it is done
    self.onmessage = async (event) => {
        if (event.data.type === 'units') {
            runUnits(event.data).catch((error) => {
                self.postMessage({'type': 'error', 'id': event.data.id, 'member': event.data.member, 'message': error.message});
            });
            return;
        }
        if (event.data.type === 'cancel') {
            currentRun++;
            return;
        }
        const {id, member = 0, text, voice, rate, seed, room, stream, overrides = {}} = event.data;
        try {
            await ready;
            const count = stream ?
                streamParagraphs(text, {voice, rate, seed, overrides}, (samples, sampleRate) => {
                    self.postMessage({'type': 'block', id, member, samples, sampleRate}, [samples.buffer]);
                }) :
                synthesizeParagraphs(text, {voice, rate, seed, room, overrides}, (index, samples, sampleRate) => {
                    self.postMessage({'type': 'chunk', id, member, index, samples, sampleRate}, [samples.buffer]);
                });
            self.postMessage({'type': 'done', id, member, count});
        } catch (error) {
            self.postMessage({'type': 'error', id, member, 'message': error.message});
        }
    };
}
