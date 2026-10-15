// The synthesis Web Worker - loads the pronouncing dictionary, then synthesizes text. In
// streaming mode, it posts small blocks of samples as they are synthesized so playback starts almost immediately (the
// page applies the room ambience); otherwise it posts each finished paragraph. The synthesis functions are exported so
// they also run under Node.
//
// Unit runs (the player): {'type': 'units', 'id', 'member', 'units': [{'index', 'text', 'trim'}], 'voice', 'rate', 'seed',
// 'overrides'} synthesizes the text's units (sentences) in order, posting {'type': 'unitBlock', 'id', 'member', 'unit',
// 'samples', 'sampleRate'} and {'type': 'unitDone', 'id', 'member', 'unit'}, then {'type': 'runDone', 'id', 'member'}.
// Each unit starts trim seconds into its audio (dropping redundant leading silence). A newer run or {'type': 'cancel'}
// supersedes a run between units. {'type': 'prepare', 'voice'} loads a voice ahead of playback. A voice named in
// neuralVoices is a speaker of the multi-speaker neural model (nn/js), a generated module (data/nn/multi.js) imported on
// first use.
//
// Whole-text requests (downloads): {'id', 'member', 'text', 'voice', 'rate', 'seed', 'room', 'stream', 'overrides'} - the
// member (a crowd member's index) is echoed back, and the overrides are voice parameters (e.g. pitchScale, tractScale)
// Messages out: {'type': 'ready'}, {'type': 'chunk', 'id', 'member', 'index', 'samples', 'sampleRate'},
// {'type': 'block', 'id', 'member', 'samples', 'sampleRate'}, {'type': 'done', 'id', 'member', 'count'},
// {'type': 'error', 'id', 'member', 'message'}

import {convolve, roomImpulse} from '../lib/room.js';
import {setDictionary, textToPhrases} from '../lib/lexicon.js';
import {createRenderer} from '../nn/js/render.js';
import {loadPackagedModel} from '../nn/js/package.js';
import {normalize} from '../lib/klatt.js';
import {speak} from '../lib/speak.js';
import {speakFrames} from '../nn/js/model.js';


// The neural voices - each a speaker (by key) of the multi-speaker model's speaker table
export const neuralVoices = {'neural-male': {'speaker': 'arctic-bdl'}, 'neural-female': {'speaker': 'arctic-slt'}};


// Neural playback - the output gain (the multi-speaker model's speech runs about 2 dB below the earlier single-speaker
// voices', whose gain was 1.05), and the pauses before a sentence and a paragraph (s, at normal speed - the model's own
// edge silences are short)
const neuralGain = 1.3;
const neuralSentencePause = 0.24;
const neuralParagraphPause = 0.4;


// A soft limiter - linear below 0.8, approaching 1 above
function softLimit(value) {
    const magnitude = Math.abs(value);
    if (magnitude <= 0.8) {
        return value;
    }
    return Math.sign(value) * (0.8 + 0.2 * Math.tanh((magnitude - 0.8) / 0.2));
}


// Load the multi-speaker neural model - imported and unpacked once, shared by every neural voice. importModule is the
// generated module's import (replaceable for tests).
let neuralModel = null;
export function loadNeuralModel(importModule = () => import('../data/nn/multi.js')) {
    if (neuralModel === null) {
        neuralModel = importModule().then((module) => loadPackagedModel(module));
        neuralModel.catch(() => {
            neuralModel = null;
        });
    }
    return neuralModel;
}


// A neural voice's speaker id in the model's speaker table
export function neuralSpeaker(model, voice) {
    const {speaker} = neuralVoices[voice];
    const entry = model.manifest.speakers.find((item) => item.key === speaker);
    if (!entry) {
        throw new Error(`The ${voice} voice's speaker (${speaker}) isn't in the model.`);
    }
    return entry.id;
}


// Stream a unit (sentence) with a neural model - a pause, then the unit's audio - calling onBlock(samples, sampleRate).
// rate is the speaking speed (1 is normal); paragraphStart lengthens the pause.
export function streamNeuralUnit(model, text, options, onBlock) {
    const {rate = 1, seed = 1, paragraphStart = false, speaker} = options;
    const sampleRate = 44100;
    const pause = paragraphStart ? neuralParagraphPause : neuralSentencePause;
    onBlock(new Float32Array(Math.round(sampleRate * pause / rate)), sampleRate);
    const {frames, eq} = speakFrames(model, text, {'rate': 1 / rate, speaker});
    const renderer = createRenderer({sampleRate, seed, eq, 'blockSize': 2048, 'onBlock': (block) => {
        const samples = new Float32Array(block.length);
        for (let ix = 0; ix < block.length; ix++) {
            samples[ix] = softLimit(block[ix] * neuralGain);
        }
        onBlock(samples, sampleRate);
    }});
    for (const frame of frames) {
        renderer.push([frame]);
    }
    renderer.end();
}


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
export function streamParagraphs(text, options, onBlock) {
    const {voice = 'male', rate = 1, seed = 1, overrides = {}} = options === undefined ? {} : options;
    const paragraphs = splitParagraphs(text);
    for (const paragraph of paragraphs) {
        speak(paragraph, voice, {seed, rate, overrides, 'onBlock': (samples) => onBlock(samples, 44100)});
    }
    return paragraphs.length;
}


// Stream a unit (sentence) of text, dropping its first trim seconds - calls onBlock(samples, sampleRate)
export function streamUnit(text, options, trim, onBlock) {
    const {voice = 'male', rate = 1, seed = 1, overrides = {}} = options === undefined ? {} : options;
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


// The worker message loop (only when running as a worker)
if (typeof globalThis.WorkerGlobalScope !== 'undefined' && self instanceof globalThis.WorkerGlobalScope) {
    // Fetch the dictionary on startup
    const ready = fetch(new URL('../data/cmudict.dict', import.meta.url)).then((response) => {
        if (!response.ok) {
            throw new Error(`Could not load the pronouncing dictionary (${response.status})`);
        }
        return response.text();
    }).then((text) => {
        setDictionary(text);
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
            const post = (samples, sampleRate) => {
                self.postMessage({'type': 'unitBlock', id, member, 'unit': unit.index, samples, sampleRate}, [samples.buffer]);
            };
            const unitSeed = seed * 1000 + unit.index;
            if (voice in neuralVoices) {
                const model = await loadNeuralModel();
                const speaker = neuralSpeaker(model, voice);
                streamNeuralUnit(model, unit.text, {rate, 'seed': unitSeed, 'paragraphStart': unit.trim === 0, speaker}, post);
            } else {
                streamUnit(unit.text, {voice, rate, 'seed': unitSeed, overrides}, unit.trim, post);
            }
            self.postMessage({'type': 'unitDone', id, member, 'unit': unit.index});
            // Yield, so a newer run or a cancel can arrive
            await new Promise((resolve) => {
                setTimeout(resolve, 0);
            });
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
        if (event.data.type === 'prepare') {
            // Load a neural voice ahead of playback
            if (event.data.voice in neuralVoices) {
                loadNeuralModel().catch((error) => {
                    self.postMessage({'type': 'error', 'id': null, 'member': 0, 'message': error.message});
                });
            }
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
