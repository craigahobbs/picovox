// The synthesis Web Worker - loads the pronouncing dictionary, then speaks the player's units (sentences) with the
// neural voices, posting small blocks of samples as they are synthesized so playback starts almost immediately. The
// synthesis functions are exported so they also run under Node.
//
// Messages in:
// - {'type': 'units', 'id', 'units': [{'index', 'text', 'paragraphStart'}], 'voice', 'rate'} speaks the units in order,
//   posting {'type': 'unitBlock', 'id', 'unit', 'samples', 'sampleRate'} and {'type': 'unitDone', 'id', 'unit'}, then
//   {'type': 'runDone', 'id'}. A newer run or a cancel supersedes a run between units.
// - {'type': 'prepare', 'voice'} loads the voice's model ahead of playback.
// - {'type': 'cancel'} stops the current run.
// Messages out also include {'type': 'ready'} (the dictionary is loaded) and {'type': 'error', 'id', 'message'}.
//
// A voice is a speaker of the multi-speaker neural model (nn/js), a generated module (data/nn/multi.js) imported on first
// use.

import {createRenderer} from '../nn/js/render.js';
import {loadPackagedModel} from '../nn/js/package.js';
import {setDictionary} from '../lib/lexicon.js';
import {speakFrames} from '../nn/js/model.js';


// The neural voices - each a speaker (by key) of the multi-speaker model's speaker table
export const neuralVoices = {'neural-male': {'speaker': 'arctic-bdl'}, 'neural-female': {'speaker': 'arctic-slt'}};


// Neural playback - the output gain (the multi-speaker model's speech runs about 2 dB below the earlier single-speaker
// voices', whose gain was 1.05), and the pauses before a sentence and a paragraph (s, at normal speed - the model's own
// edge silences are short)
export const neuralGain = 1.3;
const neuralSentencePause = 0.24;
const neuralParagraphPause = 0.4;


// A soft limiter - linear below 0.8, approaching 1 above
export function softLimit(value) {
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
    const runUnits = async ({id, units, voice, rate}) => {
        const run = ++currentRun;
        await ready;
        const model = await loadNeuralModel();
        const speaker = neuralSpeaker(model, voice);
        for (const unit of units) {
            if (run !== currentRun) {
                return;
            }
            const post = (samples, sampleRate) => {
                self.postMessage({'type': 'unitBlock', id, 'unit': unit.index, samples, sampleRate}, [samples.buffer]);
            };
            streamNeuralUnit(model, unit.text, {rate, 'seed': 1000 + unit.index, 'paragraphStart': unit.paragraphStart, speaker}, post);
            self.postMessage({'type': 'unitDone', id, 'unit': unit.index});
            // Yield, so a newer run or a cancel can arrive
            await new Promise((resolve) => {
                setTimeout(resolve, 0);
            });
        }
        if (run === currentRun) {
            self.postMessage({'type': 'runDone', id});
        }
    };

    self.onmessage = (event) => {
        if (event.data.type === 'units') {
            runUnits(event.data).catch((error) => {
                self.postMessage({'type': 'error', 'id': event.data.id, 'message': error.message});
            });
        } else if (event.data.type === 'prepare') {
            // Load the voice's model ahead of playback
            loadNeuralModel().catch((error) => {
                self.postMessage({'type': 'error', 'id': null, 'message': error.message});
            });
        } else if (event.data.type === 'cancel') {
            currentRun++;
        }
    };
}
