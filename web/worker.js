// The synthesis Web Worker - loads the G2P network (letters to phonemes, for every word - no dictionary), then speaks
// the player's units (sentences) with the neural voices, posting small blocks of samples as they are synthesized so
// playback starts almost immediately. The synthesis functions are exported so they also run under Node.
//
// Messages in:
// - {'type': 'units', 'id', 'units': [{'index', 'text', 'paragraphStart'}], 'voice', 'rate'} speaks the units in order,
//   posting {'type': 'unitBlock', 'id', 'unit', 'samples', 'sampleRate'} and {'type': 'unitDone', 'id', 'unit'}, then
//   {'type': 'runDone', 'id'}. A newer run or a cancel supersedes a run between units.
// - {'type': 'prepare', 'voice'} loads the voice's model ahead of playback.
// - {'type': 'cancel'} stops the current run.
// Messages out also include {'type': 'ready'} (the G2P network is loaded) and {'type': 'error', 'id', 'message'}.
//
// A voice is a speaker of the multi-speaker neural model (nn/js), a generated module (lib/models/voiceModel.js) imported on first
// use.

import {setPronouncer, textToPhrases} from '../lib/lexicon.js';
import {createRenderer} from '../nn/js/render.js';
import {loadG2p} from '../lib/g2p.js';
import {loadPackagedModel} from '../nn/js/package.js';
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
export function loadNeuralModel(importModule = () => import('../lib/models/voiceModel.js'), trace = () => null) {
    if (neuralModel === null) {
        neuralModel = importModule().then((module) => {
            trace('voice model imported');
            return loadPackagedModel(module);
        }).then((model) => {
            trace('voice model unpacked');
            return model;
        });
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
    const networkStart = performance.now();
    const {frames, eq} = speakFrames(model, text, {'rate': 1 / rate, speaker});
    options.onNetwork?.(performance.now() - networkStart);
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
    // Trace the worker's milestones (ms since it started) - the engine's startup timeline
    const workerStart = performance.now();
    const trace = (event) => {
        self.postMessage({'type': 'trace', event, 'ms': Math.round(performance.now() - workerStart)});
    };
    const loadModel = () => loadNeuralModel(undefined, trace);
    trace('started');

    // A JavaScript speed check - a fixed arithmetic workload (a few milliseconds with a JIT; far slower without one, as
    // in iOS Lockdown Mode)
    const speedStart = performance.now();
    const values = new Float64Array(4096);
    let sum = 0;
    for (let pass = 0; pass < 500; pass++) {
        for (let ix = 0; ix < values.length; ix++) {
            values[ix] = values[ix] * 0.999 + ix;
            sum += values[ix];
        }
    }
    trace(`speed check: ${Math.round(performance.now() - speedStart)} ms (checksum ${Math.round(sum) % 1000})`);

    // Load the G2P network on startup - every word's pronunciation
    const ready = import('../lib/models/g2pModel.js').then(({g2pModel}) => {
        trace('G2P model imported');
        return loadG2p(g2pModel);
    }).then((pronounce) => {
        setPronouncer(pronounce);
        trace('G2P model unpacked');
        self.postMessage({'type': 'ready'});
    });
    ready.catch((error) => {
        self.postMessage({'type': 'error', 'id': null, 'message': `${error.message} (${error.stack})`});
    });

    // Unit runs - the latest run supersedes earlier ones between units
    let currentRun = 0;
    const runUnits = async ({id, units, voice, rate}) => {
        const run = ++currentRun;
        await ready;
        const model = await loadModel();
        const speaker = neuralSpeaker(model, voice);
        for (const unit of units) {
            if (run !== currentRun) {
                return;
            }
            const unitStart = performance.now();
            let blocks = 0;
            const post = (samples, sampleRate) => {
                if (blocks === 0) {
                    trace(`unit ${unit.index} first block after ${Math.round(performance.now() - unitStart)} ms`);
                }
                blocks += 1;
                self.postMessage({'type': 'unitBlock', id, 'unit': unit.index, samples, sampleRate}, [samples.buffer]);
            };
            // Pronounce the unit's words first (cached), to time the G2P apart from the voice network
            const words = textToPhrases(unit.text).reduce((count, phrase) => count + phrase.words.length, 0);
            trace(`unit ${unit.index}: G2P ${words} words in ${Math.round(performance.now() - unitStart)} ms`);
            const onNetwork = (ms) => trace(`unit ${unit.index}: voice network in ${Math.round(ms)} ms`);
            const unitOptions = {rate, 'seed': 1000 + unit.index, 'paragraphStart': unit.paragraphStart, speaker, onNetwork};
            streamNeuralUnit(model, unit.text, unitOptions, post);
            trace(`unit ${unit.index} done in ${Math.round(performance.now() - unitStart)} ms (${blocks} blocks)`);
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
                self.postMessage({'type': 'error', 'id': event.data.id, 'message': `${error.message} (${error.stack})`});
            });
        } else if (event.data.type === 'prepare') {
            // Load the voice's model ahead of playback
            loadModel().catch((error) => {
                self.postMessage({'type': 'error', 'id': null, 'message': `${error.message} (${error.stack})`});
            });
        } else if (event.data.type === 'cancel') {
            currentRun++;
        }
    };
}
