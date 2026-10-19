// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The synthesis pipeline - text to audio, as PicoShelf (the audiobook player) plays it: the text is split into
// sentences (lib/sentences.js), and each, after a short pause, is spoken by the multi-speaker neural model
// (lib/model.js - every word pronounced by the G2P network, lib/g2p.js) a fraction of a second at a time, rendered by
// the source-filter synthesizer (lib/render.js) at 24 kHz, with the output gain and a soft limiter.

import {speechText, splitSentences} from './sentences.js';
import {createRenderer} from './render.js';
import {loadG2p} from './g2p.js';
import {loadPackagedModel} from './package.js';
import {setPronouncer} from './lexicon.js';
import {streamFrames} from './model.js';


// The voices - each a speaker (by key) of the multi-speaker model's speaker table - and the default
export const voiceTable = [
    {'key': 'male', 'label': 'Male', 'sex': 'male', 'speaker': 'arctic-bdl'},
    {'key': 'female', 'label': 'Female', 'sex': 'female', 'speaker': 'arctic-slt'}
];
export const defaultVoice = 'male';


// The pauses before a sentence and a paragraph (s, at normal speed - the model's own edge silences are short)
export const sentencePause = 0.24;
export const paragraphPause = 0.4;


// The output sample rate (Hz) - the voices' training rate (their content stops below 10.5 kHz), at about half 44.1
// kHz's rendering cost
export const defaultSampleRate = 24000;


// The output gain (the multi-speaker model's speech runs about 2 dB below the earlier single-speaker voices', whose
// gain was 1.05), and the rendered block size (samples)
export const outputGain = 1.3;
export const blockSize = 2048;


// A soft limiter - linear below 0.8, approaching 1 above
export function softLimit(value) {
    const magnitude = Math.abs(value);
    if (magnitude <= 0.8) {
        return value;
    }
    return Math.sign(value) * (0.8 + 0.2 * Math.tanh((magnitude - 0.8) / 0.2));
}


// A model loader - a function returning a Promise of the voice model, loaded once: the G2P network (importG2p's
// module, lib/models/g2pModel.js) set as the text front end's pronouncer, and the multi-speaker model (importVoice's
// module, lib/models/voiceModel.js) unpacked. A failed load is retried on the next call.
export function modelLoader(importG2p, importVoice) {
    let loading = null;
    return () => {
        if (loading === null) {
            const pronouncer = importG2p().then((module) => loadG2p(module.g2pModel));
            const model = importVoice().then((module) => loadPackagedModel(module));
            loading = Promise.all([pronouncer, model]).then(([pronounce, voiceModel]) => {
                setPronouncer(pronounce);
                return voiceModel;
            });
            loading.catch(() => {
                loading = null;
            });
        }
        return loading;
    };
}


// A voice's speaker id in the model's speaker table
export function voiceSpeaker(model, voice) {
    const {speaker} = voiceTable.find((item) => item.key === voice);
    const entry = model.manifest.speakers.find((item) => item.key === speaker);
    if (!entry) {
        throw new Error(`The ${voice} voice's speaker (${speaker}) isn't in the model`);
    }
    return entry.id;
}


// A sentence's frames from the model - {chunks (an iterator of the frames, a fraction of a second at a time - the network
// runs as they're taken), eq (the voice's channel equalizer)}. options: speaker (the model's speaker id) and rate (the
// speaking speed - 1 is normal). With createSentenceRenderer, the network and the renderer can run apart (in two
// workers, say); sentenceBlocks is the two together.
export function sentenceFrames(model, text, {speaker, rate = 1} = {}) {
    const {chunks, eq} = streamFrames(model, text, {'rate': 1 / rate, speaker});
    return {chunks, eq};
}


// A sentence's renderer - onBlock(block) gets the sentence's audio (Float32Array blocks, with the output gain and the
// limiter): its pause (s, at normal speed) at once, then each block as it's final. push(frames) the sentence's frames (a
// chunk of sentenceFrames's) as they come, then end(). options: eq (sentenceFrames's), rate, seed (the noise seed),
// pause, and sampleRate.
export function createSentenceRenderer({onBlock, eq = null, rate = 1, seed = 1, pause = 0, sampleRate = defaultSampleRate}) {
    const pauseLength = Math.round(sampleRate * pause / rate);
    if (pauseLength > 0) {
        onBlock(new Float32Array(pauseLength));
    }
    return createRenderer({sampleRate, seed, eq, blockSize, 'onBlock': (block) => {
        const samples = new Float32Array(block.length);
        for (let ix = 0; ix < block.length; ix++) {
            samples[ix] = softLimit(block[ix] * outputGain);
        }
        onBlock(samples);
    }});
}


// Speak a sentence with the model - a pause (s, at normal speed), then the sentence's audio - as an iterator of blocks
// (Float32Array) made as they're taken. options: speaker (the model's speaker id), rate (the speaking speed - 1 is
// normal), seed (the renderer's noise seed), pause, and sampleRate.
export function* sentenceBlocks(model, text, {speaker, rate = 1, seed = 1, pause = 0, sampleRate = defaultSampleRate} = {}) {
    // The network streams its frames a fraction of a second at a time, each chunk rendered as it comes
    const {chunks, eq} = sentenceFrames(model, text, {speaker, rate});
    const ready = [];
    const renderer = createSentenceRenderer({'onBlock': (block) => ready.push(block), eq, rate, seed, pause, sampleRate});
    yield* ready.splice(0);
    for (const chunk of chunks) {
        renderer.push(chunk);
        yield* ready.splice(0);
    }
    renderer.end();
    yield* ready.splice(0);
}


// Speak text with the model - each sentence (lib/sentences.js) after its pause (a longer one before a paragraph), the
// sentence at index i with the noise seed seed + i - as an iterator of blocks (Float32Array). options: speaker, rate,
// seed, and sampleRate, as sentenceBlocks's.
export function* textBlocks(model, text, {speaker, rate = 1, seed = 1, sampleRate = defaultSampleRate} = {}) {
    for (const sentence of splitSentences(text)) {
        const pause = sentence.paragraphStart ? paragraphPause : sentencePause;
        yield* sentenceBlocks(model, speechText(sentence.text), {speaker, rate, 'seed': seed + sentence.index, pause, sampleRate});
    }
}
