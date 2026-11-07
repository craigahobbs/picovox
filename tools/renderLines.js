// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Render texts exactly as the package speaks them (lib/synthesis.js textBlocks - the voices, gain, and limiter at
// 24 kHz, seed 1000) to 16-bit WAV files - one JSON line per text in (picovox_train/quality.py's render jobs):
//
//     {"text", "wav", "voice" (a voice key), "edits" (optional - each sentence's frames edited before rendering, for
//      listening experiments: {"scale": {name: factor}, "add": {name: value}, "smooth": {name: milliseconds - a moving
//      average}})}
//
//     node tools/renderLines.js [lib/models/voiceModel.js] < items.jsonl

import {
    createSentenceRenderer, defaultSampleRate, modelLoader, paragraphPause, sentenceFrames, sentencePause, textBlocks
} from '../lib/synthesis.js';
import {speechText, splitSentences} from '../lib/sentences.js';
import {Buffer} from 'node:buffer';
import {NAMES} from '../lib/params.js';
import {concatBlocks} from '../lib/render.js';
import {pathToFileURL} from 'node:url';
import process from 'node:process';
import {resolve} from 'node:path';
import {toWav} from '../lib/index.js';
import {writeFileSync} from 'node:fs';


// The model - the package's, or another voice model module
const [, , modulePath] = process.argv;
const loadModel = modelLoader(
    () => import('../lib/models/g2pModel.js'),
    () => import(modulePath ? pathToFileURL(resolve(modulePath)).href : '../lib/models/voiceModel.js')
);
const model = await loadModel();

// The renderer's noise seed (a text's first sentence's - each sentence's is the next)
const seed = 1000;

// A parameter's index (an unknown name throws)
function parameterIndex(name) {
    const index = NAMES.indexOf(name);
    if (index < 0) {
        throw new Error(`Unknown parameter ${name}`);
    }
    return index;
}


// A sentence's frames, edited - each parameter scaled, added to, and smoothed (a centered moving average of the
// sentence's frames, 5 ms each)
function editFrames(frames, {scale = {}, add = {}, smooth = {}}) {
    const out = frames.map((frame) => Float64Array.from(frame));
    for (const [name, factor] of Object.entries(scale)) {
        const index = parameterIndex(name);
        out.forEach((frame) => {
            frame[index] *= factor;
        });
    }
    for (const [name, value] of Object.entries(add)) {
        const index = parameterIndex(name);
        out.forEach((frame) => {
            frame[index] += value;
        });
    }
    for (const [name, milliseconds] of Object.entries(smooth)) {
        const index = parameterIndex(name);
        const half = Math.round(milliseconds / 10);
        const column = out.map((frame) => frame[index]);
        out.forEach((frame, row) => {
            const window = column.slice(Math.max(0, row - half), row + half + 1);
            frame[index] = window.reduce((sum, value) => sum + value, 0) / window.length;
        });
    }
    return out;
}


// Speak text with edited frames - textBlocks's sentences, pauses, and seeds, each sentence's frames edited whole - the
// blocks
function editedBlocks(text, voice, edits) {
    const blocks = [];
    for (const sentence of splitSentences(text)) {
        const {chunks, eq} = sentenceFrames(model, speechText(sentence.text), {voice});
        const frames = editFrames([...chunks].flat(), edits);
        const pause = sentence.paragraphStart ? paragraphPause : sentencePause;
        const renderer = createSentenceRenderer({'onBlock': (block) => blocks.push(block), eq, 'seed': seed + sentence.index, pause});
        if (frames.length !== 0) {
            renderer.push(frames);
        }
        renderer.end();
    }
    return blocks;
}


// Read standard input (asynchronously - after the awaits above, a synchronous read of a pipe may find it not ready)
const chunks = [];
for await (const chunk of process.stdin) {
    chunks.push(chunk);
}

// Render each item - its audio as the package speaks it
const lines = Buffer.concat(chunks).toString('utf8').split('\n').filter((line) => line.trim() !== '');
for (const line of lines) {
    const {text, wav, voice, edits} = JSON.parse(line);
    const blocks = edits ? editedBlocks(text, voice, edits) : [...textBlocks(model, text, {voice, seed})];
    writeFileSync(wav, toWav(concatBlocks(blocks), defaultSampleRate));
}
