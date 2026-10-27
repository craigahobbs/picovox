// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Render texts exactly as the package speaks them (lib/synthesis.js textBlocks - the voices, styles, gain, and limiter
// at 24 kHz) to 16-bit WAV files, with the network's parameter frames - one JSON line per input line in, one out (used
// by picovox_train/quality.py):
//
//     {"text", "wav", "voice" (a voice key) or "speaker" (a speaker key of the model's speaker table), "style",
//      "seed", "frames" (optional - a file for the frames: float32, row-major, NAMES.length values a 5 ms frame),
//      "edits" (optional - each sentence's frames edited before rendering, for listening experiments: {"scale":
//      {name: factor}, "add": {name: value}, "smooth": {name: milliseconds - a moving average}})}
//  -> {"wav", "seconds", "frames", "frameCount"}
//
//     node tools/renderLines.js [lib/models/voiceModel.js] < items.jsonl > rendered.jsonl

import {
    createSentenceRenderer, defaultSampleRate, modelLoader, paragraphPause, sentenceFrames, sentencePause, textBlocks
} from '../lib/synthesis.js';
import {speechText, splitSentences} from '../lib/sentences.js';
import {Buffer} from 'node:buffer';
import {NAMES} from '../lib/params.js';
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
const speakerIds = new Map(model.manifest.speakers.map((speaker) => [speaker.key, speaker.id]));

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


// Speak text with edited frames - textBlocks's sentences, pauses, and seeds, each sentence's frames edited whole -
// {blocks, frames}
function editedRender(text, {voice, speaker, style, seed, edits}) {
    const blocks = [];
    const allFrames = [];
    for (const sentence of splitSentences(text)) {
        const {chunks, eq} = sentenceFrames(model, speechText(sentence.text), {voice, speaker, style});
        const frames = editFrames([...chunks].flat(), edits);
        const pause = sentence.paragraphStart ? paragraphPause : sentencePause;
        const renderer = createSentenceRenderer({'onBlock': (block) => blocks.push(block), eq, 'seed': seed + sentence.index, pause});
        if (frames.length !== 0) {
            renderer.push(frames);
        }
        renderer.end();
        allFrames.push(...frames);
    }
    return {blocks, 'frames': allFrames};
}


// Read standard input (asynchronously - after the awaits above, a synchronous read of a pipe may find it not ready)
const chunks = [];
for await (const chunk of process.stdin) {
    chunks.push(chunk);
}

// Render each item - its audio as the package speaks it, and the network's frames of each sentence (deterministic -
// the seed is the renderer's noise only)
const lines = Buffer.concat(chunks).toString('utf8').split('\n').filter((line) => line.trim() !== '');
for (const line of lines) {
    const {text, wav, voice, speaker: speakerKey, style, seed = 1000, frames: framesPath, edits} = JSON.parse(line);
    if (speakerKey !== undefined && !speakerIds.has(speakerKey)) {
        throw new Error(`Unknown speaker ${speakerKey}`);
    }
    const speaker = speakerKey === undefined ? undefined : speakerIds.get(speakerKey);
    const edited = edits ? editedRender(text, {voice, speaker, style, seed, edits}) : null;
    const blocks = edited ? edited.blocks : [...textBlocks(model, text, {voice, speaker, style, seed})];
    const samples = new Float32Array(blocks.reduce((sum, block) => sum + block.length, 0));
    blocks.reduce((offset, block) => {
        samples.set(block, offset);
        return offset + block.length;
    }, 0);
    writeFileSync(wav, toWav(samples, defaultSampleRate));
    let frameCount = null;
    if (framesPath !== undefined) {
        const rows = edited ? edited.frames : [];
        for (const sentence of edited ? [] : splitSentences(text)) {
            for (const chunk of sentenceFrames(model, speechText(sentence.text), {voice, speaker, style}).chunks) {
                rows.push(...chunk);
            }
        }
        const values = new Float32Array(rows.length * NAMES.length);
        rows.forEach((row, index) => values.set(row, index * NAMES.length));
        writeFileSync(framesPath, values);
        frameCount = rows.length;
    }
    const result = {wav, 'seconds': samples.length / defaultSampleRate, 'frames': framesPath ?? null, frameCount};
    process.stdout.write(`${JSON.stringify(result)}\n`);
}
