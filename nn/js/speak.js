// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// Speak text with a neural voice (nn/js/model.js) rendered by nn/js/render.js, under Node
//
//     node nn/js/speak.js -m data/nn/slt "Hello, world." -o hello.wav [-r 44100] [-s seed] [--params out.json]
//         [--transform '[{"pitchScale": 1.2}]'] [--pause 0.5] [--rate 1] [--bandwidth-scale 1 (the network's own)]
//     node nn/js/speak.js -m data/nn/slt --batch list.json    # [{text, wav, params, seed, transform}, ...]
//
// The multi-speaker, multi-style model (data/nn/multi) takes a voice and a style (batch items may set speaker,
// embedding, and style too):
//
//     node nn/js/speak.js -m data/nn/multi "Hello." -o hello.wav --speaker 5        # a speaker id (data/nn/speakers.json)
//         [--blend '{"5": 0.6, "12": 0.4}'] [--embedding '[...]']                  # a blend, or a raw embedding
//         [--style happy --intensity 0.7] [--style '{"happy": 0.5, "sad": 0.2}']   # an emotion, or a blend
//
// or a designed voice and style of data/nn/voices.json (nn/js/voices.js; batch items may set voice and voiceStyle):
//
//     node nn/js/speak.js -m data/nn/multi "Hello." -o hello.wav --voice narrator-f [--voice-style whisper]
//         [--voices data/nn/voices.json] [--rate 1 (the duration multiplier - 0.9 is 10% faster)]
//
// It prints one JSON line: the audio seconds, and the seconds spent loading, in the network, and rendering.
// --frames-only skips the rendering (the params file only).

/* eslint-disable no-console */

import {loadModel, speakFrames} from './model.js';
import {readFileSync, writeFileSync} from 'node:fs';
import {NAMES} from './params.js';
import {applyTransforms} from './transforms.js';
import {encodeWav} from '../../lib/wav.js';
import process from 'node:process';
import {render} from './render.js';
import {voiceOptions} from './voices.js';


// Parse the command line
function parseArgs(argv) {
    const args = {'model': 'data/nn/slt', 'text': null, 'wav': null, 'params': null, 'sampleRate': 44100, 'seed': 1, 'batch': null,
        'transform': null, 'pauseThreshold': 0.5, 'rate': 1, 'framesOnly': false, 'peak': 0.89, 'bandwidthScale': null,
        'speaker': undefined, 'embedding': undefined, 'style': undefined, 'intensity': 1,
        'voices': 'data/nn/voices.json', 'voice': undefined, 'voiceStyle': undefined};
    for (let ix = 0; ix < argv.length; ix++) {
        const arg = argv[ix];
        const take = () => argv[++ix];
        if (arg === '-m') {
            args.model = take();
        } else if (arg === '-o') {
            args.wav = take();
        } else if (arg === '-r') {
            args.sampleRate = Number(take());
        } else if (arg === '-s') {
            args.seed = Number(take());
        } else if (arg === '--params') {
            args.params = take();
        } else if (arg === '--batch') {
            args.batch = take();
        } else if (arg === '--transform') {
            args.transform = JSON.parse(take());
        } else if (arg === '--pause') {
            args.pauseThreshold = Number(take());
        } else if (arg === '--rate') {
            args.rate = Number(take());
        } else if (arg === '--bandwidth-scale') {
            args.bandwidthScale = Number(take());
        } else if (arg === '--peak') {
            args.peak = Number(take());
        } else if (arg === '--speaker') {
            args.speaker = Number(take());
        } else if (arg === '--blend') {
            args.speaker = JSON.parse(take());
        } else if (arg === '--embedding') {
            args.embedding = JSON.parse(take());
        } else if (arg === '--style') {
            const style = take();
            args.style = style.startsWith('{') ? JSON.parse(style) : style;
        } else if (arg === '--intensity') {
            args.intensity = Number(take());
        } else if (arg === '--voices') {
            args.voices = take();
        } else if (arg === '--voice') {
            args.voice = take();
        } else if (arg === '--voice-style') {
            args.voiceStyle = take();
        } else if (arg === '--frames-only') {
            args.framesOnly = true;
        } else {
            args.text = arg;
        }
    }
    return args;
}


// A Node buffer's bytes as an ArrayBuffer of their own
function toArrayBuffer(buffer) {
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}


// Peak-normalize samples (in place)
function normalize(samples, peak) {
    let max = 0;
    for (const sample of samples) {
        max = Math.max(max, Math.abs(sample));
    }
    const scale = max > 0 ? peak / max : 1;
    for (let ix = 0; ix < samples.length; ix++) {
        samples[ix] *= scale;
    }
    return samples;
}


function main() {
    const args = parseArgs(process.argv.slice(2));
    const defaultStyle = typeof args.style === 'string' ? {'emotion': args.style, 'intensity': args.intensity} : args.style;
    const startLoad = performance.now();
    const model = loadModel(JSON.parse(readFileSync(`${args.model}.json`, 'utf8')), toArrayBuffer(readFileSync(`${args.model}.bin`)));
    const loadSeconds = (performance.now() - startLoad) / 1000;
    const items = args.batch ? JSON.parse(readFileSync(args.batch, 'utf8')) :
        [{'text': args.text, 'wav': args.wav, 'params': args.params, 'seed': args.seed, 'transform': args.transform}];
    let voices = null;
    let networkSeconds = 0;
    let renderSeconds = 0;
    let audioSeconds = 0;
    for (const item of items) {
        const startNetwork = performance.now();
        let options = {'pauseThreshold': item.pauseThreshold ?? args.pauseThreshold,
            'rate': item.rate ?? args.rate, 'bandwidthScale': item.bandwidthScale ?? args.bandwidthScale ?? undefined,
            'speaker': item.speaker ?? args.speaker, 'embedding': item.embedding ?? args.embedding, 'style': item.style ?? defaultStyle};
        let voiceTransforms = [];
        const voiceId = item.voice ?? args.voice;
        if (voiceId !== undefined) {
            // A designed voice - its blend, style, and transforms (the rate is the listener's speed: 1 / the duration multiplier)
            voices ??= JSON.parse(readFileSync(args.voices, 'utf8'));
            const resolved = voiceOptions(model, voices, {'voice': voiceId, 'style': item.voiceStyle ?? args.voiceStyle,
                'rate': 1 / options.rate, 'pauseThreshold': options.pauseThreshold, 'bandwidthScale': options.bandwidthScale});
            // (an explicit model style - --style or the item's style - replaces the designed style's)
            options = {...resolved.options, 'style': options.style ?? resolved.options.style};
            voiceTransforms = resolved.transforms;
        }
        const spoken = speakFrames(model, item.text, options);
        networkSeconds += (performance.now() - startNetwork) / 1000;
        const frames = applyTransforms(applyTransforms(spoken.frames, voiceTransforms), item.transform ?? args.transform ?? []);
        if (item.params) {
            const round = (value) => Math.round(value * 10000) / 10000;
            const rows = frames.map((frame) => Array.from(frame.subarray(0, NAMES.length), round));
            const data = {'frameRate': 200, 'names': NAMES, 'frames': rows, 'eq': spoken.eq, 'sentences': spoken.sentences};
            writeFileSync(item.params, JSON.stringify(data));
        }
        audioSeconds += Math.max(0, frames.length - 1) / 200;
        if (!args.framesOnly && item.wav) {
            const startRender = performance.now();
            const samples = render(frames, {'sampleRate': args.sampleRate, 'seed': item.seed ?? args.seed, 'eq': spoken.eq});
            renderSeconds += (performance.now() - startRender) / 1000;
            writeFileSync(item.wav, encodeWav(normalize(samples, args.peak), args.sampleRate));
        }
    }
    console.log(JSON.stringify({'count': items.length, audioSeconds, loadSeconds, networkSeconds, renderSeconds,
        'realTimeNetwork': audioSeconds / networkSeconds, 'realTimeRender': renderSeconds > 0 ? audioSeconds / renderSeconds : null,
        'realTimeTotal': audioSeconds / (networkSeconds + renderSeconds)}));
}

main();
