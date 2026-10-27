// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Speak text with a neural voice (lib/model.js) rendered by lib/render.js, under Node - every word pronounced by the G2P
// network (lib/g2p.js), as the package pronounces
//
// The model is an export (picovox_train.export's MODEL.json and MODEL.bin) or a packaged module (lib/models/voiceModel.js).
// The multi-speaker, multi-style model takes a speaker and a style (--rate is the duration multiplier - 0.9 is 10% faster):
//
//     node nn/js/speak.js -m build/voice "Hello, world." -o hello.wav [-r 44100] [-s seed] [--params out.json] [--rate 1]
//     node nn/js/speak.js -m build/voice --batch list.json    # [{text, wav, params, seed, speaker, style, rate}, ...]
//     node nn/js/speak.js -m build/voice "Hello." -o hello.wav --speaker 5            # a speaker id (data/nn/speakers.json)
//         [--blend '{"5": 0.6, "12": 0.4}']                                        # or a blend
//         [--style happy --intensity 0.7] [--style '{"happy": 0.5, "sad": 0.2}']   # an emotion, or a blend
//
// It prints one JSON line: the audio seconds, and the seconds spent loading, in the network, and rendering.

/* eslint-disable no-console */

import {readFileSync, writeFileSync} from 'node:fs';
import {NAMES} from '../../lib/params.js';
import {encodeWav} from '../../lib/wav.js';
import {g2pModel} from '../../lib/models/g2pModel.js';
import {loadG2p} from '../../lib/g2p.js';
import {loadVoice} from './loadVoice.js';
import process from 'node:process';
import {render} from '../../lib/render.js';
import {setPronouncer} from '../../lib/lexicon.js';
import {speakFrames} from '../../lib/model.js';


// The rendered audio's peak
const PEAK = 0.89;


// Parse the command line
function parseArgs(argv) {
    const args = {'model': null, 'text': null, 'wav': null, 'params': null, 'sampleRate': 44100, 'seed': 1, 'batch': null,
        'rate': 1, 'speaker': undefined, 'style': undefined, 'intensity': 1};
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
        } else if (arg === '--rate') {
            args.rate = Number(take());
        } else if (arg === '--speaker') {
            args.speaker = Number(take());
        } else if (arg === '--blend') {
            args.speaker = JSON.parse(take());
        } else if (arg === '--style') {
            const style = take();
            args.style = style.startsWith('{') ? JSON.parse(style) : style;
        } else if (arg === '--intensity') {
            args.intensity = Number(take());
        } else {
            args.text = arg;
        }
    }
    return args;
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


async function main() {
    const args = parseArgs(process.argv.slice(2));
    if (args.model === null) {
        throw new Error('usage: node nn/js/speak.js -m MODEL (an export without its .json and .bin, or a packaged module) TEXT -o WAV');
    }
    const defaultStyle = typeof args.style === 'string' ? {'emotion': args.style, 'intensity': args.intensity} : args.style;
    const startLoad = performance.now();
    const model = await loadVoice(args.model);
    const loadSeconds = (performance.now() - startLoad) / 1000;
    const items = args.batch ? JSON.parse(readFileSync(args.batch, 'utf8')) :
        [{'text': args.text, 'wav': args.wav, 'params': args.params, 'seed': args.seed}];
    let networkSeconds = 0;
    let renderSeconds = 0;
    let audioSeconds = 0;
    for (const item of items) {
        const startNetwork = performance.now();
        const options = {'rate': item.rate ?? args.rate, 'speaker': item.speaker ?? args.speaker, 'style': item.style ?? defaultStyle};
        const {frames, eq, sentences} = speakFrames(model, item.text, options);
        networkSeconds += (performance.now() - startNetwork) / 1000;
        if (item.params) {
            const round = (value) => Math.round(value * 10000) / 10000;
            const rows = frames.map((frame) => Array.from(frame.subarray(0, NAMES.length), round));
            writeFileSync(item.params, JSON.stringify({'frameRate': 200, 'names': NAMES, 'frames': rows, eq, sentences}));
        }
        audioSeconds += Math.max(0, frames.length - 1) / 200;
        if (item.wav) {
            const startRender = performance.now();
            const samples = render(frames, {'sampleRate': args.sampleRate, 'seed': item.seed ?? args.seed, eq});
            renderSeconds += (performance.now() - startRender) / 1000;
            writeFileSync(item.wav, encodeWav(normalize(samples, PEAK), args.sampleRate));
        }
    }
    console.log(JSON.stringify({'count': items.length, audioSeconds, loadSeconds, networkSeconds, renderSeconds,
        'realTimeNetwork': audioSeconds / networkSeconds, 'realTimeRender': renderSeconds > 0 ? audioSeconds / renderSeconds : null,
        'realTimeTotal': audioSeconds / (networkSeconds + renderSeconds)}));
}

// Every word pronounced by the G2P network, as the package pronounces
setPronouncer(await loadG2p(g2pModel));
await main();
