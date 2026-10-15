// Speak text with the rule voice (lib/speak.js) and write its parameter frames (nn/js/bridge.js), its Klatt audio,
// and its rendering by nn/js/render.js - the rule voice through the parameter set
//
//     node nn/js/speakparams.js -v male [-s seed] [--params out.json] [--klatt klatt.wav] [--render render.wav]
//         [--rate 44100] [--ob1 0.5] "text"
//     node nn/js/speakparams.js -v female --batch list.json    # [{text, params, klatt, klattAlt, render}, ...]
//
// The WAVs are peak-normalized to 0.89, as speak() does. klattAlt is lib/klatt.js on the same frames with another
// synthesis seed (jitter, shimmer, and noise realizations only) - the floor of any rendering's distance from klatt.

/* eslint-disable no-console */

import {readFileSync, writeFileSync} from 'node:fs';
import {NAMES} from './params.js';
import {encodeWav} from '../../lib/wav.js';
import {paramsFromSpeak} from './bridge.js';
import process from 'node:process';
import {render} from './render.js';
import {speak} from '../../lib/speak.js';
import {synthesize} from '../../lib/klatt.js';


// Parse the command line
function parseArgs(argv) {
    const args = {'voice': 'male', 'seed': 1, 'params': null, 'klatt': null, 'render': null, 'rate': 44100, 'ob1': 0.5, 'batch': null,
        'text': null, 'renderSeed': null};
    for (let ix = 0; ix < argv.length; ix++) {
        const arg = argv[ix];
        const take = () => argv[++ix];
        if (arg === '-v') {
            args.voice = take();
        } else if (arg === '-s') {
            args.seed = Number(take());
        } else if (arg === '--params') {
            args.params = take();
        } else if (arg === '--klatt') {
            args.klatt = take();
        } else if (arg === '--render') {
            args.render = take();
        } else if (arg === '--rate') {
            args.rate = Number(take());
        } else if (arg === '--ob1') {
            args.ob1 = Number(take());
        } else if (arg === '--render-seed') {
            args.renderSeed = Number(take());
        } else if (arg === '--batch') {
            args.batch = take();
        } else {
            args.text = arg;
        }
    }
    return args;
}


// Peak-normalize samples
function normalized(samples, peak = 0.89) {
    let max = 0;
    for (const sample of samples) {
        max = Math.max(max, Math.abs(sample));
    }
    return samples.map((sample) => sample * (max > 0 ? peak / max : 1));
}


// One text - returns timing
function one(text, item, args) {
    const voice = args.voice.endsWith('.json') ? JSON.parse(readFileSync(args.voice, 'utf-8')) : args.voice;
    const startSpeak = performance.now();
    const result = speak(text, voice, {'seed': item.seed ?? args.seed});
    const speakSeconds = (performance.now() - startSpeak) / 1000;
    const frames = paramsFromSpeak(result, {'openB1Weight': args.ob1});
    if (item.params) {
        const rows = frames.map((frame) => NAMES.map((name) => Math.round(frame[name] * 10000) / 10000));
        writeFileSync(item.params, JSON.stringify({'frameRate': 200, 'names': NAMES, 'frames': rows}));
    }
    if (item.klatt) {
        writeFileSync(item.klatt, encodeWav(result.samples, result.sampleRate));
    }
    if (item.klattAlt) {
        // lib/speak.js's synthesize() call, with another seed
        const {voice: resolved} = result;
        const higherFormants = resolved.higherFormants.map(([frequency, bandwidth]) => [frequency * resolved.tractScale, bandwidth]);
        const samples = synthesize(result.frames, {'frameMs': 5, 'sampleRate': result.sampleRate, 'seed': (item.seed ?? args.seed) + 1000,
            'jitter': resolved.jitter, 'shimmer': resolved.shimmer, 'noiseModulation': resolved.noiseModulation,
            'sourceShelfHz': resolved.shelfHz, 'sourceShelfDb': resolved.shelfDb, higherFormants});
        writeFileSync(item.klattAlt, encodeWav(normalized(samples), result.sampleRate));
    }
    let renderSeconds = null;
    if (item.render) {
        const startRender = performance.now();
        const samples = render(frames, {'sampleRate': args.rate, 'seed': item.renderSeed ?? args.renderSeed ?? item.seed ?? args.seed});
        renderSeconds = (performance.now() - startRender) / 1000;
        writeFileSync(item.render, encodeWav(normalized(samples), args.rate));
    }
    return {'seconds': result.samples.length / result.sampleRate, speakSeconds, renderSeconds};
}


function main() {
    const args = parseArgs(process.argv.slice(2));
    const items = args.batch ? JSON.parse(readFileSync(args.batch, 'utf-8')) :
        [{'text': args.text, 'params': args.params, 'klatt': args.klatt, 'render': args.render}];
    const timings = items.map((item) => one(item.text, item, args));
    const total = (key) => timings.reduce((sum, timing) => sum + (timing[key] ?? 0), 0);
    console.log(JSON.stringify({'count': items.length, 'audioSeconds': total('seconds'), 'speakSeconds': total('speakSeconds'),
        'renderSeconds': total('renderSeconds')}));
}

main();
