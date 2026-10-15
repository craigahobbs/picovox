// Render a parameter file (nn/params.py's JSON format) to a WAV file (and optionally raw float32 samples)
//
//     node nn/js/rendercli.js params.json -o out.wav [-r 44100] [-s seed] [--raw out.f32] [--peak 0.89] [--block 1024]
//         [--push 7]    # stream: push the frames 7 at a time to createRenderer (the default renders them all at once)

/* eslint-disable no-console */

import {createRenderer, render} from './render.js';
import {readFileSync, writeFileSync} from 'node:fs';
import {encodeWav} from '../../lib/wav.js';
import process from 'node:process';


// Parse the command line
function parseArgs(argv) {
    const args = {'input': null, 'output': null, 'sampleRate': 44100, 'seed': 1, 'raw': null, 'peak': null, 'block': 4096, 'push': 0};
    for (let ix = 0; ix < argv.length; ix++) {
        const arg = argv[ix];
        if (arg === '-o') {
            args.output = argv[++ix];
        } else if (arg === '-r') {
            args.sampleRate = Number(argv[++ix]);
        } else if (arg === '-s') {
            args.seed = Number(argv[++ix]);
        } else if (arg === '--raw') {
            args.raw = argv[++ix];
        } else if (arg === '--peak') {
            args.peak = Number(argv[++ix]);
        } else if (arg === '--push') {
            args.push = Number(argv[++ix]);
        } else if (arg === '--block') {
            args.block = Number(argv[++ix]);
        } else {
            args.input = arg;
        }
    }
    return args;
}


function main() {
    const args = parseArgs(process.argv.slice(2));
    const data = JSON.parse(readFileSync(args.input, 'utf-8'));
    const frames = data.frames.map((row) => Object.fromEntries(data.names.map((name, ix) => [name, row[ix]])));
    const start = performance.now();
    let samples;
    if (args.push > 0) {
        // Stream - push the frames a few at a time, collecting the blocks as they're delivered
        const blocks = [];
        const renderer = createRenderer({'sampleRate': args.sampleRate, 'seed': args.seed, 'blockSize': args.block, 'eq': data.eq ?? null,
            'onBlock': (block) => blocks.push(block.slice())});
        for (let ix = 0; ix < frames.length; ix += args.push) {
            renderer.push(frames.slice(ix, ix + args.push));
        }
        samples = new Float32Array(renderer.end());
        let offset = 0;
        for (const block of blocks) {
            samples.set(block, offset);
            offset += block.length;
        }
    } else {
        samples = render(frames, {'sampleRate': args.sampleRate, 'seed': args.seed, 'blockSize': args.block, 'eq': data.eq ?? null});
    }
    const seconds = (performance.now() - start) / 1000;
    if (args.raw !== null) {
        writeFileSync(args.raw, new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
    }
    if (args.output !== null) {
        let out = samples;
        if (args.peak !== null) {
            let max = 0;
            for (const sample of samples) {
                max = Math.max(max, Math.abs(sample));
            }
            out = samples.map((sample) => sample * (max > 0 ? args.peak / max : 1));
        }
        writeFileSync(args.output, encodeWav(out, args.sampleRate));
    }
    const audioSeconds = samples.length / args.sampleRate;
    console.log(JSON.stringify({'samples': samples.length, 'seconds': audioSeconds, 'renderSeconds': seconds,
        'realtime': audioSeconds / seconds}));
}

main();
