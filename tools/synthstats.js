// Synthesized speech statistics - speak the CMU ARCTIC prompts with a voice, write the WAVs and our exact phone labels in
// the ARCTIC layout (wav/, lab/, etc/txt.done.data), and measure them with tools/targets.py - so our voice and a human
// target speaker are measured identically - optionally comparing with a target's statistics
//
// node tools/synthstats.js [-v male|female|spec.json] [--overrides json] [--style json] [-r rate] [-s seed] [-n count]
//     [-j jobs] [--name name] [--out dir] [--prompts txt.done.data] [--target data/targets/arctic-bdl.json] [--ceiling hz]
//     [--no-analyze]
//
// The voice is a voice name or a JSON voice spec file (lib/voices.js); --overrides and --style are JSON.

import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';
import {existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {availableParallelism} from 'node:os';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import process from 'node:process';
import {spawnSync} from 'node:child_process';
import {speak} from '../lib/speak.js';
import {writeWav} from '../lib/wav.js';


const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const defaultPrompts = path.join(root, 'data', 'corpora', 'arctic', 'cmu_us_bdl_arctic', 'etc', 'txt.done.data');
const python = path.join(root, 'build', 'venv-analysis', 'bin', 'python');


// Read a Festival prompt file - [{id, text}]
export function readPrompts(promptsPath) {
    const prompts = [];
    for (const line of readFileSync(promptsPath, 'utf8').split('\n')) {
        const match = line.trim().match(/^\(\s*(\S+)\s+"(.*)"\s*\)$/u);
        if (match) {
            prompts.push({'id': match[1], 'text': match[2]});
        }
    }
    return prompts;
}


// An evenly spread, deterministic subset of at most count items
export function spread(items, count) {
    if (!count || items.length <= count) {
        return items;
    }
    const step = items.length / count;
    return Array.from({'length': count}, (_unused, ix) => items[Math.floor(ix * step)]);
}


// The ARCTIC (Festival radio) label of a segment - "pau" for pauses, "ax" for the reduced vowel (unstressed AH)
export function arcticPhone(segment) {
    if (segment.type === 'pause') {
        return 'pau';
    }
    if (segment.ph === 'AH' && segment.stress === 0) {
        return 'ax';
    }
    return segment.ph.toLowerCase();
}


// Segments (start and dur in ms) as an xwaves label file, the ARCTIC format - each line is a phone's end time
export function xwavesLabels(segments) {
    const lines = ['#'];
    for (const segment of segments) {
        lines.push(`${((segment.start + segment.dur) / 1000).toFixed(6)} 125 ${arcticPhone(segment)}`);
    }
    return `${lines.join('\n')}\n`;
}


// A prompt as a Festival prompt-file line
function promptLine({id, text}) {
    return `( ${id} "${text.replaceAll('"', '\\"')}" )`;
}


// The synthesizer's own formant targets (F1-F3, Hz) at the middle of each vowel at least 40 ms long (the duration
// threshold of tools/targets.py) - {phone: [[f1, f2, f3], ...]}
export function vowelTruth(segments, frames, frameMs = 5) {
    const truth = {};
    for (const segment of segments) {
        if (segment.type === 'vowel' && segment.dur >= 40) {
            const frame = frames[Math.min(frames.length - 1, Math.round((segment.start + segment.dur / 2) / frameMs))];
            const phone = arcticPhone(segment).toUpperCase();
            truth[phone] ??= [];
            truth[phone].push([frame.f1, frame.f2, frame.f3]);
        }
    }
    return truth;
}


// Synthesize prompts into the output directory (wav/ and lab/) - returns the total audio seconds and the vowel truth
function synthesizePrompts(prompts, options) {
    let seconds = 0;
    const truth = {};
    for (const {id, text, index} of prompts) {
        const {samples, sampleRate, segments, frames} = speak(text, options.voice,
            {'seed': options.seed + index, 'rate': options.rate, 'overrides': options.overrides, 'style': options.style});
        writeWav(path.join(options.out, 'wav', `${id}.wav`), samples, sampleRate);
        writeFileSync(path.join(options.out, 'lab', `${id}.lab`), xwavesLabels(segments));
        seconds += samples.length / sampleRate;
        for (const [phone, values] of Object.entries(vowelTruth(segments, frames))) {
            truth[phone] = [...(truth[phone] ?? []), ...values];
        }
    }
    return {seconds, truth};
}


// The median of numbers
function median(values) {
    const sorted = [...values].sort((first, second) => first - second);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}


// Merge synthesis results - total seconds, and the vowel truth summarized as per-vowel medians
function mergeResults(results) {
    const values = {};
    for (const result of results) {
        for (const [phone, tokens] of Object.entries(result.truth)) {
            values[phone] = [...(values[phone] ?? []), ...tokens];
        }
    }
    const truth = Object.fromEntries(Object.entries(values).map(([phone, tokens]) => [phone, {
        'n': tokens.length, 'f1': Math.round(median(tokens.map((token) => token[0]))),
        'f2': Math.round(median(tokens.map((token) => token[1]))), 'f3': Math.round(median(tokens.map((token) => token[2])))
    }]));
    return {'seconds': results.reduce((sum, result) => sum + result.seconds, 0), truth};
}


// Report the formant measurement error - Praat's measured vowel formants (tools/targets.py) against the synthesizer's
// own targets
function reportMeasurementError(statsPath, truth) {
    const measured = JSON.parse(readFileSync(statsPath, 'utf8')).formants ?? {};
    const lines = ['Formant measurement check - measured (Praat) minus the synthesizer\'s targets at the vowel middles, Hz:'];
    const errors = [[], [], []];
    for (const [phone, target] of Object.entries(truth).sort()) {
        const measure = measured[phone];
        if (measure) {
            const diffs = [measure.f1 - target.f1, measure.f2 - target.f2, measure.f3 - target.f3];
            diffs.forEach((diff, ix) => errors[ix].push(Math.abs(diff)));
            lines.push(`  ${phone.padEnd(3)} n=${String(target.n).padStart(4)}  F1 ${String(target.f1).padStart(5)} ` +
                `${diffs[0] >= 0 ? '+' : ''}${diffs[0]}  F2 ${String(target.f2).padStart(5)} ${diffs[1] >= 0 ? '+' : ''}${diffs[1]}  ` +
                `F3 ${String(target.f3).padStart(5)} ${diffs[2] >= 0 ? '+' : ''}${diffs[2]}`);
        }
    }
    lines.push(`  median absolute error: F1 ${median(errors[0])}, F2 ${median(errors[1])}, F3 ${median(errors[2])} Hz`);
    process.stdout.write(`${lines.join('\n')}\n\n`);
}


// Synthesize in worker threads, each taking an interleaved share of the prompts
async function synthesizeParallel(prompts, options, jobs) {
    const shares = Array.from({'length': jobs}, (_unused, ix) => prompts.filter((_prompt, jx) => jx % jobs === ix));
    const results = await Promise.all(shares.filter((share) => share.length).map((share) => new Promise((resolve, reject) => {
        const worker = new Worker(new URL(import.meta.url), {'workerData': {'prompts': share, options}});
        worker.on('message', resolve);
        worker.on('error', reject);
    })));
    return mergeResults(results);
}


// Parse the command line
function parseArgs(args) {
    const options = {
        'voice': 'male', 'overrides': {}, 'style': null, 'rate': 1, 'seed': 1, 'count': 0, 'jobs': Math.max(1, availableParallelism() - 1),
        'name': null, 'out': null, 'prompts': defaultPrompts, 'target': null, 'ceiling': null, 'analyze': true
    };
    for (let ix = 0; ix < args.length; ix++) {
        const arg = args[ix];
        if (arg === '-v') {
            const voice = args[++ix];
            options.voice = voice.endsWith('.json') ? JSON.parse(readFileSync(voice, 'utf8')) : voice;
        } else if (arg === '--overrides') {
            options.overrides = JSON.parse(args[++ix]);
        } else if (arg === '--style') {
            options.style = JSON.parse(args[++ix]);
        } else if (arg === '-r') {
            options.rate = Number(args[++ix]);
        } else if (arg === '-s') {
            options.seed = Number(args[++ix]);
        } else if (arg === '-n') {
            options.count = Number(args[++ix]);
        } else if (arg === '-j') {
            options.jobs = Number(args[++ix]);
        } else if (arg === '--name') {
            options.name = args[++ix];
        } else if (arg === '--out') {
            options.out = args[++ix];
        } else if (arg === '--prompts') {
            options.prompts = args[++ix];
        } else if (arg === '--target') {
            options.target = args[++ix];
        } else if (arg === '--ceiling') {
            options.ceiling = args[++ix];
        } else if (arg === '--no-analyze') {
            options.analyze = false;
        } else {
            throw new Error(`Unknown argument: ${arg}`);
        }
    }
    options.name ??= `synth-${typeof options.voice === 'string' ? options.voice : options.voice.name}`;
    options.out ??= path.join(root, 'out', 'synthstats', options.name);
    return options;
}


// Run tools/targets.py in the analysis environment
function runTargets(args) {
    if (!existsSync(python)) {
        throw new Error(`Missing ${python} - create it from requirements/analysis.txt (see the README)`);
    }
    const result = spawnSync(python, ['-I', path.join(root, 'tools', 'targets.py'), ...args], {'stdio': 'inherit'});
    if (result.status !== 0) {
        throw new Error(`tools/targets.py ${args[0]} failed`);
    }
}


// Main - synthesize, then measure and compare
async function main() {
    const options = parseArgs(process.argv.slice(2));
    const prompts = spread(readPrompts(options.prompts).map((prompt, index) => ({...prompt, index})), options.count);
    mkdirSync(path.join(options.out, 'wav'), {'recursive': true});
    mkdirSync(path.join(options.out, 'lab'), {'recursive': true});
    mkdirSync(path.join(options.out, 'etc'), {'recursive': true});
    writeFileSync(path.join(options.out, 'etc', 'txt.done.data'), `${prompts.map(promptLine).join('\n')}\n`);
    writeFileSync(path.join(options.out, 'etc', 'voice.json'),
        `${JSON.stringify({'voice': options.voice, 'overrides': options.overrides, 'style': options.style, 'rate': options.rate,
            'seed': options.seed})}\n`);

    const start = Date.now();
    const workerOptions = {'voice': options.voice, 'overrides': options.overrides, 'style': options.style, 'rate': options.rate,
        'seed': options.seed, 'out': options.out};
    const {seconds, truth} = options.jobs > 1 ? await synthesizeParallel(prompts, workerOptions, options.jobs)
        : mergeResults([synthesizePrompts(prompts, workerOptions)]);
    writeFileSync(path.join(options.out, 'etc', 'truth.json'), `${JSON.stringify(truth)}\n`);
    process.stderr.write(`${options.out} - ${prompts.length} prompts, ${seconds.toFixed(1)} s of speech, ` +
        `synthesized in ${((Date.now() - start) / 1000).toFixed(1)} s\n`);
    if (!options.analyze) {
        return;
    }

    const statsPath = `${options.out}.json`;
    const extractArgs = ['extract', '--dir', options.out, '--name', options.name, '-o', statsPath];
    if (options.ceiling) {
        extractArgs.push('--ceiling', options.ceiling);
    }
    runTargets(extractArgs);
    reportMeasurementError(statsPath, truth);
    if (options.target) {
        runTargets(['compare', statsPath, options.target]);
    }
}


if (isMainThread) {
    if (process.argv[1] && process.argv[1].endsWith('synthstats.js')) {
        await main();
    }
} else {
    parentPort.postMessage(synthesizePrompts(workerData.prompts, workerData.options));
}
