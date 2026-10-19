// The speed of libs (lib/ and variants of it - tools/perf/variant.sh) under Node's interpreter - each lib timed in its own
// process (node --jitless: an iPhone without a JIT ran within 20% of it) by the lab's timing (tools/perf/timing.js), its
// second pass: each stage's speed (times real time) over the default texts
//
//     node tools/perf/speed.js LIB [LIB ...] [--voices male] [--jit]
//
// A LIB is a variant's name (build/perf/variants/NAME/lib), a lib directory, or lib.

/* eslint-disable no-console */

import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import process from 'node:process';
import {resolve} from 'node:path';
import {runTiming} from './timing.js';


// Time one lib (in this process) - print its result's JSON
async function timeOne(directory, voices) {
    const {pathToFileURL} = await import('node:url');
    const result = await runTiming({'lib': pathToFileURL(resolve(directory)).href, voices, 'passes': 2, 'stream': false});
    const last = result.runs.filter((run) => run.pass === 1);
    const audio = last.reduce((sum, run) => sum + run.stages.audio, 0);
    const network = last.reduce((sum, run) => sum + run.stages.networkMs, 0);
    const render = last.reduce((sum, run) => sum + run.stages.renderMs, 0);
    console.log(JSON.stringify({audio, network, render}));
}


async function main(argv) {
    const option = (name, fallback) => {
        const index = argv.indexOf(name);
        return index >= 0 ? argv.splice(index, 2)[1] : fallback;
    };
    const voices = option('--voices', 'male').split(',');
    const one = option('--one', null);
    if (one) {
        await timeOne(one, voices);
        return;
    }
    const jit = argv.includes('--jit');
    for (const name of argv.filter((arg) => !arg.startsWith('--'))) {
        const directory = existsSync(name) ? name : `build/perf/variants/${name}/lib`;
        const output = execFileSync(process.execPath, [...(jit ? [] : ['--jitless']), fileURLToPath(import.meta.url),
            '--one', directory, '--voices', voices.join(',')], {'encoding': 'utf8'});
        const {audio, network, render} = JSON.parse(output.trim().split('\n').pop());
        const both = 1000 * audio / (network + render);
        console.log(`${name.padEnd(12)} network ${(1000 * audio / network).toFixed(2)}x  render ${(1000 * audio / render).toFixed(2)}x  ` +
            `both ${both.toFixed(2)}x real time`);
    }
}


main(process.argv.slice(2));
