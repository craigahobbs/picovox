// The speed of libs (lib/ and variants of it - tools/perf/variant.sh) without a JIT - each lib timed in its own process
// by the lab's timing (tools/perf/timing.js) over the default texts: the front end's time for their words, every one new
// (the G2P network - the first pass), and each stage's speed (times real time - the second pass). The engine is Node's
// interpreter (node --jitless: an iPhone without a JIT ran within 20% of it) or, with --jsc, JavaScriptCore's - Safari's
// engine, as macOS's jsc shell runs it (--useJIT=false; the JSC environment variable is another jsc's path). The two
// weigh a change differently (their profilers too: node --jitless --cpu-prof, jsc --sample), so time both.
//
//     node tools/perf/speed.js LIB [LIB ...] [--voices male] [--jit] [--jsc]
//
// A LIB is a variant's name (build/perf/variants/NAME/lib), a lib directory, or lib. For jsc, the lib is copied to
// build/perf/speed/ with the G2P's weights inflated (the shell has no DecompressionStream).

/* eslint-disable no-console */

import {cpSync, existsSync, mkdirSync, rmSync, writeFileSync} from 'node:fs';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {join, resolve} from 'node:path';
import {Buffer} from 'node:buffer';
import {execFileSync} from 'node:child_process';
import process from 'node:process';


// macOS's JavaScriptCore shell
const jscPath = '/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc';


// The timings' directory - the module each engine runs, and JavaScriptCore's copy of the lib
const speedDirectory = resolve('build/perf/speed');


// A model section with its weights inflated
async function inflated(section, {decodeBase64, inflateRaw}) {
    if (!section.deflate) {
        return section;
    }
    return {...section, 'deflate': false, 'weights': Buffer.from(await inflateRaw(decodeBase64(section.weights))).toString('base64')};
}


// A copy of a lib with the G2P's weights inflated - its directory
async function inflatedLib(directory) {
    const copy = join(speedDirectory, 'lib');
    cpSync(directory, copy, {'recursive': true, 'dereference': true});
    const base = pathToFileURL(resolve(directory)).href;
    const {g2pModel} = await import(`${base}/models/g2pModel.js`);
    const packageModule = await import(`${base}/package.js`);
    const model = await inflated({...g2pModel, 'context': await inflated(g2pModel.context, packageModule)}, packageModule);
    writeFileSync(join(copy, 'models', 'g2pModel.js'), `export const g2pModel = ${JSON.stringify(model)};\n`);
    return copy;
}


// Time one lib in a new process of an engine - the timing's result
async function timeLib(directory, voices, {jsc, jit}) {
    rmSync(speedDirectory, {'recursive': true, 'force': true});
    mkdirSync(speedDirectory, {'recursive': true});
    const options = {'lib': jsc ? await inflatedLib(directory) : resolve(directory), voices, 'passes': 2, 'stream': false};
    const entry = join(speedDirectory, 'entry.mjs');
    writeFileSync(entry, [
        `const {runTiming} = await import(${JSON.stringify(fileURLToPath(new URL('timing.js', import.meta.url)))});`,
        `(globalThis.print ?? console.log)(JSON.stringify(await runTiming(${JSON.stringify(options)})));`,
        ''
    ].join('\n'));
    const [engine, args] = jsc ? [process.env.JSC ?? jscPath, [...(jit ? [] : ['--useJIT=false']), '-m', entry]] :
        [process.execPath, [...(jit ? [] : ['--jitless']), entry]];
    return JSON.parse(execFileSync(engine, args, {'encoding': 'utf8'}).trim().split('\n').pop());
}


// A timing's totals over its texts and voices - the front end's time in the first pass (ms - every word new: the G2P
// network), and the last pass's audio (s) and network and render times (ms)
function stageTotals(result) {
    const total = (pass, key) => result.runs.filter((run) => run.pass === pass).reduce((sum, run) => sum + run.stages[key], 0);
    const last = result.passes - 1;
    return {
        'front': total(0, 'frontMs'), 'audio': total(last, 'audio'), 'network': total(last, 'networkMs'), 'render': total(last, 'renderMs')
    };
}


async function main(argv) {
    const option = (name, fallback) => {
        const index = argv.indexOf(name);
        return index >= 0 ? argv.splice(index, 2)[1] : fallback;
    };
    const voices = option('--voices', 'male').split(',');
    const engine = {'jsc': argv.includes('--jsc'), 'jit': argv.includes('--jit')};
    for (const name of argv.filter((arg) => !arg.startsWith('--'))) {
        const directory = existsSync(name) ? name : `build/perf/variants/${name}/lib`;
        const {front, audio, network, render} = stageTotals(await timeLib(directory, voices, engine));
        const both = 1000 * audio / (network + render);
        console.log(`${name.padEnd(12)} new words ${Math.round(front)} ms  network ${(1000 * audio / network).toFixed(2)}x  ` +
            `render ${(1000 * audio / render).toFixed(2)}x  both ${both.toFixed(2)}x real time`);
    }
}


main(process.argv.slice(2));
