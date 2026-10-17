// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// make train's listening samples (tools/trainsamples.js) - its command line, its file names, the UTMOS scores, and
// TRAIN_DIR/samples/index.json, updated under a lock and replaced atomically

import {basename, dirname, join} from 'node:path';
import {mkdirSync, readFileSync, renameSync, rmSync, rmdirSync, statSync, writeFileSync} from 'node:fs';
import process from 'node:process';


// Parse the command line - {stage, checkpoint, model, run, out, settings}
export function parseArgs(argv) {
    const args = {'stage': null, 'checkpoint': null, 'model': null, 'run': null, 'out': null, 'settings': 'data/nn/train.json'};
    const options = {'--stage': 'stage', '--checkpoint': 'checkpoint', '--model': 'model', '--run': 'run', '--out': 'out',
        '--settings': 'settings'};
    for (let ix = 0; ix < argv.length; ix++) {
        if (!(argv[ix] in options) || ix + 1 >= argv.length) {
            throw new Error(`unknown or incomplete argument ${argv[ix]}`);
        }
        args[options[argv[ix]]] = argv[++ix];
    }
    if (args.stage === null || args.out === null || (args.checkpoint === null) === (args.model === null)) {
        throw new Error('usage: node tools/trainsamples.js --stage STAGE (--checkpoint PATH [--run RUN] | --model MODULE) --out DIR');
    }
    if (args.checkpoint !== null && args.run === null) {
        args.run = basename(dirname(args.checkpoint));
    }
    return args;
}


// A sample's WAV file name
export function sampleFile(stage, speaker, text) {
    return `${stage}-${speaker}-${text}.wav`;
}


// UTMOS scores from picovox_train/listeners.py's output (JSON lines) - {file name: mos}
export function parseScores(output) {
    const scores = {};
    for (const line of output.split('\n')) {
        let record;
        try {
            record = JSON.parse(line);
        } catch {
            continue;
        }
        if (record !== null && typeof record.file === 'string') {
            scores[basename(record.file)] = record.mos ?? null;
        }
    }
    return scores;
}


// An index with a stage's samples replaced - the other stages' entries kept, in their order
export function mergeIndex(index, stage, entries) {
    return [...(Array.isArray(index) ? index : []).filter((entry) => entry.stage !== stage), ...entries];
}


// Run a function holding a lock directory (mkdir is atomic) - waits up to timeoutMs for another holder, and takes over
// a lock older than staleMs (a holder that died)
export function withLock(lockPath, callback, {timeoutMs = 120000, staleMs = 300000, now = Date.now, wait = sleep} = {}) {
    const start = now();
    for (;;) {
        try {
            mkdirSync(lockPath);
            break;
        } catch (error) {
            if (error.code !== 'EEXIST') {
                throw error;
            }
            let age = 0;
            try {
                age = now() - statSync(lockPath).mtimeMs;
            } catch {
                // Released meanwhile
            }
            if (age > staleMs) {
                rmSync(lockPath, {'recursive': true, 'force': true});
            } else if (now() - start > timeoutMs) {
                throw new Error(`timed out waiting for ${lockPath}`, {'cause': error});
            } else {
                wait(100);
            }
        }
    }
    try {
        return callback();
    } finally {
        rmdirSync(lockPath);
    }
}


// Sleep synchronously
function sleep(ms) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}


// Add a stage's samples to OUT/index.json (under the lock; the new index written to a temporary file, then renamed)
export function updateIndex(out, stage, entries) {
    const indexPath = join(out, 'index.json');
    withLock(join(out, 'index.lock'), () => {
        let index = [];
        try {
            index = JSON.parse(readFileSync(indexPath, 'utf8'));
        } catch {
            // No index yet
        }
        const temporary = join(out, `index.json.${process.pid}.tmp`);
        writeFileSync(temporary, `${JSON.stringify(mergeIndex(index, stage, entries), null, 1)}\n`);
        renameSync(temporary, indexPath);
    });
}
