// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// make train's listening samples - a training stage's final model (a checkpoint, exported by picovox_train.export as the gate
// exports the voice) or a packaged model (the shipped lib/models/voiceModel.js, the reference) speaking
// train/data/train.json's samples texts in its speakers' voices exactly as the package speaks a sentence (lib/synthesis.js
// sentenceBlocks, every word pronounced by the G2P network), each scored with UTMOS on the CPU
// (picovox_train/listeners.py - the GPU is left to training). The WAV files are OUT/<stage>-<speaker>-<text>.wav, and
// OUT/index.json lists every stage's samples - [{stage, run (a checkpoint's - its training stage's), checkpoint, step,
// label, sex, speaker, text, file, seconds, mos}] - updated under a lock and replaced atomically, so sample stages
// finishing at once keep each other's entries.
//
//     node tools/trainsamples.js --stage voice-a --checkpoint build/train/runs/voice-a/last.pt --out build/train/samples
//     node tools/trainsamples.js --stage reference --model lib/models/voiceModel.js --out build/train/samples
//
// The export reads the training data (NN_DATA_DIR - make train's TRAIN_ENV) for its token remap.

import {defaultSampleRate, sentenceBlocks, sentencePause} from '../lib/synthesis.js';
import {dirname, join, relative, resolve} from 'node:path';
import {mkdirSync, readFileSync, renameSync, rmSync, writeFileSync} from 'node:fs';
import {parseArgs, parseScores, sampleFile, updateIndex} from './trainsamplesindex.js';
import {concatBlocks} from '../lib/render.js';
import {encodeWav} from '../lib/wav.js';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {g2pModel} from '../lib/models/g2pModel.js';
import {loadG2p} from '../lib/g2p.js';
import {loadVoice} from './loadVoice.js';
import process from 'node:process';
import {setPronouncer} from '../lib/lexicon.js';


// The repository, and make train's Python
const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const python = join(repository, 'build/venv-train/bin/python');


// Render the samples
async function main() {
    const args = parseArgs(process.argv.slice(2));
    const settings = JSON.parse(readFileSync(resolve(repository, 'train/data/train.json'), 'utf8')).samples;
    const out = resolve(args.out);
    const work = join(out, `.work-${args.stage}`);
    rmSync(work, {'recursive': true, 'force': true});
    mkdirSync(work, {'recursive': true});
    const log = (message) => process.stdout.write(`trainsamples: ${message}\n`);

    // The G2P network - the package's pronouncer
    setPronouncer(await loadG2p(g2pModel));

    // The model - a checkpoint (its training stage finished) exported to the browser's format, or a packaged module
    let model;
    let checkpoint;
    if (args.checkpoint !== null) {
        // (its path relative to the samples' parent - TRAIN_DIR)
        checkpoint = relative(dirname(out), resolve(args.checkpoint));
        execFileSync(python, ['-B', '-m', 'picovox_train.export', resolve(args.checkpoint), '--voice', 'model', '--out', work],
            {'cwd': repository, 'env': {...process.env, 'PYTHONPATH': join(repository, 'train/src')},
                'stdio': ['ignore', 'ignore', 'inherit']});
        model = await loadVoice(join(work, 'model'));
    } else {
        checkpoint = relative(repository, resolve(args.model));
        model = await loadVoice(args.model);
    }
    const step = model.manifest.training?.step ?? null;
    const run = args.checkpoint === null ? null : args.stage;
    log(`${args.stage}: ${checkpoint}${step === null ? '' : ` (step ${step})`}`);

    // Each speaker's texts, rendered as the package renders a sentence (seed 1, normal speed)
    const samples = [];
    for (const speaker of settings.speakers) {
        const entry = model.manifest.speakers.find((item) => item.key === speaker.key);
        if (!entry) {
            throw new Error(`the speaker ${speaker.key} isn't in the model`);
        }
        for (const text of settings.texts) {
            const blocks = [...sentenceBlocks(model, text.text, {'speaker': entry.id, 'seed': 1, 'pause': sentencePause})];
            const audio = concatBlocks(blocks);
            const file = sampleFile(args.stage, speaker.key, text.key);
            writeFileSync(join(work, file), encodeWav(audio, defaultSampleRate));
            samples.push({'stage': args.stage, run, checkpoint, step, 'label': speaker.label, 'sex': speaker.sex,
                'speaker': speaker.key, 'text': text.key, file, 'seconds': Math.round(audio.length / defaultSampleRate * 100) / 100});
        }
    }
    log(`${samples.length} samples rendered`);

    // UTMOS on the CPU
    const output = execFileSync(python, ['-I', 'train/src/picovox_train/listeners.py', '--no-embedding',
        ...samples.map((sample) => join(work, sample.file))], {'cwd': repository, 'encoding': 'utf8',
        'maxBuffer': 16 * 1024 * 1024, 'stdio': ['ignore', 'pipe', 'inherit']});
    const scores = parseScores(output);
    for (const sample of samples) {
        sample.mos = Number.isFinite(scores[sample.file]) ? Math.round(scores[sample.file] * 1000) / 1000 : null;
    }

    // The samples in place, then the index
    for (const sample of samples) {
        renameSync(join(work, sample.file), join(out, sample.file));
    }
    rmSync(work, {'recursive': true, 'force': true});
    updateIndex(out, args.stage, samples);
    const mean = (sex) => {
        const values = samples.filter((sample) => sample.sex === sex && sample.mos !== null).map((sample) => sample.mos);
        return values.length === 0 ? '-' : (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3);
    };
    log(`${args.stage}: UTMOS female ${mean('female')}, male ${mean('male')} - ${join(args.out, 'index.json')}`);
}
await main();
