// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// make train's listening samples - a voice model module (the gate's evaluation of the trained voice, TRAIN_DIR/eval/voice/voiceModel.js,
// or the shipped lib/models/voiceModel.js, the reference) speaking train/data/train.json's samples texts in its
// speakers' voices exactly as the package speaks a sentence (lib/synthesis.js sentenceBlocks, every word pronounced by
// the G2P network). The WAV files are OUT/<stage>-<speaker>-<text>.wav, and OUT/index.json lists every stage's samples -
// [{stage, model (the module's path), step (its training step, if it records one), label, sex, speaker, text, file,
// seconds}] - updated under a lock and replaced atomically, so sample stages finishing at once keep each other's entries.
//
//     node tools/trainsamples.js --stage voice-b --model build/train/eval/voice/voiceModel.js --out build/train/samples
//     node tools/trainsamples.js --stage reference --model lib/models/voiceModel.js --out build/train/samples

import {defaultSampleRate, sentenceBlocks, sentencePause} from '../lib/synthesis.js';
import {dirname, join, relative, resolve} from 'node:path';
import {mkdirSync, readFileSync, renameSync, rmSync, writeFileSync} from 'node:fs';
import {parseArgs, sampleFile, updateIndex} from './trainsamplesindex.js';
import {concatBlocks} from '../lib/render.js';
import {encodeWav} from '../lib/wav.js';
import {fileURLToPath} from 'node:url';
import {g2pModel} from '../lib/models/g2pModel.js';
import {loadG2p} from '../lib/g2p.js';
import {loadVoice} from './loadVoice.js';
import process from 'node:process';
import {setPronouncer} from '../lib/lexicon.js';


// The repository
const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..');


// Render the samples
async function main() {
    const args = parseArgs(process.argv.slice(2));
    const settings = JSON.parse(readFileSync(resolve(repository, 'train/data/train.json'), 'utf8')).samples;
    const out = resolve(args.out);
    const work = join(out, `.work-${args.stage}`);
    rmSync(work, {'recursive': true, 'force': true});
    mkdirSync(work, {'recursive': true});
    const log = (message) => process.stdout.write(`trainsamples: ${message}\n`);

    // The G2P network - the package's pronouncer - and the model
    setPronouncer(await loadG2p(g2pModel));
    const model = await loadVoice(args.model);
    const modelPath = relative(repository, resolve(args.model));
    const step = model.manifest.training?.step ?? null;
    log(`${args.stage}: ${modelPath}${step === null ? '' : ` (step ${step})`}`);

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
            samples.push({'stage': args.stage, 'model': modelPath, step, 'label': speaker.label, 'sex': speaker.sex,
                'speaker': speaker.key, 'text': text.key, file, 'seconds': Math.round(audio.length / defaultSampleRate * 100) / 100});
        }
    }

    // The samples in place, then the index
    for (const sample of samples) {
        renameSync(join(work, sample.file), join(out, sample.file));
    }
    rmSync(work, {'recursive': true, 'force': true});
    updateIndex(out, args.stage, samples);
    log(`${args.stage}: ${samples.length} samples rendered - ${join(args.out, 'index.json')}`);
}
await main();
