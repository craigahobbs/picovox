// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The package's outputs of the parity checks' inputs (test/parity-inputs.json - the training's Python port's checks,
// train/src/tests/parity.py), for those checks - make parity:
//
//     node tools/parity.js [build/parity-outputs.json]

import {gaussAt, streamSeed, uniformAt} from '../lib/rng.js';
import {loadModel, speakFrames} from '../lib/model.js';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {Buffer} from 'node:buffer';
import {NAMES} from '../lib/params.js';
import {dirname} from 'node:path';
import {g2pModel} from '../lib/models/g2pModel.js';
import {lfHarmonics} from '../lib/lf.js';
import {loadG2p} from '../lib/g2p.js';
import process from 'node:process';
import {render} from '../lib/render.js';
import {setPronouncer} from '../lib/lexicon.js';
import {tokenize} from '../lib/tokens.js';


const outPath = process.argv[2] ?? 'build/parity-outputs.json';
const inputs = JSON.parse(readFileSync(new URL('../test/parity-inputs.json', import.meta.url), 'utf8'));
const pronounce = await loadG2p(g2pModel);
setPronouncer(pronounce);


// A Math function of doubles, as bits (hex)
const view = new DataView(new ArrayBuffer(8));
function mathBits(fn, hex) {
    view.setBigUint64(0, BigInt(`0x${hex}`));
    view.setFloat64(0, fn(view.getFloat64(0)));
    return view.getBigUint64(0).toString(16).padStart(16, '0');
}


// An exported model's frames, durations, and eq of its texts
function exportOutputs({manifest, bin, speaker, style, intensity, texts}) {
    const bytes = Buffer.from(bin, 'base64');
    const model = loadModel(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    return texts.map((text) => {
        const spoken = speakFrames(model, text, {speaker, 'style': style === null ? undefined : {'emotion': style, intensity}});
        return {
            'durations': spoken.sentences.flatMap((sentence) => sentence.durations),
            'frames': spoken.frames.map((frame) => Array.from(frame.subarray(0, NAMES.length))),
            'eq': spoken.eq
        };
    });
}


const vocabulary = JSON.parse(readFileSync(new URL('../train/src/picovox_train/vocab.json', import.meta.url), 'utf8')).tokens;
const {rng, render: renderInputs} = inputs;
const seed = streamSeed(rng.seed, rng.stream);
const outputs = {
    'tokenize': inputs.tokenize.texts.map((text) => tokenize(text, vocabulary)),
    'pronounce': inputs.pronounce.words.map((word) => pronounce(word)),
    'expTanh': {
        'exp': inputs.expTanh.values.map((hex) => mathBits(Math.exp, hex)),
        'tanh': inputs.expTanh.values.map((hex) => mathBits(Math.tanh, hex))
    },
    'rng': {
        'streamSeed': seed,
        'uniform': rng.uniform.map((index) => uniformAt(seed, index)),
        'gauss': rng.gauss.map((index) => gaussAt(seed, index))
    },
    'lfHarmonics': {
        'real': inputs.lfHarmonics.rows.map((row) => Array.from(lfHarmonics(row).re.slice(0, inputs.lfHarmonics.count))),
        'imag': inputs.lfHarmonics.rows.map((row) => Array.from(lfHarmonics(row).im.slice(0, inputs.lfHarmonics.count)))
    },
    'render': {
        'samples': Array.from(render(renderInputs.frames, {'sampleRate': renderInputs.sampleRate, 'seed': renderInputs.seed})),
        'samplesEq': Array.from(render(renderInputs.frames,
            {'sampleRate': renderInputs.sampleRate, 'seed': renderInputs.seed, 'eq': renderInputs.eq}))
    },
    'export-8-0': exportOutputs(inputs['export-8-0']),
    'export-6-32': exportOutputs(inputs['export-6-32'])
};
mkdirSync(dirname(outPath), {'recursive': true});
writeFileSync(outPath, JSON.stringify(outputs));
