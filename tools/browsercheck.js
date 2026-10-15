// Check that the browser build works - the app-time modules (lib/, static/audiobook/, nn/js/, and the generated model modules) load
// with no Node built-ins (any node: import from them fails, and process.getBuiltinModule is hidden, as in a browser), the
// front end pronounces with the G2P network (no dictionary), and the worker's neural speech matches the same pipeline run
// directly (speakFrames, render, gain, and limiter), sample for sample
//
// node tools/browsercheck.js

import {createHash} from 'node:crypto';
import process from 'node:process';
import {registerHooks} from 'node:module';


// Fail any node: import (or bare package import) from the app-time code
const browserRoots = ['lib', 'static', 'nn/js'].map((dir) => new URL(`../${dir}/`, import.meta.url).href);
registerHooks({
    resolve(specifier, context, nextResolve) {
        if (context.parentURL && browserRoots.some((root) => context.parentURL.startsWith(root)) &&
            (specifier.startsWith('node:') || !/^[./]/.test(specifier))) {
            throw new Error(`${context.parentURL} imports ${specifier}`);
        }
        return nextResolve(specifier, context);
    }
});


// Load the app's modules as a browser would - no process.getBuiltinModule
const {getBuiltinModule} = process;
process.getBuiltinModule = undefined;
const modules = ['catalog', 'epub', 'g2p', 'models/g2pModel', 'gutenberg', 'lexicon', 'phonemes', 'wav', 'zip'];
const lib = {};
for (const name of modules) {
    lib[name] = await import(`../lib/${name}.js`);
}
const worker = await import('../static/audiobook/worker.js');
const {speakFrames} = await import('../nn/js/model.js');
const {render} = await import('../nn/js/render.js');
process.stdout.write(`Loaded lib/{${modules.join(',')}}.js, static/audiobook/worker.js, and nn/js/ with no node: imports\n`);

// Without a pronouncer (and no Node to load a dictionary), the text front end must ask for one
let threw = false;
try {
    lib.lexicon.textToPhrases('hello');
} catch {
    threw = true;
}
if (!threw) {
    throw new Error('textToPhrases worked without a pronouncer or Node');
}

// Pronounce with the G2P network, as the worker does - then speak with the worker's functions
lib.lexicon.setPronouncer(await lib.g2p.loadG2p(lib['models/g2pModel'].g2pModel));
const [phrase] = lib.lexicon.textToPhrases('Queequeg and Netherfield');
const spelled = (word) => `${word.text} ${word.phones.map(({ph, stress}) => ph + (stress ?? '')).join(' ')}`;
process.stdout.write(`G2P: ${phrase.words.map(spelled).join(', ')}\n`);
const model = await worker.loadNeuralModel();
const speaker = worker.neuralSpeaker(model, 'neural-female');
const text = 'The birch canoe slid on the smooth planks.';
const options = {'rate': 1.2, 'seed': 1, 'paragraphStart': true, speaker};
const blocks = [];
worker.streamNeuralUnit(model, text, options, (samples, sampleRate) => blocks.push({samples, sampleRate}));
const total = blocks.reduce((sum, block) => sum + block.samples.length, 0);
const samples = new Float32Array(total);
let offset = 0;
for (const block of blocks) {
    samples.set(block.samples, offset);
    offset += block.samples.length;
}
const wav = lib.wav.encodeWav(samples, 24000);
process.getBuiltinModule = getBuiltinModule;
process.stdout.write(`Worker speech: ${blocks.length} blocks, ${(total / 24000).toFixed(2)} s, WAV ${wav.length} bytes\n`);

// The worker's audio must match the pipeline run directly: the pause, then the rendered frames with the gain and limiter
const {frames, eq} = speakFrames(model, text, {'rate': 1 / options.rate, speaker});
const rendered = render(frames, {'sampleRate': 24000, 'seed': options.seed, eq});
const pause = total - rendered.length;
const expected = new Float32Array(total);
rendered.forEach((value, index) => {
    expected[pause + index] = worker.softLimit(value * worker.neuralGain);
});
const sha = (values) => createHash('sha256').update(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)).digest('hex');
if (pause <= 0 || sha(samples) !== sha(expected)) {
    throw new Error('The worker speech differs from the pipeline run directly');
}
process.stdout.write('Worker speech matches the pipeline run directly\n');
