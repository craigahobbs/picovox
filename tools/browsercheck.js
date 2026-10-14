// Check that the browser build works - lib/ and web/ modules load with no Node built-ins (any node: import from them
// fails, and process.getBuiltinModule is hidden, as in a browser), and the worker's synthesis matches tools/say.js
//
// node tools/browsercheck.js

import {createHash} from 'node:crypto';
import process from 'node:process';
import {readFileSync} from 'node:fs';
import {registerHooks} from 'node:module';


// Fail any node: import from lib/ or web/
const browserRoots = ['lib', 'web'].map((dir) => new URL(`../${dir}/`, import.meta.url).href);
registerHooks({
    resolve(specifier, context, nextResolve) {
        if (context.parentURL && browserRoots.some((root) => context.parentURL.startsWith(root)) &&
            (specifier.startsWith('node:') || !/^[./]/.test(specifier))) {
            throw new Error(`${context.parentURL} imports ${specifier}`);
        }
        return nextResolve(specifier, context);
    }
});


// Load every lib and web module as a browser would - no process.getBuiltinModule
const dictionaryText = readFileSync(new URL('../data/cmudict.dict', import.meta.url), 'utf8');
const {getBuiltinModule} = process;
process.getBuiltinModule = undefined;
const modules = ['analysis', 'glottal', 'klatt', 'lexicon', 'phonemes', 'room', 'speak', 'wav'];
const lib = {};
for (const name of modules) {
    lib[name] = await import(`../lib/${name}.js`);
}
const worker = await import('../web/worker.js');
process.stdout.write(`Loaded lib/{${modules.join(',')}}.js and web/worker.js with no node: imports\n`);

// Without a dictionary, the text front end must ask for one
let threw = false;
try {
    lib.lexicon.textToPhrases('hello');
} catch {
    threw = true;
}
if (!threw) {
    throw new Error('textToPhrases worked without a dictionary or Node');
}

// Synthesize with the worker's functions, from the dictionary string
lib.lexicon.setDictionary(dictionaryText);
const text = 'The birch canoe slid on the smooth planks.\n\nGlue the sheet to the dark blue background!\n\n...';
const chunks = [];
const options = {'voice': 'female', 'rate': 1.2, 'seed': 1, 'room': true};
const count = worker.synthesizeParagraphs(text, options, (index, samples, sampleRate) => chunks.push({index, samples, sampleRate}));
const wav = lib.wav.encodeWav(chunks[0].samples, chunks[0].sampleRate);
process.getBuiltinModule = getBuiltinModule;
const chunkSeconds = chunks.map((chunk) => (chunk.samples.length / chunk.sampleRate).toFixed(2));
process.stdout.write(`Worker synthesis: ${count} paragraphs, ${chunkSeconds.join(' + ')} s, WAV ${wav.length} bytes\n`);

// The worker's audio must match the Node path (speak, room, normalize) sample for sample
const {speak} = lib.speak;
const sha = (samples) => createHash('sha256').update(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)).digest('hex');
const dry = speak('The birch canoe slid on the smooth planks.', 'female', {'seed': 1, 'rate': 1.2}).samples;
const expected = lib.klatt.normalize(lib.room.convolve(dry, lib.room.roomImpulse(44100)), 0.89);
if (count !== 2 || sha(chunks[0].samples) !== sha(expected)) {
    throw new Error('The worker synthesis differs from the Node synthesis');
}
process.stdout.write('Worker synthesis matches the Node synthesis\n');
