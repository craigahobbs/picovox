// The performance lab's timings - the package's playback pipeline, stage by stage, from a lib/ base URL (the package's
// lib/ or a variant of it), in a browser worker, on a page's main thread, or under Node (the same code everywhere):
//
// - load: the generated model modules' import, and the G2P and voice networks' unpacking
// - stages: for each text and voice, the front end (textToPhrases - the G2P network, cached after the first pass), the
//   voice network (streamFrames - every chunk), and the renderer (createRenderer, all frames pushed) apart
// - stream: the pipeline as PicoShelf plays it (synthesis.js sentenceBlocks - the network and renderer interleaved)
//
// Each is timed over passes (the first warms the JIT); a stage's speed is seconds of audio per second of work (times
// real time). A checksum of the samples identifies the output, so two libs' outputs can be compared.
//
//     import {runTiming} from './timing.js';
//     const result = await runTiming({'lib': '/lib', 'voices': ['male'], 'passes': 2});


// The default texts - a long and a short paragraph (Pride and Prejudice)
export const defaultTexts = [
    'It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. ' +
        'However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is ' +
        'so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or ' +
        'other of their daughters.',
    '"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"'
];


// The time now (ms)
function now() {
    return performance.now();
}


// A checksum of samples (FNV-1a over their float32 bits)
export function checksum(blocks) {
    let hash = 0x811c9dc5;
    for (const block of blocks) {
        const bytes = new Uint8Array(Float32Array.from(block).buffer);
        for (const byte of bytes) {
            hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
        }
    }
    return hash.toString(16).padStart(8, '0');
}


// A fixed arithmetic workload (PicoShelf's speed check - a few milliseconds with a JIT) - ms
function speedCheck() {
    const start = now();
    const values = new Float64Array(4096);
    let sum = 0;
    for (let pass = 0; pass < 500; pass++) {
        for (let ix = 0; ix < values.length; ix++) {
            values[ix] = values[ix] * 0.999 + ix;
            sum += values[ix];
        }
    }
    return {'ms': Math.round(now() - start), 'checksum': Math.round(sum) % 1000};
}


// A WebAssembly module (hand-assembled): memory, and dot(n) - the sum of a[i] * b[i] over f64 arrays at bytes 0 and 8 n
const dotWasm = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 127, 1, 124, 3, 2, 1, 0, 5, 3, 1, 0, 16, 7, 13, 2, 3, 100, 111, 116,
    0, 0, 3, 109, 101, 109, 2, 0, 10, 57, 1, 55, 2, 1, 127, 1, 124, 2, 64, 3, 64, 32, 1, 32, 0, 78, 13, 1, 32, 2, 32,
    1, 65, 3, 116, 43, 3, 0, 32, 1, 32, 0, 106, 65, 3, 116, 43, 3, 0, 162, 160, 33, 2, 32, 1, 65, 1, 106, 33, 1, 12,
    0, 11, 11, 32, 2, 11
]);


// The speed check repeated, and three more fixed workloads (trigonometry, and a typed-array dot product in JavaScript and
// in WebAssembly) - each's ms per repetition: with a JIT, the later repetitions run faster than the first (the code is
// compiled); an interpreter never speeds up
export function runBench({repeats = 10} = {}) {
    const result = {'speedCheck': [], 'trig': [], 'dot': []};
    const left = new Float64Array(1 << 16).map((_value, ix) => Math.sin(ix));
    const right = new Float64Array(1 << 16).map((_value, ix) => Math.cos(ix));
    for (let repeat = 0; repeat < repeats; repeat++) {
        result.speedCheck.push(speedCheck().ms);
        let start = now();
        let sum = 0;
        for (let ix = 0; ix < 200000; ix++) {
            sum += Math.cos(ix * 0.001) * Math.exp(-ix * 1e-6);
        }
        result.trig.push(Math.round(now() - start));
        start = now();
        for (let pass = 0; pass < 30; pass++) {
            for (let ix = 0; ix < left.length; ix++) {
                sum += left[ix] * right[ix];
            }
        }
        result.dot.push(Math.round(now() - start));
        result.checksum = Math.round(sum) % 1000;
    }

    // The dot product in WebAssembly, if it's available (Lockdown Mode disables it)
    try {
        const instance = new WebAssembly.Instance(new WebAssembly.Module(dotWasm));
        const memory = new Float64Array(instance.exports.mem.buffer);
        memory.set(left);
        memory.set(right, left.length);
        result.wasm = [];
        for (let repeat = 0; repeat < repeats; repeat++) {
            const start = now();
            let sum = 0;
            for (let pass = 0; pass < 30; pass++) {
                sum += instance.exports.dot(left.length);
            }
            result.wasm.push(Math.round(10 * (now() - start)) / 10);
            result.wasmChecksum = Math.round(sum) % 1000;
        }
    } catch (error) {
        result.wasm = `unavailable: ${error.message}`;
    }
    return result;
}


// Import a lib's modules - {lexicon, g2p, packaged, model, render, synthesis, g2pModule, voiceModule} with the import times
async function importLib(lib) {
    const start = now();
    const [lexicon, g2p, packaged, model, render, synthesis] = await Promise.all(
        ['lexicon', 'g2p', 'package', 'model', 'render', 'synthesis'].map((name) => import(`${lib}/${name}.js`))
    );
    const codeMs = now() - start;
    const g2pStart = now();
    const g2pModule = await import(`${lib}/models/g2pModel.js`);
    const g2pImportMs = now() - g2pStart;
    const voiceStart = now();
    const voiceModule = await import(`${lib}/models/voiceModel.js`);
    const voiceImportMs = now() - voiceStart;
    return {lexicon, g2p, packaged, model, render, synthesis, g2pModule, voiceModule, codeMs, g2pImportMs, voiceImportMs};
}


// Time one text and voice, stage by stage - {audio (s), frontMs, networkMs, renderMs, checksum}
function timeStages(lib, voiceModel, text, speaker, sampleRate) {
    let start = now();
    lib.lexicon.textToPhrases(text);
    const frontMs = now() - start;
    start = now();
    const {chunks, eq} = lib.model.streamFrames(voiceModel, text, {speaker});
    const frames = [];
    for (const chunk of chunks) {
        frames.push(...chunk);
    }
    const networkMs = now() - start;
    start = now();
    const blocks = [];
    const renderer = lib.render.createRenderer({sampleRate, 'seed': 1, eq, 'onBlock': (block) => blocks.push(block)});
    renderer.push(frames);
    renderer.end();
    const renderMs = now() - start;
    return {'audio': frames.length / 200, frontMs, networkMs, renderMs, 'checksum': checksum(blocks)};
}


// Time one text and voice as PicoShelf plays it (textBlocks - sentence by sentence, each with its pause; the first
// speech's latency too) - {audio, streamMs, firstMs, checksum}
function timeStream(lib, voiceModel, text, speaker, sampleRate) {
    const start = now();
    let firstMs = null;
    let samples = 0;
    const blocks = [];
    for (const block of lib.synthesis.textBlocks(voiceModel, text, {speaker, 'seed': 1, sampleRate})) {
        if (firstMs === null && block.some((value) => value !== 0)) {
            firstMs = now() - start;
        }
        samples += block.length;
        blocks.push(block);
    }
    return {'audio': samples / sampleRate, 'streamMs': now() - start, firstMs, 'checksum': checksum(blocks)};
}


// Round a result's milliseconds and add each stage's speed (times real time)
function summarize(entry) {
    const out = {...entry};
    for (const [key, value] of Object.entries(entry)) {
        if (key.endsWith('Ms') && typeof value === 'number') {
            out[key] = Math.round(value);
            if (key !== 'firstMs') {
                out[`${key.slice(0, -2)}Speed`] = Math.round(10 * entry.audio * 1000 / value) / 10;
            }
        }
    }
    out.audio = Math.round(100 * entry.audio) / 100;
    return out;
}


// Run a timing - options: lib (base URL), texts, voices, passes, sampleRate, stages, stream - the result
export async function runTiming({lib = '/lib', texts = defaultTexts, voices = ['male'], passes = 2, sampleRate = 24000, stages = true,
    stream = true} = {}) {
    const result = {lib, passes, sampleRate, 'speedCheck': speedCheck(), 'runs': []};
    const modules = await importLib(lib);
    let start = now();
    modules.lexicon.setPronouncer(await modules.g2p.loadG2p(modules.g2pModule.g2pModel));
    const g2pUnpackMs = now() - start;
    start = now();
    const voiceModel = await modules.packaged.loadPackagedModel(modules.voiceModule);
    const voiceUnpackMs = now() - start;
    result.load = {
        'codeMs': Math.round(modules.codeMs), 'g2pImportMs': Math.round(modules.g2pImportMs), 'g2pUnpackMs': Math.round(g2pUnpackMs),
        'voiceImportMs': Math.round(modules.voiceImportMs), 'voiceUnpackMs': Math.round(voiceUnpackMs)
    };
    for (let pass = 0; pass < passes; pass++) {
        for (const voice of voices) {
            const speaker = modules.synthesis.voiceSpeaker(voiceModel, voice);
            for (const [index, text] of texts.entries()) {
                const run = {pass, voice, 'text': index};
                if (stages) {
                    run.stages = summarize(timeStages(modules, voiceModel, text, speaker, sampleRate));
                }
                if (stream) {
                    run.stream = summarize(timeStream(modules, voiceModel, text, speaker, sampleRate));
                }
                if (modules.render.renderProfile) {
                    run.profile = modules.render.renderProfile(true);
                }
                if (modules.model.modelProfile) {
                    run.modelProfile = modules.model.modelProfile(true);
                }
                result.runs.push(run);
            }
        }
    }
    return result;
}
