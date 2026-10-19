// Compare a variant of the package's lib/ (tools/perf/variant.sh) with lib/ - the output's fidelity:
//
// - network: the variant's frames against lib/'s (the largest difference, relative to each parameter's range)
// - render: the variant's renderer against lib/'s on the same (lib/'s) frames - identical, else the SNR (dB)
// - pipeline: the variant end to end against lib/ end to end - identical, else the SNR
//
//     node tools/perf/compare.js VARIANT [--ref REFERENCE] [--voices male,female]
//
// VARIANT (and REFERENCE - lib/ by default) is a variant's name (build/perf/variants/NAME/lib) or a lib directory. The
// speed is tools/perf/speed.js's (one lib in a process - two libs timed in one process time unevenly).

/* eslint-disable no-console */

import {defaultTexts} from './timing.js';
import {existsSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import process from 'node:process';
import {resolve} from 'node:path';


// Import a lib's modules (by directory)
async function importLib(directory) {
    const base = pathToFileURL(resolve(directory)).href;
    const names = ['lexicon', 'g2p', 'package', 'model', 'render', 'synthesis', 'params'];
    const modules = Object.fromEntries(await Promise.all(names.map(async (name) => [name, await import(`${base}/${name}.js`)])));
    modules.g2pModule = await import(`${base}/models/g2pModel.js`);
    modules.voiceModule = await import(`${base}/models/voiceModel.js`);
    modules.lexicon.setPronouncer(await modules.g2p.loadG2p(modules.g2pModule.g2pModel));
    modules.voice = await modules.package.loadPackagedModel(modules.voiceModule);
    return modules;
}


// A lib's frames of a text - {frames, eq}
function speak(lib, text, voice) {
    const {chunks, eq} = lib.model.streamFrames(lib.voice, text, {'speaker': lib.synthesis.voiceSpeaker(lib.voice, voice)});
    const frames = [];
    for (const chunk of chunks) {
        frames.push(...chunk);
    }
    return {frames, eq};
}


// A lib's render of frames (24 kHz, seed 1) - Float32Array
function renderFrames(lib, frames, eq) {
    const blocks = [];
    const onBlock = (block) => blocks.push(Float32Array.from(block));
    const renderer = lib.render.createRenderer({'sampleRate': 24000, 'seed': 1, eq, onBlock});
    renderer.push(frames);
    const total = renderer.end();
    const out = new Float32Array(total);
    let offset = 0;
    for (const block of blocks) {
        out.set(block, offset);
        offset += block.length;
    }
    return out;
}


// The fidelity of samples to a reference - 'identical', else the SNR (dB) and the largest difference
function fidelity(reference, samples) {
    if (reference.length !== samples.length) {
        return `length ${samples.length} vs ${reference.length}`;
    }
    let signal = 0;
    let noise = 0;
    let largest = 0;
    for (let ix = 0; ix < reference.length; ix++) {
        const difference = samples[ix] - reference[ix];
        signal += reference[ix] * reference[ix];
        noise += difference * difference;
        largest = Math.max(largest, Math.abs(difference));
    }
    return noise === 0 ? 'identical' : `SNR ${(10 * Math.log10(signal / noise)).toFixed(1)} dB (largest ${largest.toExponential(1)})`;
}


// The largest frame difference relative to each parameter's range
function frameDifference(params, reference, frames) {
    if (reference.length !== frames.length) {
        return `${frames.length} frames vs ${reference.length}`;
    }
    let largest = 0;
    for (const [index, frame] of frames.entries()) {
        for (const [ix, parameter] of params.PARAMETERS.entries()) {
            largest = Math.max(largest, Math.abs(frame[ix] - reference[index][ix]) / (parameter.high - parameter.low));
        }
    }
    return largest === 0 ? 'identical' : `largest ${largest.toExponential(1)} of a range`;
}


async function main(argv) {
    const option = (name, fallback) => {
        const index = argv.indexOf(name);
        return index >= 0 ? argv.splice(index, 2)[1] : fallback;
    };
    const voices = option('--voices', 'male,female').split(',');
    const libDirectory = (name) => (existsSync(name) ? name : `build/perf/variants/${name}/lib`);
    const reference = await importLib(libDirectory(option('--ref', 'lib')));
    const [variant] = argv.filter((arg) => !arg.startsWith('--'));
    const lib = await importLib(libDirectory(variant));
    for (const voice of voices) {
        for (const [index, text] of defaultTexts.entries()) {
            const ref = speak(reference, text, voice);
            const out = speak(lib, text, voice);
            const refSamples = renderFrames(reference, ref.frames, ref.eq);
            console.log(`${voice}/${index}: network ${frameDifference(reference.params, ref.frames, out.frames)}; ` +
                `render ${fidelity(refSamples, renderFrames(lib, ref.frames, ref.eq))}; ` +
                `pipeline ${fidelity(refSamples, renderFrames(lib, out.frames, out.eq))}`);
        }
    }
}


main(process.argv.slice(2));
