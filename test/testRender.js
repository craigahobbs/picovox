// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {DEFAULTS, FLOOR_DB, NAMES, PARAMETERS} from '../lib/params.js';
import {createRenderer, frameValues, noiseFftSize, render, sampleCount} from '../lib/render.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A voiced vowel frame (an object of parameters), varied over time
function vowel(index, values = {}) {
    return {
        'f0': 110 + 20 * Math.sin(index / 15), 'av': -8, 'rd': 1.1, 'shelf': 2, 'jitter': 0.005, 'shimmer': 0.02, 'ah': -45,
        'f1': 650 + 100 * Math.sin(index / 9), 'f2': 1100, 'f3': 2500, 'b1': 80, 'b2': 90, 'b3': 150, ...values
    };
}


// A fricative frame - frication, no voicing
function fricative(values = {}) {
    return {'av': FLOOR_DB, 'af': -10, 'ap2': -5, 'fp2': 4500, 'bp2': 600, 'ab': -30, 'ah': -200, ...values};
}


// A sequence of frames - vowel, fricative, vowel, silence
function phrase(count = 120) {
    const frames = [];
    for (let ix = 0; ix < count; ix++) {
        const part = Math.floor(4 * ix / count);
        if (part === 0 || part === 2) {
            frames.push(vowel(ix));
        } else if (part === 1) {
            frames.push(fricative());
        } else {
            frames.push({'ah': -200, 'af': -200});
        }
    }
    return frames;
}


test('render.noiseFftSize', () => {
    assert.equal(noiseFftSize(24000), 512);
    assert.equal(noiseFftSize(44100), 1024);
    assert.equal(noiseFftSize(16000), 256);
    assert.equal(noiseFftSize(8000), 128);
});


test('render.sampleCount', () => {
    assert.equal(sampleCount(0, 24000), 0);
    assert.equal(sampleCount(1, 24000), 0);
    assert.equal(sampleCount(2, 24000), 120);
    assert.equal(sampleCount(201, 44100), 44100);
    assert.equal(sampleCount(3, 44100), 441);
});


test('render.frameValues', () => {
    // An object - missing and null parameters take their defaults
    const values = frameValues({'f0': 150, 'av': null, 'f1': undefined});
    assert(values instanceof Float64Array);
    assert.equal(values.length, NAMES.length);
    assert.equal(values[0], 150);
    assert.deepEqual(Array.from(values.slice(1)), PARAMETERS.slice(1).map((parameter) => parameter.default));
    assert.deepEqual(Array.from(frameValues({})), NAMES.map((name) => DEFAULTS[name]));

    // An array or a typed array, in parameter order
    const array = NAMES.map((_name, ix) => ix + 0.5);
    assert.deepEqual(Array.from(frameValues(array)), array);
    assert.deepEqual(Array.from(frameValues(Float32Array.from(array))), array);
    assert.deepEqual(Array.from(frameValues([7])).slice(0, 2), [7, PARAMETERS[1].default]);
});


test('render', () => {
    const frames = phrase();
    const samples = render(frames, {'sampleRate': 24000, 'seed': 3});
    assert(samples instanceof Float32Array);
    assert.equal(samples.length, sampleCount(frames.length, 24000));
    assert(samples.every(Number.isFinite));

    // Voiced and fricative parts have energy; the silent end is silent
    const energy = (start, end) => samples.slice(start, end).reduce((sum, value) => sum + value * value, 0) / (end - start);
    const quarter = samples.length / 4;
    assert(energy(0, quarter) > 1e-4);
    assert(energy(quarter + 1000, 2 * quarter - 1000) > 1e-6);
    assert.equal(energy(3 * quarter + 1000, samples.length), 0);

    // Deterministic; the seed changes the noise
    assert.deepEqual(render(frames, {'sampleRate': 24000, 'seed': 3}), samples);
    assert.notDeepEqual(render(frames, {'sampleRate': 24000, 'seed': 4}), samples);

    // The defaults - 44.1 kHz, seed 1
    assert.equal(render(frames).length, sampleCount(frames.length, 44100));
    assert.deepEqual(render(frames), render(frames, {'sampleRate': 44100, 'seed': 1}));

    // No frames, one frame
    assert.deepEqual(render([]), new Float32Array(0));
    assert.deepEqual(render([vowel(0)]), new Float32Array(0));
});


test('render, channel equalizer', () => {
    const frames = phrase();
    const flat = render(frames, {'sampleRate': 24000});
    assert.deepEqual(render(frames, {'sampleRate': 24000, 'eq': new Array(24).fill(0)}), flat);
    const boosted = render(frames, {'sampleRate': 24000, 'eq': new Array(24).fill(6)});
    const ratio = boosted.reduce((sum, value) => sum + Math.abs(value), 0) / flat.reduce((sum, value) => sum + Math.abs(value), 0);
    assert(Math.abs(ratio - 10 ** (6 / 20)) < 0.01, `ratio ${ratio}`);
    const tilted = render(frames, {'sampleRate': 24000, 'eq': Array.from({'length': 24}, (_value, ix) => ix - 12)});
    assert.notDeepEqual(tilted, flat);
});


test('render, onBlock', () => {
    const frames = phrase();
    const blocks = [];
    const samples = render(frames, {'sampleRate': 16000, 'blockSize': 1000, 'onBlock': (block) => blocks.push(block.slice())});
    assert(blocks.slice(0, -1).every((block) => block.length === 1000));
    assert(blocks[blocks.length - 1].length <= 1000);
    const joined = new Float32Array(samples.length);
    blocks.reduce((offset, block) => {
        joined.set(block, offset);
        return offset + block.length;
    }, 0);
    assert.deepEqual(joined, samples);
});


test('render, silence', () => {
    // Long silence - no voicing, no noise - renders exact zeros
    const frames = new Array(300).fill({'av': FLOOR_DB, 'ah': -200, 'af': -200});
    const samples = render(frames, {'sampleRate': 24000});
    assert.equal(samples.length, sampleCount(300, 24000));
    assert(samples.every((value) => value === 0));

    // Aspiration alone, frication alone
    const aspiration = render(new Array(40).fill({'av': FLOOR_DB, 'ah': -20, 'af': -200}), {'sampleRate': 24000});
    const frication = render(new Array(40).fill(fricative()), {'sampleRate': 24000});
    assert(aspiration.some((value) => value !== 0));
    assert(frication.some((value) => value !== 0));
    assert.notDeepEqual(aspiration, frication);
});


test('createRenderer', () => {
    // Streaming - frames pushed one at a time, as arrays, as typed arrays, or in lists - is rendering whole
    const frames = phrase(400);
    const whole = render(frames, {'sampleRate': 24000, 'seed': 9});
    const blocks = [];
    const renderer = createRenderer({'sampleRate': 24000, 'seed': 9, 'blockSize': 777, 'onBlock': (block) => blocks.push(block.slice())});
    assert.equal(renderer.sampleRate, 24000);
    renderer.push(frames[0]);
    renderer.push(NAMES.map((name) => frameValues(frames[1])[NAMES.indexOf(name)]));
    renderer.push(frameValues(frames[2]));
    renderer.push(frames.slice(3, 100));
    for (const frame of frames.slice(100)) {
        renderer.push([frame]);
    }

    // Blocks come as soon as they're final, before the end
    const beforeEnd = blocks.length;
    assert(beforeEnd > 10);
    const total = renderer.end();
    assert.equal(total, whole.length);
    assert(blocks.length > beforeEnd);
    const joined = new Float32Array(total);
    blocks.reduce((offset, block) => {
        joined.set(block, offset);
        return offset + block.length;
    }, 0);
    assert.deepEqual(joined, whole);

    // Frames pushed after the end render nothing
    const endBlocks = blocks.length;
    renderer.push(frames.slice(0, 10));
    assert.equal(blocks.length, endBlocks);
});


test('createRenderer, defaults', () => {
    // No block callback, 44.1 kHz - the total only
    const renderer = createRenderer();
    assert.equal(renderer.sampleRate, 44100);
    renderer.push(phrase(20));
    assert.equal(renderer.end(), sampleCount(20, 44100));

    // Nothing pushed - no samples (and no blocks - a block callback isn't called: render's empty result)
    assert.equal(createRenderer({'sampleRate': 8000}).end(), 0);
    assert.deepEqual(render([], {'sampleRate': 8000}), new Float32Array(0));
});
