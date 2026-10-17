// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as voiceModule from '../lib/models/voiceModel.js';
import {loadModel, sentences, speakFrames, streamFrames} from '../lib/model.js';
import {NAMES} from '../lib/params.js';
import {strict as assert} from 'node:assert';
import {bandwidthScale} from '../lib/transforms.js';
import {g2pModel} from '../lib/models/g2pModel.js';
import {loadG2p} from '../lib/g2p.js';
import {loadPackagedModel} from '../lib/package.js';
import {readFileSync} from 'node:fs';
import {setPronouncer} from '../lib/lexicon.js';
import test from 'node:test';


// The G2P network pronounces, as in the package
setPronouncer(await loadG2p(g2pModel));


// The package's multi-speaker model, and the single-speaker slt export (data/nn/slt)
const multi = await loadPackagedModel(voiceModule);
function loadSlt() {
    const bytes = readFileSync(new URL('../data/nn/slt.bin', import.meta.url));
    const manifest = JSON.parse(readFileSync(new URL('../data/nn/slt.json', import.meta.url), 'utf8'));
    return loadModel(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
const slt = loadSlt();


// Frames as plain arrays (for comparisons)
function plain(frames) {
    return frames.map((frame) => Array.from(frame));
}


// A float16's bits of a normal number (exact for the test's values)
function floatToHalf(value) {
    const sign = value < 0 ? 0x8000 : 0;
    const exponent = Math.floor(Math.log2(Math.abs(value)));
    return sign | ((exponent + 15) << 10) | Math.round((Math.abs(value) / 2 ** exponent - 1) * 1024);
}


test('loadModel, encodings', () => {
    // A float32 tensor, an int8 tensor with per-row scales, and packed tensors (4 and 3 bits, float16 scales per row or
    // per group) - in one buffer
    const buffer = new ArrayBuffer(128);
    new Float32Array(buffer, 0, 3).set([0.5, -1.25, 3]);
    new Int8Array(buffer, 16, 6).set([1, -2, 3, 127, -128, 0]);
    new Float32Array(buffer, 24, 2).set([0.5, 2]);

    // Packed 4-bit, 2 rows of 3, a scale per row: unsigned values offset by 8, LSB first
    const packed4 = [[-8, 0, 7], [1, -1, 2]];
    const bytes4 = new Uint8Array(buffer, 32, 3);
    packed4.flat().forEach((value, index) => {
        bytes4[index >> 1] |= (value + 8) << (4 * (index & 1));
    });
    new Uint16Array(buffer, 36, 2).set([floatToHalf(0.25), floatToHalf(-3)]);

    // Packed 3-bit, 1 row of 5 in groups of 2 (3 groups): fields cross byte boundaries; the scales are a normal, a
    // subnormal, and an infinity
    const packed3 = [3, -4, 1, 0, -1];
    const bytes3 = new Uint8Array(buffer, 40, 2);
    let bit = 0;
    for (const value of packed3) {
        for (let ix = 0; ix < 3; ix++) {
            bytes3[(bit + ix) >> 3] |= (((value + 4) >> ix) & 1) << ((bit + ix) & 7);
        }
        bit += 3;
    }
    new Uint16Array(buffer, 44, 3).set([floatToHalf(1.5), 0x8001, 0x7c00]);

    // A NaN scale
    new Uint8Array(buffer, 50, 1).set([0x9]);
    new Uint16Array(buffer, 52, 1).set([0x7e00]);

    const manifest = {
        'vocab': ['<pad>', '<unk>', '<bos>', '_'],
        'tensors': [
            {'name': 'float', 'shape': [3], 'offset': 0},
            {'name': 'int8', 'shape': [2, 3], 'offset': 16, 'encoding': 'int8', 'scaleOffset': 24},
            {'name': 'packed4', 'shape': [2, 3], 'offset': 32, 'encoding': 'packed', 'bits': 4, 'group': 3, 'scaleOffset': 36},
            {'name': 'packed3', 'shape': [1, 5], 'offset': 40, 'encoding': 'packed', 'bits': 3, 'group': 2, 'scaleOffset': 44},
            {'name': 'nan', 'shape': [1, 1], 'offset': 50, 'encoding': 'packed', 'bits': 4, 'group': 1, 'scaleOffset': 52}
        ],
        'config': {'kind': 'single'}
    };
    const model = loadModel(manifest, buffer);
    assert.equal(model.manifest, manifest);
    assert.equal(model.config, manifest.config);
    assert.equal(model.remap.size, 0);
    assert.deepEqual(Array.from(model.tensors.float.data), [0.5, -1.25, 3]);
    assert.deepEqual(model.tensors.float.shape, [3]);
    assert.deepEqual(Array.from(model.tensors.int8.data), [0.5, -1, 1.5, 254, -256, 0]);
    assert.deepEqual(Array.from(model.tensors.packed4.data), [-2, 0, 1.75, -3, 3, -6]);
    const subnormal = -(2 ** -24);
    assert.deepEqual(Array.from(model.tensors.packed3.data), [4.5, -6, Math.fround(subnormal), -0, -Infinity]);
    assert(Number.isNaN(model.tensors.nan.data[0]));
});


test('loadModel, remap', () => {
    // The manifest's token remap, as vocabulary ids
    const manifest = {'vocab': ['<pad>', '<unk>', '<bos>', '_'], 'remap': {'<unk>': '_'}, 'tensors': [], 'config': {}};
    const model = loadModel(manifest, new ArrayBuffer(0));
    assert.deepEqual([...model.remap], [[1, 3]]);
    assert.deepEqual(model.tensors, {});
});


test('speakFrames, multi', () => {
    const {frames, eq, sentences: parts} = speakFrames(multi, 'Hello, world. How are you?', {'speaker': 0});
    assert(frames.length > 100);
    assert(frames.every((frame) => frame instanceof Float64Array && frame.length === NAMES.length && frame.every(Number.isFinite)));
    assert.equal(eq.length, 24);
    assert.equal(parts.length, 2);
    assert.equal(parts[0].frameStart, 0);
    assert(parts[1].frameStart > 0);
    assert.equal(parts[0].ids[0], 2);
    assert.equal(parts[0].durations.length, parts[0].ids.length);

    // The end frame repeats the last frame; the frames are 5 ms, two per 10 ms duration
    assert.deepEqual(Array.from(frames[frames.length - 1]), Array.from(frames[frames.length - 2]));
    const durationTotal = parts.reduce((sum, part) => sum + part.durations.reduce((total, duration) => total + duration, 0), 0);
    assert.equal(frames.length, 2 * durationTotal + 1);

    // The jitter and shimmer are the model's
    assert(frames.every((frame) => frame[NAMES.indexOf('jitter')] === multi.config.jitter));
    assert(frames.every((frame) => frame[NAMES.indexOf('shimmer')] === multi.config.shimmer));
});


test('speakFrames, speakers', () => {
    const text = 'Call me Ishmael.';
    const female = speakFrames(multi, text, {'speaker': 0});
    const male = speakFrames(multi, text, {'speaker': 1});
    assert.notDeepEqual(plain(female.frames), plain(male.frames));
    assert.notDeepEqual(female.eq, male.eq);

    // The default speaker is the manifest's, else the first
    assert.equal(multi.manifest.defaultSpeaker, 1);
    assert.deepEqual(plain(speakFrames(multi, text).frames), plain(male.frames));
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': null}).frames), plain(male.frames));
    const noDefault = {...multi, 'manifest': {...multi.manifest, 'defaultSpeaker': undefined}};
    assert.deepEqual(plain(speakFrames(noDefault, text).frames), plain(female.frames));

    // A blend of one speaker is the speaker (ids may be strings; the weights are normalized)
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': {'0': 2}}).frames), plain(female.frames));
    const blend = speakFrames(multi, text, {'speaker': {'0': 1, '1': 1}});
    assert.notDeepEqual(plain(blend.frames), plain(female.frames));
    assert.deepEqual(blend.eq, female.eq.map((value, ix) => 0.5 * value + 0.5 * male.eq[ix]));

    // Invalid speakers
    for (const speaker of [{'0': 0}, {'x': 1}, {'999': 1}, {'0': 1, '1': -0.5}, 1.5, -1, multi.config.speakers]) {
        assert.throws(() => speakFrames(multi, text, {speaker}), {'message': `invalid speaker ${JSON.stringify(speaker)}`});
    }
});


test('speakFrames, embedding and eq', () => {
    const text = 'Call me Ishmael.';
    const {speakerDim} = multi.config;
    const table = multi.tensors['speaker.weight'].data;

    // A raw speaker embedding replaces the speaker's (the channel equalizer stays the speaker's)
    const embedding = Array.from(table.subarray(0, speakerDim));
    const female = speakFrames(multi, text, {'speaker': 0});
    const embedded = speakFrames(multi, text, {'speaker': 1, embedding});
    assert.deepEqual(plain(embedded.frames), plain(female.frames));
    assert.deepEqual(embedded.eq, speakFrames(multi, text, {'speaker': 1}).eq);
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'embedding': null}).frames), plain(female.frames));
    assert.throws(() => speakFrames(multi, text, {'embedding': [1, 2]}), {
        'message': `the speaker embedding must have ${speakerDim} values`
    });

    // A channel equalizer in place of the speaker's
    const eq = new Array(24).fill(1.5);
    const equalized = speakFrames(multi, text, {'speaker': 0, eq});
    assert.equal(equalized.eq, eq);
    assert.deepEqual(plain(equalized.frames), plain(female.frames));
});


test('speakFrames, styles', () => {
    const text = 'Call me Ishmael.';
    const neutral = plain(speakFrames(multi, text, {'speaker': 0}).frames);
    const happy = plain(speakFrames(multi, text, {'speaker': 0, 'style': {'emotion': 'happy'}}).frames);
    assert.notDeepEqual(happy, neutral);

    // The intensity defaults to 1; a blend of one emotion is the emotion; neutral is no style
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': {'emotion': 'happy', 'intensity': 1}}).frames), happy);
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': {'happy': 1}}).frames), happy);
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': {'emotion': 'neutral', 'intensity': 1}}).frames), neutral);
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': {'neutral': 0.5}}).frames), neutral);
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': null}).frames), neutral);
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': {'happy': 0}}).frames), neutral);
    const mixed = plain(speakFrames(multi, text, {'speaker': 0, 'style': {'happy': 0.5, 'sad': 0.2}}).frames);
    assert.notDeepEqual(mixed, happy);
    assert.notDeepEqual(plain(speakFrames(multi, text, {'speaker': 0, 'style': {'emotion': 'happy', 'intensity': 0.5}}).frames), happy);

    // An unknown emotion
    assert.throws(() => speakFrames(multi, text, {'style': {'emotion': 'bored'}}), {'message': 'unknown style bored'});
});


test('speakFrames, rate and pauses', () => {
    const text = 'Call me Ishmael, said he.';
    const normal = speakFrames(multi, text, {'speaker': 0});
    const slow = speakFrames(multi, text, {'speaker': 0, 'rate': 1.5});
    assert(slow.frames.length > 1.3 * normal.frames.length);

    // The pause threshold - no pauses, or a pause at every word boundary
    const noPauses = speakFrames(multi, text, {'speaker': 0, 'pauseThreshold': 1});
    const allPauses = speakFrames(multi, text, {'speaker': 0, 'pauseThreshold': 0});
    assert(noPauses.frames.length < normal.frames.length);
    assert(allPauses.frames.length > normal.frames.length);

    // A zero rate - no frames
    assert.deepEqual(speakFrames(multi, text, {'rate': 0}).frames, []);

    // No words - no frames
    const empty = speakFrames(multi, '');
    assert.deepEqual(empty.frames, []);
    assert.deepEqual(empty.sentences, []);
});


test('speakFrames, durations round half to even', () => {
    // A model whose every phone is 1 frame and no pause - at half rate, each 0.5: the running totals 0.5, 1, 1.5, 2, ...
    // round half to even (0, 1, 2, 2, 2, 3, 4, 4, ...)
    const model = loadSlt();
    model.tensors['duration_out.weight'].data.fill(0);
    model.tensors['duration_out.bias'].data.set([Math.log(2), -20, 0]);
    const {sentences: [sentence]} = speakFrames(model, 'Shapes.', {'rate': 0.5});
    const isPhone = sentence.ids.map((id) => id >= 9);
    let total = 0;
    let previous = 0;
    const expected = isPhone.map((phone) => {
        total += phone ? 0.5 : 0;
        const end = Math.round(total) % 2 === 1 && total % 1 === 0.5 ? Math.round(total) - 1 : Math.round(total);
        const duration = end - previous;
        previous = end;
        return duration;
    });
    assert.deepEqual(sentence.durations, expected);
    assert.deepEqual(sentence.durations.slice(1, 6), [0, 1, 1, 0, 0]);

    // A padding token (by the remap) takes no time
    const padded = loadSlt();
    padded.tensors['duration_out.weight'].data.fill(0);
    padded.tensors['duration_out.bias'].data.set([Math.log(3), -20, 0]);
    const plainDurations = speakFrames(padded, 'Shapes.').sentences[0].durations;
    assert(plainDurations.slice(1, -1).every((duration) => duration === 2));
    const [, firstPhone] = speakFrames(padded, 'Shapes.').sentences[0].ids;
    padded.remap.set(firstPhone, 0);
    const paddedDurations = speakFrames(padded, 'Shapes.').sentences[0].durations;
    assert.equal(paddedDurations[1], 0);
    assert.deepEqual(paddedDurations.slice(2), plainDurations.slice(2));
});


test('speakFrames, single-speaker model', () => {
    // The slt export - its own channel equalizer, and its bandwidth scale (0.6) applied to the frames
    const text = 'Call me Ishmael. "Was it?"';
    const spoken = speakFrames(slt, text);
    assert(spoken.frames.length > 100);
    assert.equal(spoken.eq, slt.manifest.eq);
    assert.equal(slt.manifest.inference.bandwidthScale, 0.6);
    const unscaled = speakFrames(slt, text, {'bandwidthScale': 1});
    assert.deepEqual(plain(spoken.frames), plain(bandwidthScale(unscaled.frames, 0.6)));
    assert.equal(spoken.sentences.length, 2);

    // Without the manifest's inference defaults, no bandwidth scale
    const bare = loadSlt();
    delete bare.manifest.inference;
    assert.deepEqual(plain(speakFrames(bare, text).frames), plain(unscaled.frames));
    bare.manifest.inference = {};
    assert.deepEqual(plain(speakFrames(bare, text).frames), plain(unscaled.frames));

    // A missing tensor
    const broken = loadSlt();
    delete broken.tensors['encoder_norm.weight'];
    assert.throws(() => speakFrames(broken, text), {'message': 'model tensor encoder_norm.weight not found'});
});


test('streamFrames', () => {
    // The chunks, in order, are speakFrames's frames - the network runs as they're taken
    const text = 'It was the best of times, it was the worst of times. It was the age of wisdom, it was the age of foolishness.';
    const whole = speakFrames(multi, text, {'speaker': 0});
    const stream = streamFrames(multi, text, {'speaker': 0});
    assert.deepEqual(stream.eq, whole.eq);
    assert.deepEqual(stream.sentences, []);
    const chunks = [...stream.chunks];
    assert(chunks.length > 10);
    // (32 decoder steps of two frames a chunk - each sentence's last chunk shorter)
    assert(chunks.every((chunk) => chunk.length <= 64));
    assert(chunks.filter((chunk) => chunk.length === 64).length >= chunks.length - 3);
    assert.equal(chunks[chunks.length - 1].length, 1);
    assert.deepEqual(plain(chunks.flat()), plain(whole.frames));
    assert.deepEqual(stream.sentences, whole.sentences);

    // The single-speaker model's chunks are bandwidth-scaled too
    assert.deepEqual(plain([...streamFrames(slt, text).chunks].flat()), plain(speakFrames(slt, text).frames));
});


test('sentences', () => {
    const vocabulary = ['<pad>', '<unk>', '<bos>', '_', '<comma>', '<period>', '<question>', '<wh>', '<exclamation>', 'AA0'];
    // Each sentence starts with <bos> and ends at a sentence end
    assert.deepEqual(sentences([2, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8], vocabulary), [[2, 9, 4, 9, 5], [2, 9, 6], [2, 9, 7], [2, 9, 8]]);

    // Without a leading <bos>, and a trailing sentence without an end
    assert.deepEqual(sentences([9, 5, 9, 4, 9], vocabulary), [[2, 9, 5], [2, 9, 4, 9]]);
    assert.deepEqual(sentences([2], vocabulary), []);
    assert.deepEqual(sentences([], vocabulary), []);

    // Text ending in a comma - its last sentence has no sentence end
    const parts = speakFrames(multi, 'Hello there,').sentences;
    assert.equal(parts.length, 1);
    assert.equal(multi.manifest.vocab[parts[0].ids[parts[0].ids.length - 1]], '<comma>');
});
