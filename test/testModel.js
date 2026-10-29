// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as voiceModule from '../lib/models/voiceModel.js';
import {loadModel, sentences, speakFrames, streamFrames} from '../lib/model.js';
import {NAMES} from '../lib/params.js';
import {strict as assert} from 'node:assert';
import {g2pModel} from '../lib/models/g2pModel.js';
import {loadG2p} from '../lib/g2p.js';
import {loadPackagedModel} from '../lib/package.js';
import {setPronouncer} from '../lib/lexicon.js';
import test from 'node:test';


// The G2P network pronounces, as in the package
setPronouncer(await loadG2p(g2pModel));


// The package's model
const multi = await loadPackagedModel(voiceModule);


// Frames as plain arrays (for comparisons)
function plain(frames) {
    return frames.map((frame) => Array.from(frame));
}


test('loadModel, encodings', () => {
    // A float32 tensor and an int8 tensor with per-row scales - in one buffer
    const buffer = new ArrayBuffer(32);
    new Float32Array(buffer, 0, 3).set([0.5, -1.25, 3]);
    new Int8Array(buffer, 16, 6).set([1, -2, 3, 127, -128, 0]);
    new Float32Array(buffer, 24, 2).set([0.5, 2]);

    const manifest = {
        'vocab': ['<pad>', '<unk>', '<bos>', '_'],
        'tensors': [
            {'name': 'float', 'shape': [3], 'offset': 0},
            {'name': 'int8', 'shape': [2, 3], 'offset': 16, 'encoding': 'int8', 'scaleOffset': 24}
        ],
        'config': {}
    };
    const model = loadModel(manifest, buffer);
    assert.equal(model.manifest, manifest);
    assert.equal(model.config, manifest.config);
    assert.equal(model.remap.size, 0);
    assert.deepEqual(Array.from(model.tensors.float.data), [0.5, -1.25, 3]);
    assert.deepEqual(model.tensors.float.shape, [3]);
    assert.deepEqual(Array.from(model.tensors.int8.data), [0.5, -1, 1.5, 254, -256, 0]);
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
    const male = speakFrames(multi, text, {'speaker': 0});
    const female = speakFrames(multi, text, {'speaker': 1});
    assert.notDeepEqual(plain(female.frames), plain(male.frames));
    assert.notDeepEqual(female.eq, male.eq);

    // The default speaker is the manifest's (ids may be strings)
    assert.equal(multi.manifest.defaultSpeaker, 0);
    assert.deepEqual(plain(speakFrames(multi, text).frames), plain(male.frames));
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': null}).frames), plain(male.frames));
    assert.deepEqual(plain(speakFrames(multi, text, {'speaker': '1'}).frames), plain(female.frames));

    // Invalid speakers
    for (const speaker of [{'0': 1}, 'x', 1.5, -1, multi.config.speakers]) {
        assert.throws(() => speakFrames(multi, text, {speaker}), {'message': `invalid speaker ${JSON.stringify(speaker)}`});
    }
});


test('speakFrames, rate and pauses', () => {
    const text = 'Call me Ishmael, said he.';
    const normal = speakFrames(multi, text, {'speaker': 0});
    const slow = speakFrames(multi, text, {'speaker': 0, 'rate': 1.5});
    assert(slow.frames.length > 1.3 * normal.frames.length);

    // A zero rate - no frames
    assert.deepEqual(speakFrames(multi, text, {'rate': 0}).frames, []);

    // No words - no frames
    const empty = speakFrames(multi, '');
    assert.deepEqual(empty.frames, []);
    assert.deepEqual(empty.sentences, []);
});


test('speakFrames, durations round half to even', async () => {
    // A model whose every phone is 1 frame and no pause - at half rate, each 0.5: the running totals 0.5, 1, 1.5, 2, ...
    // round half to even (0, 1, 2, 2, 2, 3, 4, 4, ...)
    const model = await loadPackagedModel(voiceModule);
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
    const padded = await loadPackagedModel(voiceModule);
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


test('speakFrames, a missing tensor', () => {
    const broken = {...multi, 'tensors': {...multi.tensors}};
    delete broken.tensors['encoder_norm.weight'];
    assert.throws(() => speakFrames(broken, 'Call me Ishmael.'), {'message': 'model tensor encoder_norm.weight not found'});
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
