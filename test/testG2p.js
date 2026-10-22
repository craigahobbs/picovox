// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {base64Bytes, loadG2p, unpackTensors} from '../lib/g2p.js';
import {Buffer} from 'node:buffer';
import {strict as assert} from 'node:assert';
import {g2pModel} from '../lib/models/g2pModel.js';
import test from 'node:test';


// Bytes as base64
function bytesBase64(bytes) {
    return Buffer.from(bytes).toString('base64');
}


// Compress or decompress bytes with a built-in stream ('deflate-raw')
async function streamBytes(bytes, stream) {
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer());
}


// A per-row-scaled model's weight bytes in the other layout ("packed" <-> "bytes") - the scales are copied, the values
// re-encoded
function otherLayout(model, bytes) {
    const shapes = model.tensors.map(({shape}) => (shape.length === 1 ? [1, shape[0]] : shape));
    const scaleBytes = shapes.reduce((sum, [rows]) => sum + 2 * rows, 0);
    const parts = [bytes.slice(0, scaleBytes)];
    let offset = scaleBytes;
    for (const [index, {bits}] of model.tensors.entries()) {
        const count = shapes[index][0] * shapes[index][1];
        let values = [];
        if (model.layout === 'bytes') {
            values = Array.from(bytes.subarray(offset, offset + count));
            offset += count;
            const packed = new Uint8Array(Math.ceil(count * bits / 8));
            for (const [valueIndex, value] of values.entries()) {
                for (let bitIndex = 0; bitIndex < bits; bitIndex++) {
                    const bit = valueIndex * bits + bitIndex;
                    packed[bit >> 3] |= ((value >> bitIndex) & 1) << (bit & 7);
                }
            }
            parts.push(packed);
        } else {
            for (let valueIndex = 0; valueIndex < count; valueIndex++) {
                let value = 0;
                for (let bitIndex = 0; bitIndex < bits; bitIndex++) {
                    const bit = valueIndex * bits + bitIndex;
                    value |= ((bytes[offset + (bit >> 3)] >> (bit & 7)) & 1) << bitIndex;
                }
                values.push(value);
            }
            offset += Math.ceil(count * bits / 8);
            parts.push(new Uint8Array(values));
        }
    }
    const result = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
    let position = 0;
    for (const part of parts) {
        result.set(part, position);
        position += part.length;
    }
    return result;
}


// Assert two arrays are equal, element by element - without assert.deepEqual's diff, which for large arrays that
// differ takes enormous memory
function assertArraysEqual(actual, expected, message) {
    assert.equal(actual.length, expected.length, message);
    const index = actual.findIndex((value, ix) => !Object.is(value, expected[ix]));
    assert.equal(index, -1, `${message} differ at ${index}`);
}


// Words to compare models on
const words = ['queequeg', 'longbourn', 'netherfield', 'clerval', 'raskolnikov', 'pumblechook', 'harpooneer', 'a', 'x', 'o\'clock'];


test('g2p.base64Bytes', () => {
    assert.deepEqual([...base64Bytes('AAEC\n/w==')], [0, 1, 2, 255]);
    assert.deepEqual([...base64Bytes('')], []);
});


test('g2p.unpackTensors, packed per row', () => {
    // A 4-bit 2x3 matrix (zero is 7) and an 8-bit vector (zero is 127); bfloat16 scales 1, 0.5, and 2
    const model = {
        'groupSize': 0,
        'layout': 'packed',
        'tensors': [{'name': 'matrix', 'shape': [2, 3], 'bits': 4}, {'name': 'vector', 'shape': [2], 'bits': 8}]
    };
    const bytes = new Uint8Array([
        0x80, 0x3f, 0x00, 0x3f, 0x00, 0x40,
        // Matrix values 8, 6, 14 / 0, 7, 9 - nibbles, low first
        0x68, 0x0e, 0x97,
        // Vector values 128, 125
        128, 125
    ]);
    const tensors = unpackTensors(model, bytes);
    assert.deepEqual([...tensors.get('matrix')], [1, -1, 7, -3.5, 0, 1]);
    assert.deepEqual([...tensors.get('vector')], [2, -4]);
});


test('g2p.unpackTensors, bytes per group', () => {
    // A 1x3 matrix of 5-bit values (zero is 15) in groups of 2 - scales 1 and 0.25
    const model = {'groupSize': 2, 'layout': 'bytes', 'tensors': [{'name': 'matrix', 'shape': [1, 3], 'bits': 5}]};
    const bytes = new Uint8Array([0x80, 0x3f, 0x80, 0x3e, 16, 13, 30]);
    assert.deepEqual([...unpackTensors(model, bytes).get('matrix')], [1, -2, 3.75]);
});


test('g2p.loadG2p', async () => {
    const pronounce = await loadG2p(g2pModel);
    const phoneSet = new Set(g2pModel.phones.slice(2));
    for (const word of words) {
        const phones = pronounce(word);
        assert(phones.length > 0);
        assert(phones.every((phone) => phoneSet.has(phone)), word);

        // Cached - the same array
        assert.equal(pronounce(word), phones);
    }

    // Letters outside the model's alphabet are skipped, and words are cut to the model's longest
    assert.deepEqual(pronounce('café'), pronounce('caf'));
    assert.deepEqual(pronounce('éé'), []);
    const long = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz';
    assert.deepEqual(pronounce(long), pronounce(long.slice(0, g2pModel.maxLetters)));
});


// The shipped model reads heteronyms as readers do - its targets for these are the LibriTTS-R readers' majority
// pronunciation, not the CMU dictionary's first (lead "leed", not the metal "led")
test('g2p.loadG2p, readers\' pronunciations', async () => {
    const pronounce = await loadG2p(g2pModel);
    const expected = {
        'lead': 'L IY1 D',
        'live': 'L IH1 V',
        'lives': 'L AY1 V Z',
        'wind': 'W IH1 N D',
        'wound': 'W UW1 N D',
        'tears': 'T IH1 R Z',
        'content': 'K AH0 N T EH1 N T',
        'record': 'R EH1 K ER0 D',
        // Function words keep the dictionary's full form - their reduction is contextual
        'can': 'K AE1 N',
        'and': 'AH0 N D'
    };
    for (const [word, phones] of Object.entries(expected)) {
        assert.equal(pronounce(word).join(' '), phones, word);
    }
});


// A listener's reports - gravest (the dictionary's G R AE1 V AH0 S T, fixed - data/nn/g2p-fixes.json), mindset, and
// paprika (the dictionary's second, P AH0 P R IY1 K AH0)
test('g2p.loadG2p, a listener\'s reports', async () => {
    const pronounce = await loadG2p(g2pModel);
    const expected = {
        'gravest': 'G R EY1 V AH0 S T',
        'graver': 'G R EY1 V ER0',
        'mindset': 'M AY1 N D S EH2 T',
        'mindsets': 'M AY1 N D S EH2 T S',
        'paprika': 'P AH0 P R IY1 K AH0'
    };
    for (const [word, phones] of Object.entries(expected)) {
        assert.equal(pronounce(word).join(' '), phones, word);
    }
});


// Heteronyms - the ambiguity head picks out the words whose reading the context decides, and the context network chooses
// the reading the G2P pronounces
test('g2p.loadG2p, heteronym readings', async () => {
    const pronounce = await loadG2p(g2pModel);
    assert.equal(pronounce.ambiguous('read'), true);
    assert.equal(pronounce.ambiguous('lives'), true);
    assert.equal(pronounce.ambiguous('the'), false);
    assert.equal(pronounce.ambiguous('book'), false);
    assert.equal(pronounce.ambiguous('éé'), false);

    // Each reading's pronunciation, cached per reading; a reading the model lacks is the default
    const readings = [0, 1].map((reading) => pronounce('read', reading).join(' '));
    assert.deepEqual(readings.toSorted(), ['R EH1 D', 'R IY1 D']);
    assert.equal(pronounce('read', 1), pronounce('read', 1));
    assert.equal(pronounce('read', 9), pronounce('read', 0));

    // The reading of the word at an index in a text's items (its words and phrase-end marks), cached per window
    const reading = (items, index) => pronounce(items[index], pronounce.chooseReading(items, index)).join(' ');
    assert.equal(reading(['i', 'will', 'read', 'it', 'tomorrow', '.'], 2), 'R IY1 D');
    assert.equal(reading(['i', 'will', 'read', 'it', 'tomorrow', '.'], 2), 'R IY1 D');
    assert.equal(reading(['she', 'had', 'read', 'the', 'letter', '.'], 2), 'R EH1 D');
    assert.equal(reading(['their', 'lives', 'were', 'quiet', '.'], 1), 'L AY1 V Z');
    assert.equal(reading(['she', 'lives', 'here', '.'], 1), 'L IH1 V Z');

    // At the text's edges, and with characters outside the context network's alphabet
    assert(pronounce.chooseReading(['read'], 0) < g2pModel.context.slots);
    assert(pronounce.chooseReading(['éé', 'read', '.'], 1) < g2pModel.context.slots);
});


test('g2p.loadG2p, no context network', async () => {
    // A model without the context section has no ambiguous words; one without the ambiguity head never finds one
    const pronounce = await loadG2p({...g2pModel, 'context': undefined});
    assert.equal(pronounce.ambiguous, undefined);
    assert.deepEqual(pronounce('read'), (await loadG2p(g2pModel))('read'));
    const headTensors = new Set(['ambiguity.weight', 'ambiguity_layer.bias']);
    const rename = (tensor) => (headTensors.has(tensor.name) ? {...tensor, 'name': `unused ${tensor.name}`} : tensor);
    const tensors = g2pModel.tensors.map(rename);
    const headless = await loadG2p({...g2pModel, tensors});
    assert.equal(headless.ambiguous('read'), false);
    const readingless = await loadG2p({...g2pModel, 'readings': undefined, 'context': undefined});
    assert.equal(readingless('read', 1), readingless('read', 0));
});


test('g2p.loadG2p, context network not deflated', async () => {
    const {context} = g2pModel;
    const bytes = await streamBytes(base64Bytes(context.weights), new DecompressionStream('deflate-raw'));
    const other = {...g2pModel, 'context': {...context, 'deflate': false, 'weights': bytesBase64(bytes)}};
    const [pronounce, pronounceOther] = await Promise.all([loadG2p(g2pModel), loadG2p(other)]);
    const items = ['he', 'touched', 'a', 'live', 'wire', '.'];
    assert.equal(pronounceOther.chooseReading(items, 3), pronounce.chooseReading(items, 3));
});


// The shipped model's format - per-row scales, a byte per value (6-bit values deflate smaller than bit-packed), deflated
test('g2p.g2pModel format', () => {
    assert.equal(g2pModel.design, 'ar');
    assert.equal(g2pModel.groupSize, 0);
    assert.equal(g2pModel.layout, 'bytes');
    assert.equal(g2pModel.deflate, true);
});


test('g2p.loadG2p, unsupported design', async () => {
    await assert.rejects(loadG2p({...g2pModel, 'design': 'ctc'}), {'message': 'Unsupported G2P model design ctc'});
});


test('g2p.loadG2p, deflated and not', async () => {
    const bytes = base64Bytes(g2pModel.weights);
    const other = {...g2pModel, 'deflate': false, 'weights': bytesBase64(await streamBytes(bytes, new DecompressionStream('deflate-raw')))};
    const [pronounce, pronounceOther] = await Promise.all([loadG2p(g2pModel), loadG2p(other)]);
    assert.deepEqual(words.map(pronounceOther), words.map(pronounce));
});


test('g2p.loadG2p, both layouts', async () => {
    const raw = await streamBytes(base64Bytes(g2pModel.weights), new DecompressionStream('deflate-raw'));
    const other = {
        ...g2pModel,
        'layout': {'bytes': 'packed', 'packed': 'bytes'}[g2pModel.layout],
        'deflate': false,
        'weights': bytesBase64(otherLayout(g2pModel, raw))
    };
    // Converting back gives the same bytes
    const otherRaw = base64Bytes(other.weights);
    assertArraysEqual(otherLayout(other, otherRaw), raw, 'weight bytes');
    const tensors = unpackTensors(g2pModel, raw);
    const otherTensors = unpackTensors(other, otherRaw);
    for (const [name, values] of tensors) {
        assertArraysEqual(otherTensors.get(name), values, name);
    }
    const [pronounce, pronounceOther] = await Promise.all([loadG2p(g2pModel), loadG2p(other)]);
    assert.deepEqual(words.map(pronounceOther), words.map(pronounce));
});
