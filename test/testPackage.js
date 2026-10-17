// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as voiceModule from '../lib/models/voiceModel.js';
import {decodeBase64, loadPackagedModel, packagedWeights} from '../lib/package.js';
import {Buffer} from 'node:buffer';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// Bytes compressed with a built-in stream ('deflate-raw')
async function deflate(bytes) {
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'))).arrayBuffer());
}


// Base64 of bytes, in lines (as nn/package.py wraps them)
function base64Lines(bytes) {
    return Buffer.from(bytes).toString('base64').replace(/.{1,8}/g, '$&\n');
}


// A small model's manifest and weights - a float32 tensor
const weights = new Uint8Array(new Float32Array([1.5, -2, 0.25, 8]).buffer);
const manifest = {'vocab': [], 'tensors': [{'name': 'weight', 'shape': [2, 2], 'offset': 0}], 'config': {'kind': 'single'}};


test('package.decodeBase64', () => {
    assert.deepEqual(decodeBase64('AAEC/w=='), Uint8Array.from([0, 1, 2, 255]));
    assert.deepEqual(decodeBase64(' AA\nEC\r\n/w==\t'), Uint8Array.from([0, 1, 2, 255]));
    assert.deepEqual(decodeBase64(''), new Uint8Array(0));
});


test('package.packagedWeights', async () => {
    // Raw and deflated weights
    assert.deepEqual(new Uint8Array(await packagedWeights({'encoding': 'raw', 'weights': base64Lines(weights)})), weights);
    const deflated = base64Lines(await deflate(weights));
    assert.deepEqual(new Uint8Array(await packagedWeights({'encoding': 'deflate-raw', 'weights': deflated})), weights);

    // An unknown encoding
    await assert.rejects(packagedWeights({'encoding': 'gzip', 'weights': ''}), {'message': 'unknown package encoding gzip'});
});


test('package.loadPackagedModel', async () => {
    for (const module of [
        {manifest, 'encoding': 'raw', 'weights': base64Lines(weights)},
        {manifest, 'encoding': 'deflate-raw', 'weights': base64Lines(await deflate(weights))}
    ]) {
        const model = await loadPackagedModel(module);
        assert.equal(model.manifest, manifest);
        assert.deepEqual(Array.from(model.tensors.weight.data), [1.5, -2, 0.25, 8]);
    }

    // The package's voice model
    const model = await loadPackagedModel(voiceModule);
    assert.equal(Object.keys(model.tensors).length, voiceModule.manifest.tensors.length);
    assert.equal(model.config.kind, 'multi');
});
