// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as voiceModule from '../lib/models/voiceModel.js';
import {loadPackagedModel, packagedWeights} from '../lib/package.js';
import {Buffer} from 'node:buffer';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// Base64 of bytes, in lines (as picovox_train/package.py wraps them)
function base64Lines(bytes) {
    return Buffer.from(bytes).toString('base64').replace(/.{1,8}/g, '$&\n');
}


// A small model's manifest and weights - a float32 tensor
const weights = new Uint8Array(new Float32Array([1.5, -2, 0.25, 8]).buffer);
const manifest = {'vocab': [], 'tensors': [{'name': 'weight', 'shape': [2, 2], 'offset': 0}], 'config': {'kind': 'single'}};



test('package.packagedWeights', async () => {
    // Raw weights
    assert.deepEqual(new Uint8Array(await packagedWeights({'encoding': 'raw', 'weights': base64Lines(weights)})), weights);

    // An unknown encoding
    await assert.rejects(packagedWeights({'encoding': 'deflate-raw', 'weights': ''}), {'message': 'unknown package encoding deflate-raw'});
});


test('package.loadPackagedModel', async () => {
    const packaged = await loadPackagedModel({manifest, 'encoding': 'raw', 'weights': base64Lines(weights)});
    assert.equal(packaged.manifest, manifest);
    assert.deepEqual(Array.from(packaged.tensors.weight.data), [1.5, -2, 0.25, 8]);

    // The package's voice model
    const model = await loadPackagedModel(voiceModule);
    assert.equal(Object.keys(model.tensors).length, voiceModule.manifest.tensors.length);
    assert.equal(model.config.kind, 'multi');
});
