// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import * as voiceModel from '../lib/models/voiceModel.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// The trained voice model's parameter count (the speaker and equalizer tables included)
const PARAMETER_COUNT = 282627;


test('models.voiceModel', () => {
    // The generated voice model module - its manifest (the multi-speaker model's distilled d64 student, with the package's
    // twelve voices' speakers, the default first) and its deflated weights as base64
    assert.equal(voiceModel.manifest.kind, 'multi');
    assert.equal(voiceModel.manifest.parameterCount, PARAMETER_COUNT);
    assert.equal(voiceModel.manifest.speakers.length, 12);
    assert.deepEqual(voiceModel.manifest.speakers[0], {...voiceModel.manifest.speakers[0], 'id': 0, 'key': 'male'});
    assert.equal(voiceModel.encoding, 'deflate-raw');
    assert.match(voiceModel.weights.slice(0, 1000), /^[\sA-Za-z0-9+/=]+$/);
});
