// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import * as voiceModel from '../lib/models/voiceModel.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


test('models.voiceModel', () => {
    // The generated voice model module - its manifest (the multi-speaker model, with the app's two speakers) and its
    // deflated weights as base64
    assert.equal(voiceModel.manifest.kind, 'multi');
    assert.equal(voiceModel.manifest.parameterCount, 965803);
    assert(voiceModel.manifest.speakers.some((speaker) => speaker.key === 'arctic-slt'));
    assert(voiceModel.manifest.speakers.some((speaker) => speaker.key === 'arctic-bdl'));
    assert.equal(voiceModel.encoding, 'deflate-raw');
    assert.match(voiceModel.weights.slice(0, 1000), /^[\sA-Za-z0-9+/=]+$/);
});
