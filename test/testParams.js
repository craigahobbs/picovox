// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {DB_NAMES, DEFAULTS, EQ_BAND_COUNT, FRAME_RATE, NAMES, PARAMETERS, REF_RATE, TRACT_FREQUENCIES} from '../lib/params.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


test('params', () => {
    // The parameter set (generated from nn/params.py)
    assert.equal(FRAME_RATE, 200);
    assert.equal(REF_RATE, 44100);
    assert.equal(EQ_BAND_COUNT, 24);
    assert.equal(NAMES.length, 32);
    assert.deepEqual(NAMES.slice(0, 3), ['f0', 'av', 'rd']);
    assert.deepEqual(NAMES, PARAMETERS.map((parameter) => parameter.name));
    for (const parameter of PARAMETERS) {
        assert(['log', 'linear'].includes(parameter.scale));
        assert(parameter.low <= parameter.default && parameter.default <= parameter.high, parameter.name);
        assert.equal(DEFAULTS[parameter.name], parameter.default);
        assert(parameter.doc);
    }
    assert(DB_NAMES.every((name) => NAMES.includes(name)));
    assert(TRACT_FREQUENCIES.every((name) => NAMES.includes(name)));
});
