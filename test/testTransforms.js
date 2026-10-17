// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {DEFAULTS, FLOOR_DB, NAMES, TRACT_FREQUENCIES} from '../lib/params.js';
import {applyTransforms, bandwidthScale, effort, pitchScale, tractScale, whisper} from '../lib/transforms.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A frame of parameter values (the defaults, with some voicing)
const IX = Object.fromEntries(NAMES.map((name, ix) => [name, ix]));
function frame(values = {}) {
    const out = Float64Array.from(NAMES, (name) => DEFAULTS[name]);
    for (const [name, value] of Object.entries({'f0': 120, 'av': -6, 'ah': -40, 'jitter': 0.01, 'shimmer': 0.04, ...values})) {
        out[IX[name]] = value;
    }
    return out;
}


// The parameters a transform changed - {name: [before, after]}
function changes(before, after) {
    const out = {};
    for (const [ix, name] of NAMES.entries()) {
        if (before[ix] !== after[ix]) {
            out[name] = [before[ix], after[ix]];
        }
    }
    return out;
}


test('transforms.tractScale', () => {
    const input = frame();
    const [output] = tractScale([input], 1.1);
    assert.notEqual(output, input);
    assert.deepEqual(Object.keys(changes(input, output)).sort(), [...TRACT_FREQUENCIES].sort());
    for (const name of TRACT_FREQUENCIES) {
        assert.equal(output[IX[name]], input[IX[name]] * 1.1);
    }
});


test('transforms.bandwidthScale', () => {
    const input = frame();
    const [output] = bandwidthScale([input], 0.6);
    assert.deepEqual(Object.keys(changes(input, output)), ['b1', 'b2', 'b3', 'b4', 'b5']);
    assert.equal(output[IX.b1], input[IX.b1] * 0.6);
    assert.equal(input[IX.b1], DEFAULTS.b1);
});


test('transforms.pitchScale', () => {
    const [output] = pitchScale([frame()], 1.25);
    assert.deepEqual(changes(frame(), output), {'f0': [120, 150]});
});


test('transforms.whisper', () => {
    // Full intensity - no voicing, aspiration added 28 dB below the voicing, no jitter or shimmer
    const input = frame();
    const [full] = whisper([input]);
    assert.equal(full[IX.av], FLOOR_DB);
    assert.equal(full[IX.ah], 10 * Math.log10(10 ** (-40 / 10) + 10 ** ((-6 - 28) / 10)));
    assert.equal(full[IX.jitter], 0);
    assert.equal(full[IX.shimmer], 0);

    // Half intensity - the voicing 6 dB down, the aspiration also 6 dB lower
    const [half] = whisper([input], {'intensity': 0.5, 'breathDb': -20});
    assert.equal(half[IX.av], -6 + 20 * Math.log10(0.5));
    assert.equal(half[IX.ah], 10 * Math.log10(10 ** (-40 / 10) + 10 ** ((-6 - 20 + 20 * Math.log10(0.5)) / 10)));
    assert.equal(half[IX.jitter], 0.005);

    // An unvoiced frame gains no aspiration
    const silent = frame({'av': FLOOR_DB});
    const [unvoiced] = whisper([silent]);
    assert.equal(unvoiced[IX.ah], 10 * Math.log10(10 ** (-40 / 10) + 10 ** (FLOOR_DB / 10)));
});


test('transforms.effort', () => {
    // Louder, brighter, more pressed
    const [louder] = effort([frame()], 6);
    assert.equal(louder[IX.av], 0);
    assert.equal(louder[IX.shelf], 3);
    assert.equal(louder[IX.rd], 0.98 ** 6);

    // Rd stays in 0.3 - 2.7
    assert.equal(effort([frame({'rd': 0.31})], 30)[0][IX.rd], 0.3);
    assert.equal(effort([frame({'rd': 2.6})], -30)[0][IX.rd], 2.7);
    const [custom] = effort([frame()], 2, {'shelfPerDb': 1, 'rdPerDb': 0});
    assert.deepEqual([custom[IX.av], custom[IX.shelf], custom[IX.rd]], [-4, 2, 1]);
});


test('transforms.applyTransforms', () => {
    const frames = [frame(), frame({'f0': 200})];

    // No transforms - the frames themselves
    assert.equal(applyTransforms(frames), frames);
    assert.equal(applyTransforms(frames, []), frames);

    // Each in turn, in a transform's fixed order (bandwidth, pitch, tract, whisper, effort)
    const expected = effort(whisper(tractScale(pitchScale(bandwidthScale(frames, 0.8), 1.1), 0.9), {'intensity': 0.5}), 3);
    const applied = applyTransforms(frames, [{'effort': 3, 'whisper': 0.5, 'tractScale': 0.9, 'pitchScale': 1.1, 'bandwidthScale': 0.8}]);
    assert.deepEqual(applied.map((item) => Array.from(item)), expected.map((item) => Array.from(item)));
    const listed = applyTransforms(frames, [{'pitchScale': 1.1}, {'pitchScale': 1.1}]);
    assert.deepEqual(listed.map((item) => item[IX.f0]), [120 * 1.1 * 1.1, 200 * 1.1 * 1.1]);
    assert.equal(applyTransforms(frames, [{}]), frames);
});
