// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {FLOOR_DB, NAMES, PARAMETERS, TRACT_FREQUENCIES} from '../lib/params.js';
import {
    applyTransforms, effort, frication, level, pitchRange, pitchScale, press, rdScale, shelf, tractScale, whisper
} from '../lib/transforms.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A frame of parameter values (the defaults, with some voicing)
const IX = Object.fromEntries(NAMES.map((name, ix) => [name, ix]));
function frame(values = {}) {
    const out = Float64Array.from(PARAMETERS, (parameter) => parameter.default);
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


test('transforms.pitchScale', () => {
    const [output] = pitchScale([frame()], 1.25);
    assert.deepEqual(changes(frame(), output), {'f0': [120, 150]});
});


test('transforms.pitchRange', () => {
    // Every frame's pitch moved from the voiced frames' mean pitch (log scale) - 80 and 180 Hz around 120 Hz; an unvoiced
    // frame (below -40 dB of voicing) widened too, but not counted
    const frames = [frame({'f0': 80}), frame({'f0': 180}), frame({'f0': 300, 'av': -60})];
    const [low, high, unvoiced] = pitchRange(frames, 2).map((item) => item[IX.f0]);
    assert(Math.abs(low - 80 * 80 / 120) < 1e-9 && Math.abs(high - 180 * 180 / 120) < 1e-9);
    assert(Math.abs(unvoiced - 300 * 300 / 120) < 1e-9);
    assert.deepEqual(changes(frames[0], pitchRange(frames, 2)[0]), {'f0': [80, low]});

    // No voiced frames - unchanged
    const silent = [frame({'av': -90})];
    assert.deepEqual(pitchRange(silent, 2).map((item) => Array.from(item)), silent.map((item) => Array.from(item)));
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
    const [half] = whisper([input], {'intensity': 0.5});
    assert.equal(half[IX.av], -6 + 20 * Math.log10(0.5));
    assert.equal(half[IX.ah], 10 * Math.log10(10 ** (-40 / 10) + 10 ** ((-6 - 28 + 20 * Math.log10(0.5)) / 10)));
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
});


test('transforms.press', () => {
    // Effort's voice quality without its loudness - brighter and more pressed, the levels unchanged
    const [pressed] = press([frame()], 6);
    assert.deepEqual(changes(frame(), pressed), {'shelf': [0, 3], 'rd': [1, 0.98 ** 6]});

    // Rd stays in 0.3 - 2.7
    assert.equal(press([frame({'rd': 0.31})], 30)[0][IX.rd], 0.3);
    assert.equal(press([frame({'rd': 2.6})], -30)[0][IX.rd], 2.7);
});


test('transforms.level', () => {
    // Every source louder or quieter (not the frication filter's gains), a silent source silent, nothing below the floor
    const before = frame({'af': FLOOR_DB, 'ab': -50});
    const [quieter] = level([before], -6);
    assert.deepEqual(changes(before, quieter), {'av': [-6, -12], 'ah': [-40, -46]});
    assert.equal(level([frame({'ah': FLOOR_DB + 2})], -10)[0][IX.ah], FLOOR_DB);
    assert.deepEqual(changes(frame({'af': -30}), level([frame({'af': -30})], 3)[0]), {'av': [-6, -3], 'ah': [-40, -37], 'af': [-30, -27]});
});


test('transforms.frication', () => {
    // The frication noise louder or quieter, a silent frication silent, nothing below the floor
    const before = frame({'af': -30});
    assert.deepEqual(changes(before, frication([before], 4)[0]), {'af': [-30, -26]});
    assert.deepEqual(changes(frame(), frication([frame()], 4)[0]), {});
    assert.equal(frication([frame({'af': FLOOR_DB + 2})], -10)[0][IX.af], FLOOR_DB);
});


test('transforms.rdScale', () => {
    // The pulse's Rd scaled, within its range, nothing else
    const before = frame({'rd': 1.5});
    assert.deepEqual(changes(before, rdScale([before], 1.2)[0]), {'rd': [1.5, 1.5 * 1.2]});
    assert.equal(rdScale([frame({'rd': 2.5})], 1.2)[0][IX.rd], 2.7);
    assert.equal(rdScale([frame({'rd': 0.35})], 0.5)[0][IX.rd], 0.3);
});


test('transforms.shelf', () => {
    // The source shelf raised or lowered, nothing else
    const before = frame({'shelf': 12});
    assert.deepEqual(changes(before, shelf([before], -5)[0]), {'shelf': [12, 7]});
    assert.deepEqual(changes(before, shelf([before], 0)[0]), {});
});


test('transforms.applyTransforms', () => {
    const frames = [frame(), frame({'f0': 200})];

    // No transforms - the frames themselves
    assert.equal(applyTransforms(frames), frames);
    assert.equal(applyTransforms(frames, []), frames);

    // Each in turn, in a transform's fixed order (pitch, tract, whisper, effort, press, level, frication, Rd, shelf)
    const shaped = whisper(tractScale(pitchScale(frames, 1.1), 0.9), {'intensity': 0.5});
    const expected = shelf(rdScale(frication(level(press(effort(shaped, 3), 2), -4), 1), 1.1), -5);
    const applied = applyTransforms(frames, [
        {'shelf': -5, 'rdScale': 1.1, 'frication': 1, 'level': -4, 'press': 2, 'effort': 3, 'whisper': 0.5, 'tractScale': 0.9,
            'pitchScale': 1.1}
    ]);
    assert.deepEqual(applied.map((item) => Array.from(item)), expected.map((item) => Array.from(item)));
    assert.deepEqual(applyTransforms(frames, [{'pitchRange': 1.5}]).map((item) => Array.from(item)),
        pitchRange(frames, 1.5).map((item) => Array.from(item)));
    const listed = applyTransforms(frames, [{'pitchScale': 1.1}, {'pitchScale': 1.1}]);
    assert.deepEqual(listed.map((item) => item[IX.f0]), [120 * 1.1 * 1.1, 200 * 1.1 * 1.1]);
    assert.equal(applyTransforms(frames, [{}]), frames);
});
