// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {LF_HARMONICS, LF_RD_MAX, LF_RD_MIN} from '../lib/params.js';
import {LF_RD_COUNT, lfHarmonics, lfPulse, lfTiming} from '../lib/lf.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


test('lf.lfTiming', () => {
    // picovox_train/lf.py's lf_timing
    assert.deepEqual(lfTiming(1), {'tp': 0.4843626347650026, 'te': 0.6500146558546335, 'ta': 0.038});

    // Rd is clamped to 0.3 - 2.7
    assert.deepEqual(lfTiming(0), lfTiming(0.3));
    assert.deepEqual(lfTiming(5), lfTiming(2.7));
    assert.deepEqual(lfTiming(0.3), {'tp': 0.27969515718880367, 'te': 0.3522480809635793, 'ta': 0.004399999999999999});
    assert.deepEqual(lfTiming(2.7), {'tp': 0.5101717311446059, 'te': 0.7869909124636691, 'ta': 0.11960000000000001});
});


test('lf.lfPulse', () => {
    for (const rd of [0.3, 0.6, 1, 1.7, 2.7]) {
        const pulse = lfPulse(rd);
        const {te} = lfTiming(rd);
        const points = 20000;
        let flow = 0;
        let minimum = Infinity;
        for (let ix = 0; ix < points; ix++) {
            const value = pulse(ix / points);
            flow += value / points;
            minimum = Math.min(minimum, value);
        }

        // The flow derivative's net flow is zero, its negative peak (Ee, at te) -1
        assert(Math.abs(flow) < 1e-3, `rd ${rd} net flow ${flow}`);
        assert(Math.abs(minimum + 1) < 1e-2, `rd ${rd} peak ${minimum}`);
        assert(Math.abs(pulse(te) + 1) < 1e-9);
        assert.equal(pulse(0), 0);
    }
});


test('lf.lfHarmonics', () => {
    assert.equal(LF_RD_COUNT, 241);
    assert.equal(LF_RD_COUNT, Math.round((LF_RD_MAX - LF_RD_MIN) / 0.01) + 1);

    // picovox_train/lf.py's harmonic_table (Rd 1.0 - grid index 70)
    const row = lfHarmonics(70);
    assert.equal(row.re.length, LF_HARMONICS);
    assert.equal(row.im.length, LF_HARMONICS);
    const close = (value, expected) => assert(Math.abs(value - expected) < 1e-15, `${value} != ${expected}`);
    close(row.re[1], 0.062216950987419264);
    close(row.im[1], -0.1705324149150758);
    close(row.re[5], -0.014752542877681525);
    close(row.im[5], 0.022730526258628213);

    // The derivative has (about) no DC; rows are computed once
    assert(Math.abs(row.re[0]) < 1e-3);
    assert.equal(lfHarmonics(70), row);
    assert.notEqual(lfHarmonics(0), row);
});

