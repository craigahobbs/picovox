// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {strict as assert} from 'node:assert';
import {fft} from '../lib/fft.js';
import test from 'node:test';


// A direct DFT
function dft(re, im, inverse = false) {
    const size = re.length;
    const outRe = new Float64Array(size);
    const outIm = new Float64Array(size);
    const sign = inverse ? 1 : -1;
    for (let bin = 0; bin < size; bin++) {
        for (let ix = 0; ix < size; ix++) {
            const angle = sign * 2 * Math.PI * bin * ix / size;
            outRe[bin] += re[ix] * Math.cos(angle) - im[ix] * Math.sin(angle);
            outIm[bin] += re[ix] * Math.sin(angle) + im[ix] * Math.cos(angle);
        }
    }
    return [outRe, outIm];
}


// Arrays equal within a tolerance
function assertClose(actual, expected, tolerance = 1e-9) {
    assert.equal(actual.length, expected.length);
    for (const [ix, value] of actual.entries()) {
        assert(Math.abs(value - expected[ix]) < tolerance, `${ix}: ${value} != ${expected[ix]}`);
    }
}


test('fft', () => {
    for (const size of [1, 2, 8, 64]) {
        const re = Float64Array.from({'length': size}, (_value, ix) => Math.sin(ix * 1.3) + 0.5);
        const im = Float64Array.from({'length': size}, (_value, ix) => Math.cos(ix * 0.7) - 0.2);
        const [expectedRe, expectedIm] = dft(re, im);
        const outRe = Float64Array.from(re);
        const outIm = Float64Array.from(im);
        fft(outRe, outIm);
        assertClose(outRe, expectedRe);
        assertClose(outIm, expectedIm);

        // The inverse (unscaled)
        const [inverseRe, inverseIm] = dft(re, im, true);
        const backRe = Float64Array.from(re);
        const backIm = Float64Array.from(im);
        fft(backRe, backIm, true);
        assertClose(backRe, inverseRe);
        assertClose(backIm, inverseIm);

        // Forward then inverse is the input, times the size
        fft(outRe, outIm, true);
        assertClose(outRe.map((value) => value / size), re);
        assertClose(outIm.map((value) => value / size), im);
    }
});


test('fft, impulse', () => {
    const re = new Float64Array(16);
    const im = new Float64Array(16);
    re[0] = 1;
    fft(re, im);
    assert.deepEqual(Array.from(re), new Array(16).fill(1));
    assert.deepEqual(Array.from(im), new Array(16).fill(0));
});
