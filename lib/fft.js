// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// A radix-4 complex FFT (a radix-2 pass first for odd powers of 2) with cached plans - for the renderer's noise filtering,
// its voicing tables, and the LF table. Few operations per point (three complex multiplies per four points, half the
// passes of radix 2), for JavaScript engines without a JIT.


// The plans by size - the bit-reversal swaps, and each radix-4 pass's twiddles (W^k, W^2k, W^3k for k < quarter, forward)
const plans = new Map();


// The plan of a size
function plan(size) {
    if (!plans.has(size)) {
        const swaps = [];
        for (let ix = 1, jx = 0; ix < size; ix++) {
            let bit = size >> 1;
            for (; jx & bit; bit >>= 1) {
                jx ^= bit;
            }
            jx ^= bit;
            if (ix < jx) {
                swaps.push(ix, jx);
            }
        }
        const passes = [];
        let quarter = Math.log2(size) % 2 === 1 ? 2 : 1;
        for (; 4 * quarter <= size; quarter *= 4) {
            const length = 4 * quarter;
            const twiddles = new Float64Array(6 * quarter);
            for (let kx = 0; kx < quarter; kx++) {
                for (let power = 1; power <= 3; power++) {
                    const angle = -2 * Math.PI * power * kx / length;
                    twiddles[6 * kx + 2 * (power - 1)] = Math.cos(angle);
                    twiddles[6 * kx + 2 * (power - 1) + 1] = Math.sin(angle);
                }
            }
            passes.push({quarter, twiddles});
        }
        plans.set(size, {'swaps': Uint32Array.from(swaps), passes, 'radix2': Math.log2(size) % 2 === 1});
    }
    return plans.get(size);
}


// In-place FFT of real and imaginary arrays (length a power of 2) - inverse (unscaled) if inverse is true
export function fft(re, im, inverse = false) {
    const size = re.length;
    const {swaps, passes, radix2} = plan(size);
    for (let ix = 0; ix < swaps.length; ix += 2) {
        const aIx = swaps[ix];
        const bIx = swaps[ix + 1];
        const swapRe = re[aIx];
        re[aIx] = re[bIx];
        re[bIx] = swapRe;
        const swapIm = im[aIx];
        im[aIx] = im[bIx];
        im[bIx] = swapIm;
    }
    // (an odd power of 2 - one radix-2 pass of pairs first)
    if (radix2) {
        for (let ix = 0; ix < size; ix += 2) {
            const bRe = re[ix + 1];
            const bIm = im[ix + 1];
            re[ix + 1] = re[ix] - bRe;
            im[ix + 1] = im[ix] - bIm;
            re[ix] += bRe;
            im[ix] += bIm;
        }
    }
    // The radix-4 passes - four sub-transforms of quarter points (in bit-reversed order: the even-even, even-odd,
    // odd-even, and odd-odd samples) to one of 4 quarter
    const sign = inverse ? -1 : 1;
    for (const {quarter, twiddles} of passes) {
        const length = 4 * quarter;
        for (let kx = 0; kx < quarter; kx++) {
            const w1Re = twiddles[6 * kx];
            const w1Im = sign * twiddles[6 * kx + 1];
            const w2Re = twiddles[6 * kx + 2];
            const w2Im = sign * twiddles[6 * kx + 3];
            const w3Re = twiddles[6 * kx + 4];
            const w3Im = sign * twiddles[6 * kx + 5];
            for (let start = kx; start < size; start += length) {
                const i1 = start + quarter;
                const i2 = i1 + quarter;
                const i3 = i2 + quarter;
                // (E0 at start, E2 at i1, E1 at i2, E3 at i3)
                const e0Re = re[start];
                const e0Im = im[start];
                const x2Re = re[i1];
                const x2Im = im[i1];
                const x1Re = re[i2];
                const x1Im = im[i2];
                const x3Re = re[i3];
                const x3Im = im[i3];
                const e2Re = x2Re * w2Re - x2Im * w2Im;
                const e2Im = x2Re * w2Im + x2Im * w2Re;
                const e1Re = x1Re * w1Re - x1Im * w1Im;
                const e1Im = x1Re * w1Im + x1Im * w1Re;
                const e3Re = x3Re * w3Re - x3Im * w3Im;
                const e3Im = x3Re * w3Im + x3Im * w3Re;
                const aRe = e0Re + e2Re;
                const aIm = e0Im + e2Im;
                const bRe = e0Re - e2Re;
                const bIm = e0Im - e2Im;
                const cRe = e1Re + e3Re;
                const cIm = e1Im + e3Im;
                const dRe = e1Re - e3Re;
                const dIm = e1Im - e3Im;
                re[start] = aRe + cRe;
                im[start] = aIm + cIm;
                re[i2] = aRe - cRe;
                im[i2] = aIm - cIm;
                // (X[k + quarter] = B - i D forward, B + i D inverse; X[k + 3 quarter] the other)
                re[i1] = bRe + sign * dIm;
                im[i1] = bIm - sign * dRe;
                re[i3] = bRe - sign * dIm;
                im[i3] = bIm + sign * dRe;
            }
        }
    }
}
