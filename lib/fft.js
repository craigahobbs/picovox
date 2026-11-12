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


// A radix-4 pass - four sub-transforms of quarter points (in bit-reversed order: the even-even, even-odd, odd-even, and
// odd-odd samples) to one of 4 quarter. sign is 1 forward, -1 inverse.
function radix4Pass(re, im, size, sign, quarter, twiddles) {
    const length = 4 * quarter;
    for (let kx = 0; kx < quarter; kx++) {
        const base = 6 * kx;
        const w1Re = twiddles[base];
        const w1Im = sign * twiddles[base + 1];
        const w2Re = twiddles[base + 2];
        const w2Im = sign * twiddles[base + 3];
        const w3Re = twiddles[base + 4];
        const w3Im = sign * twiddles[base + 5];
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
            const dRe = sign * (e1Re - e3Re);
            const dIm = sign * (e1Im - e3Im);
            re[start] = aRe + cRe;
            im[start] = aIm + cIm;
            re[i2] = aRe - cRe;
            im[i2] = aIm - cIm;
            // (X[k + quarter] = B - i D forward, B + i D inverse; X[k + 3 quarter] the other - D with the sign)
            re[i1] = bRe + dIm;
            im[i1] = bIm - dRe;
            re[i3] = bRe - dIm;
            im[i3] = bIm + dRe;
        }
    }
}


// Two radix-4 passes at once (of quarter and 4 quarter points - radix4Pass's arithmetic exactly): sixteen points read
// into locals, the first pass's four butterflies (points 0-3, 4-7, 8-11, 12-15 - one twiddle), the second's (points
// 0 4 8 12, 1 5 9 13, 2 6 10 14, 3 7 11 15 - a twiddle each), and the points written back - half the array reads and
// writes of a pass at a time. The butterflies share their temporaries, and the points one index: an interpreter's
// locals are slower past its first hundred or so.
function radix4Pair(re, im, size, sign, quarter, twiddles, nextTwiddles) {
    let e2Re, e2Im, e1Re, e1Im, e3Re, e3Im;
    let aRe, aIm, bRe, bIm, cRe, cIm, dRe, dIm;
    const length = 16 * quarter;
    for (let kx = 0; kx < quarter; kx++) {
        const next0 = 6 * kx;
        const wa1Re = twiddles[next0];
        const wa1Im = sign * twiddles[next0 + 1];
        const wa2Re = twiddles[next0 + 2];
        const wa2Im = sign * twiddles[next0 + 3];
        const wa3Re = twiddles[next0 + 4];
        const wa3Im = sign * twiddles[next0 + 5];
        const wb01Re = nextTwiddles[next0];
        const wb01Im = sign * nextTwiddles[next0 + 1];
        const wb02Re = nextTwiddles[next0 + 2];
        const wb02Im = sign * nextTwiddles[next0 + 3];
        const wb03Re = nextTwiddles[next0 + 4];
        const wb03Im = sign * nextTwiddles[next0 + 5];
        const next1 = 6 * (kx + quarter);
        const wb11Re = nextTwiddles[next1];
        const wb11Im = sign * nextTwiddles[next1 + 1];
        const wb12Re = nextTwiddles[next1 + 2];
        const wb12Im = sign * nextTwiddles[next1 + 3];
        const wb13Re = nextTwiddles[next1 + 4];
        const wb13Im = sign * nextTwiddles[next1 + 5];
        const next2 = 6 * (kx + 2 * quarter);
        const wb21Re = nextTwiddles[next2];
        const wb21Im = sign * nextTwiddles[next2 + 1];
        const wb22Re = nextTwiddles[next2 + 2];
        const wb22Im = sign * nextTwiddles[next2 + 3];
        const wb23Re = nextTwiddles[next2 + 4];
        const wb23Im = sign * nextTwiddles[next2 + 5];
        const next3 = 6 * (kx + 3 * quarter);
        const wb31Re = nextTwiddles[next3];
        const wb31Im = sign * nextTwiddles[next3 + 1];
        const wb32Re = nextTwiddles[next3 + 2];
        const wb32Im = sign * nextTwiddles[next3 + 3];
        const wb33Re = nextTwiddles[next3 + 4];
        const wb33Im = sign * nextTwiddles[next3 + 5];
        for (let start = kx; start < size; start += length) {
            let ix = start;
            let p00Re = re[ix];
            let p00Im = im[ix];
            ix += quarter;
            let p01Re = re[ix];
            let p01Im = im[ix];
            ix += quarter;
            let p02Re = re[ix];
            let p02Im = im[ix];
            ix += quarter;
            let p03Re = re[ix];
            let p03Im = im[ix];
            ix += quarter;
            let p04Re = re[ix];
            let p04Im = im[ix];
            ix += quarter;
            let p05Re = re[ix];
            let p05Im = im[ix];
            ix += quarter;
            let p06Re = re[ix];
            let p06Im = im[ix];
            ix += quarter;
            let p07Re = re[ix];
            let p07Im = im[ix];
            ix += quarter;
            let p08Re = re[ix];
            let p08Im = im[ix];
            ix += quarter;
            let p09Re = re[ix];
            let p09Im = im[ix];
            ix += quarter;
            let p10Re = re[ix];
            let p10Im = im[ix];
            ix += quarter;
            let p11Re = re[ix];
            let p11Im = im[ix];
            ix += quarter;
            let p12Re = re[ix];
            let p12Im = im[ix];
            ix += quarter;
            let p13Re = re[ix];
            let p13Im = im[ix];
            ix += quarter;
            let p14Re = re[ix];
            let p14Im = im[ix];
            ix += quarter;
            let p15Re = re[ix];
            let p15Im = im[ix];
            e2Re = p01Re * wa2Re - p01Im * wa2Im;
            e2Im = p01Re * wa2Im + p01Im * wa2Re;
            e1Re = p02Re * wa1Re - p02Im * wa1Im;
            e1Im = p02Re * wa1Im + p02Im * wa1Re;
            e3Re = p03Re * wa3Re - p03Im * wa3Im;
            e3Im = p03Re * wa3Im + p03Im * wa3Re;
            aRe = p00Re + e2Re;
            aIm = p00Im + e2Im;
            bRe = p00Re - e2Re;
            bIm = p00Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p00Re = aRe + cRe;
            p00Im = aIm + cIm;
            p02Re = aRe - cRe;
            p02Im = aIm - cIm;
            p01Re = bRe + dIm;
            p01Im = bIm - dRe;
            p03Re = bRe - dIm;
            p03Im = bIm + dRe;
            e2Re = p05Re * wa2Re - p05Im * wa2Im;
            e2Im = p05Re * wa2Im + p05Im * wa2Re;
            e1Re = p06Re * wa1Re - p06Im * wa1Im;
            e1Im = p06Re * wa1Im + p06Im * wa1Re;
            e3Re = p07Re * wa3Re - p07Im * wa3Im;
            e3Im = p07Re * wa3Im + p07Im * wa3Re;
            aRe = p04Re + e2Re;
            aIm = p04Im + e2Im;
            bRe = p04Re - e2Re;
            bIm = p04Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p04Re = aRe + cRe;
            p04Im = aIm + cIm;
            p06Re = aRe - cRe;
            p06Im = aIm - cIm;
            p05Re = bRe + dIm;
            p05Im = bIm - dRe;
            p07Re = bRe - dIm;
            p07Im = bIm + dRe;
            e2Re = p09Re * wa2Re - p09Im * wa2Im;
            e2Im = p09Re * wa2Im + p09Im * wa2Re;
            e1Re = p10Re * wa1Re - p10Im * wa1Im;
            e1Im = p10Re * wa1Im + p10Im * wa1Re;
            e3Re = p11Re * wa3Re - p11Im * wa3Im;
            e3Im = p11Re * wa3Im + p11Im * wa3Re;
            aRe = p08Re + e2Re;
            aIm = p08Im + e2Im;
            bRe = p08Re - e2Re;
            bIm = p08Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p08Re = aRe + cRe;
            p08Im = aIm + cIm;
            p10Re = aRe - cRe;
            p10Im = aIm - cIm;
            p09Re = bRe + dIm;
            p09Im = bIm - dRe;
            p11Re = bRe - dIm;
            p11Im = bIm + dRe;
            e2Re = p13Re * wa2Re - p13Im * wa2Im;
            e2Im = p13Re * wa2Im + p13Im * wa2Re;
            e1Re = p14Re * wa1Re - p14Im * wa1Im;
            e1Im = p14Re * wa1Im + p14Im * wa1Re;
            e3Re = p15Re * wa3Re - p15Im * wa3Im;
            e3Im = p15Re * wa3Im + p15Im * wa3Re;
            aRe = p12Re + e2Re;
            aIm = p12Im + e2Im;
            bRe = p12Re - e2Re;
            bIm = p12Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p12Re = aRe + cRe;
            p12Im = aIm + cIm;
            p14Re = aRe - cRe;
            p14Im = aIm - cIm;
            p13Re = bRe + dIm;
            p13Im = bIm - dRe;
            p15Re = bRe - dIm;
            p15Im = bIm + dRe;
            e2Re = p04Re * wb02Re - p04Im * wb02Im;
            e2Im = p04Re * wb02Im + p04Im * wb02Re;
            e1Re = p08Re * wb01Re - p08Im * wb01Im;
            e1Im = p08Re * wb01Im + p08Im * wb01Re;
            e3Re = p12Re * wb03Re - p12Im * wb03Im;
            e3Im = p12Re * wb03Im + p12Im * wb03Re;
            aRe = p00Re + e2Re;
            aIm = p00Im + e2Im;
            bRe = p00Re - e2Re;
            bIm = p00Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p00Re = aRe + cRe;
            p00Im = aIm + cIm;
            p08Re = aRe - cRe;
            p08Im = aIm - cIm;
            p04Re = bRe + dIm;
            p04Im = bIm - dRe;
            p12Re = bRe - dIm;
            p12Im = bIm + dRe;
            e2Re = p05Re * wb12Re - p05Im * wb12Im;
            e2Im = p05Re * wb12Im + p05Im * wb12Re;
            e1Re = p09Re * wb11Re - p09Im * wb11Im;
            e1Im = p09Re * wb11Im + p09Im * wb11Re;
            e3Re = p13Re * wb13Re - p13Im * wb13Im;
            e3Im = p13Re * wb13Im + p13Im * wb13Re;
            aRe = p01Re + e2Re;
            aIm = p01Im + e2Im;
            bRe = p01Re - e2Re;
            bIm = p01Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p01Re = aRe + cRe;
            p01Im = aIm + cIm;
            p09Re = aRe - cRe;
            p09Im = aIm - cIm;
            p05Re = bRe + dIm;
            p05Im = bIm - dRe;
            p13Re = bRe - dIm;
            p13Im = bIm + dRe;
            e2Re = p06Re * wb22Re - p06Im * wb22Im;
            e2Im = p06Re * wb22Im + p06Im * wb22Re;
            e1Re = p10Re * wb21Re - p10Im * wb21Im;
            e1Im = p10Re * wb21Im + p10Im * wb21Re;
            e3Re = p14Re * wb23Re - p14Im * wb23Im;
            e3Im = p14Re * wb23Im + p14Im * wb23Re;
            aRe = p02Re + e2Re;
            aIm = p02Im + e2Im;
            bRe = p02Re - e2Re;
            bIm = p02Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p02Re = aRe + cRe;
            p02Im = aIm + cIm;
            p10Re = aRe - cRe;
            p10Im = aIm - cIm;
            p06Re = bRe + dIm;
            p06Im = bIm - dRe;
            p14Re = bRe - dIm;
            p14Im = bIm + dRe;
            e2Re = p07Re * wb32Re - p07Im * wb32Im;
            e2Im = p07Re * wb32Im + p07Im * wb32Re;
            e1Re = p11Re * wb31Re - p11Im * wb31Im;
            e1Im = p11Re * wb31Im + p11Im * wb31Re;
            e3Re = p15Re * wb33Re - p15Im * wb33Im;
            e3Im = p15Re * wb33Im + p15Im * wb33Re;
            aRe = p03Re + e2Re;
            aIm = p03Im + e2Im;
            bRe = p03Re - e2Re;
            bIm = p03Im - e2Im;
            cRe = e1Re + e3Re;
            cIm = e1Im + e3Im;
            dRe = sign * (e1Re - e3Re);
            dIm = sign * (e1Im - e3Im);
            p03Re = aRe + cRe;
            p03Im = aIm + cIm;
            p11Re = aRe - cRe;
            p11Im = aIm - cIm;
            p07Re = bRe + dIm;
            p07Im = bIm - dRe;
            p15Re = bRe - dIm;
            p15Im = bIm + dRe;
            ix = start;
            re[ix] = p00Re;
            im[ix] = p00Im;
            ix += quarter;
            re[ix] = p01Re;
            im[ix] = p01Im;
            ix += quarter;
            re[ix] = p02Re;
            im[ix] = p02Im;
            ix += quarter;
            re[ix] = p03Re;
            im[ix] = p03Im;
            ix += quarter;
            re[ix] = p04Re;
            im[ix] = p04Im;
            ix += quarter;
            re[ix] = p05Re;
            im[ix] = p05Im;
            ix += quarter;
            re[ix] = p06Re;
            im[ix] = p06Im;
            ix += quarter;
            re[ix] = p07Re;
            im[ix] = p07Im;
            ix += quarter;
            re[ix] = p08Re;
            im[ix] = p08Im;
            ix += quarter;
            re[ix] = p09Re;
            im[ix] = p09Im;
            ix += quarter;
            re[ix] = p10Re;
            im[ix] = p10Im;
            ix += quarter;
            re[ix] = p11Re;
            im[ix] = p11Im;
            ix += quarter;
            re[ix] = p12Re;
            im[ix] = p12Im;
            ix += quarter;
            re[ix] = p13Re;
            im[ix] = p13Im;
            ix += quarter;
            re[ix] = p14Re;
            im[ix] = p14Im;
            ix += quarter;
            re[ix] = p15Re;
            im[ix] = p15Im;
        }
    }
}


// In-place FFT of real and imaginary arrays (length a power of 2) - inverse (unscaled) if inverse is true
export function fft(re, im, inverse = false) {
    const size = re.length;
    const {swaps, passes, radix2} = plan(size);
    const swapCount = swaps.length;
    for (let ix = 0; ix < swapCount; ix += 2) {
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
            const aRe = re[ix];
            const aIm = im[ix];
            const bRe = re[ix + 1];
            const bIm = im[ix + 1];
            re[ix + 1] = aRe - bRe;
            im[ix + 1] = aIm - bIm;
            re[ix] = aRe + bRe;
            im[ix] = aIm + bIm;
        }
    }
    // The radix-4 passes, two at a time - and the last alone, if their number is odd
    const sign = inverse ? -1 : 1;
    const passCount = passes.length;
    let passIx = 0;
    for (; passIx + 1 < passCount; passIx += 2) {
        radix4Pair(re, im, size, sign, passes[passIx].quarter, passes[passIx].twiddles, passes[passIx + 1].twiddles);
    }
    if (passIx < passCount) {
        radix4Pass(re, im, size, sign, passes[passIx].quarter, passes[passIx].twiddles);
    }
}
