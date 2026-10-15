// A radix-2 complex FFT with cached twiddles - for the renderer's noise filtering and the LF table


const twiddleCache = new Map();


// The twiddle factors and bit reversal of a size
function plan(size) {
    if (!twiddleCache.has(size)) {
        const cos = new Float64Array(size / 2);
        const sin = new Float64Array(size / 2);
        for (let ix = 0; ix < size / 2; ix++) {
            cos[ix] = Math.cos(2 * Math.PI * ix / size);
            sin[ix] = Math.sin(2 * Math.PI * ix / size);
        }
        const reverse = new Uint32Array(size);
        for (let ix = 1, jx = 0; ix < size; ix++) {
            let bit = size >> 1;
            for (; jx & bit; bit >>= 1) {
                jx ^= bit;
            }
            jx ^= bit;
            reverse[ix] = jx;
        }
        twiddleCache.set(size, {cos, sin, reverse});
    }
    return twiddleCache.get(size);
}


// In-place FFT of real and imaginary arrays (length a power of 2) - inverse (unscaled) if inverse is true
export function fft(re, im, inverse = false) {
    const size = re.length;
    const {cos, sin, reverse} = plan(size);
    for (let ix = 1; ix < size; ix++) {
        const jx = reverse[ix];
        if (ix < jx) {
            [re[ix], re[jx]] = [re[jx], re[ix]];
            [im[ix], im[jx]] = [im[jx], im[ix]];
        }
    }
    const sign = inverse ? 1 : -1;
    for (let len = 2; len <= size; len <<= 1) {
        const half = len >> 1;
        const stride = size / len;
        for (let start = 0; start < size; start += len) {
            for (let kx = 0; kx < half; kx++) {
                const wRe = cos[kx * stride];
                const wIm = sign * sin[kx * stride];
                const aIx = start + kx;
                const bIx = aIx + half;
                const bRe = re[bIx] * wRe - im[bIx] * wIm;
                const bIm = re[bIx] * wIm + im[bIx] * wRe;
                re[bIx] = re[aIx] - bRe;
                im[bIx] = im[aIx] - bIm;
                re[aIx] += bRe;
                im[aIx] += bIm;
            }
        }
    }
}
