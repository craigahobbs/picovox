// A small-room ambience - convolution with a synthetic impulse response (early reflections and a diffuse tail)

import {makeRandom} from './klatt.js';
import {fft} from './analysis.js';


// A synthetic room impulse response - RT60 (s), the wet level (dB), and the seed
export function roomImpulse(sampleRate, {rt60 = 0.35, wetDb = -16, seed = 9} = {}) {
    const random = makeRandom(seed);
    const length = Math.floor(rt60 * sampleRate);
    const impulse = new Float32Array(length);
    impulse[0] = 1;
    const wet = Math.pow(10, wetDb / 20);

    // Early reflections (3-25 ms)
    for (let ix = 0; ix < 8; ix++) {
        const delay = Math.floor((0.003 + 0.022 * random.uniform()) * sampleRate);
        impulse[delay] += wet * (random.uniform() < 0.5 ? -1 : 1) * (0.6 + 0.4 * random.uniform()) * 1.5;
    }

    // The diffuse tail - decaying noise, low-passed (air and wall absorption)
    let low = 0;
    const decay = Math.log(1000) / (rt60 * sampleRate);
    const start = Math.floor(0.012 * sampleRate);
    for (let ix = start; ix < length; ix++) {
        low = 0.6 * low + 0.4 * random.gauss();
        impulse[ix] += wet * 0.35 * low * Math.exp(-decay * (ix - start));
    }
    return impulse;
}


// Convolve samples with an impulse response (FFT overlap-add)
export function convolve(samples, impulse) {
    const outLength = samples.length + impulse.length - 1;
    let size = 1;
    while (size < 2 * impulse.length) {
        size <<= 1;
    }
    const block = size - impulse.length + 1;
    const impulseRe = new Float64Array(size);
    const impulseIm = new Float64Array(size);
    impulseRe.set(impulse);
    fft(impulseRe, impulseIm);
    const out = new Float32Array(outLength);
    for (let start = 0; start < samples.length; start += block) {
        const re = new Float64Array(size);
        const im = new Float64Array(size);
        re.set(samples.subarray(start, Math.min(samples.length, start + block)));
        fft(re, im);
        for (let ix = 0; ix < size; ix++) {
            const r = re[ix] * impulseRe[ix] - im[ix] * impulseIm[ix];
            const i = re[ix] * impulseIm[ix] + im[ix] * impulseRe[ix];
            re[ix] = r;
            im[ix] = -i;
        }
        // Inverse FFT by conjugation
        fft(re, im);
        for (let ix = 0; ix < size && start + ix < outLength; ix++) {
            out[start + ix] += re[ix] / size;
        }
    }
    return out;
}
