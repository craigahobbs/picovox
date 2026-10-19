// The parameter-frame renderer - picovox_train/dsynth.py's source-filter synthesizer in JavaScript, for the browser and Node.
// It renders frames of the parameter set (picovox_train/params.py, lib/params.js) at any sample rate, streaming: frames are
// pushed as they're predicted, and blocks of samples are delivered as soon as they're final.
//
//     import {render} from './lib/render.js';
//     const samples = render(frames, {'sampleRate': 44100, 'seed': 1});   // frames: [{f0, av, ...}, ...] or arrays
//
//     const renderer = createRenderer({'sampleRate': 44100, 'onBlock': (block) => play(block)});
//     renderer.push(frame); ...; renderer.end();
//
// The algorithm is picovox_train/dsynth.py's (with its counter-based noise, lib/rng.js) - the noise sample for sample,
// the voicing to within about 1e-5 (SNR near 100 dB), so it renders what the voice was trained to hear:
//
// - Voicing - harmonic synthesis: each frame's complex harmonic amplitudes are the voicing amplitude times the LF
//   pulse's Fourier coefficients (lib/lf.js, interpolated in Rd), the source shelf, and the vocal tract's response
//   at k * F0 (the Klatt cascade's filters at the 44.1 kHz reference rate, as analytic frequency responses), tapered near
//   the Nyquist frequency. Their sum over the glottal phase is a frame's one period of voicing - computed once per frame
//   as a table (an inverse FFT, two frames' at once, about 8 points a harmonic) and read at the phase with cubic
//   interpolation, rather than summed for every sample; the frames' values are interpolated linearly, and the phase
//   carries the jitter (a Gaussian per cycle) and the shimmer (Gaussians interpolated across each cycle).
// - Noise - white noise scaled per sample by its level (ah, af interpolated; the aspiration also modulated by the
//   glottal cycle), filtered in the frequency domain: Hann frames of noiseFftSize samples, hop a quarter of that,
//   each spectrum multiplied by the filters' magnitudes (the spectral shapes - a real biquad's squared magnitude is a
//   cosine polynomial, so the tract and the high-pass need no complex arithmetic) at the frame's interpolated
//   parameters, overlap-added and normalized by the windows' squared sum.
//
// It's written for JavaScript engines without a JIT too (an iPhone's Safari can run without one, interpreting every
// operation): few operations per sample - each frame's values loaded once for its samples, the buffers read and
// written directly, and the noise draws inline.

import {
    DB_NAMES, EQ_BAND_COUNT, EQ_HIGH_HZ, EQ_LOW_HZ, F0_MIN, FLOOR_DB, FRAME_RATE, HARMONIC_TAPER_END,
    HARMONIC_TAPER_START, HIGHER_POLES, HIGHER_POLE_BANDWIDTH, HIGHER_POLE_BANDWIDTH_STEP, HIGHER_POLE_SPACING,
    LF_HARMONICS, LF_RD_MAX, LF_RD_MIN, LF_RD_STEP, NAMES, NOISE_MODULATION, PARAMETERS, REF_RATE, SHELF_HZ
} from './params.js';
import {LF_RD_COUNT, lfHarmonics} from './lf.js';
import {STREAM_ASPIRATION, STREAM_FRICATION, STREAM_JITTER, STREAM_SHIMMER, gaussAt, streamSeed} from './rng.js';
import {fft} from './fft.js';


// The parameter indexes
const IX = Object.fromEntries(NAMES.map((name, ix) => [name, ix]));
const IX_F0 = IX.f0;
const IX_JITTER = IX.jitter;
const IX_RD = IX.rd;
const IX_SHIMMER = IX.shimmer;
const IX_AV = IX.av;
const IX_AH = IX.ah;
const IX_AF = IX.af;
const DB_SET = new Set(DB_NAMES);

// The Gaussian draws' scale (rng.js gaussAt)
const SQRT_HALF = Math.sqrt(0.5);

// Voicing below this linear amplitude (FLOOR_DB + 10 dB) is skipped
const VOICING_MIN = 10 ** ((FLOOR_DB + 10) / 20);


// The noise STFT size for a sample rate - the power of two at or above 16 ms
export function noiseFftSize(sampleRate) {
    return 2 ** Math.ceil(Math.log2(sampleRate * 0.016));
}


// The number of samples a sequence of frames renders
export function sampleCount(frameCount, sampleRate) {
    return Math.max(0, Math.floor((frameCount - 1) * sampleRate / FRAME_RATE));
}


// A frame (an object of parameters - missing ones take their defaults - or an array in parameter order) as a
// Float64Array of parameter values
export function frameValues(frame) {
    const values = new Float64Array(NAMES.length);
    for (let ix = 0; ix < NAMES.length; ix++) {
        const value = Array.isArray(frame) || ArrayBuffer.isView(frame) ? frame[ix] : frame[NAMES[ix]];
        values[ix] = value === undefined || value === null ? PARAMETERS[ix].default : value;
    }
    return values;
}


// Parameter values with the levels as linear amplitudes (the interpolable form)
function linearValues(values) {
    const linear = new Float64Array(values.length);
    for (let ix = 0; ix < values.length; ix++) {
        linear[ix] = DB_SET.has(NAMES[ix]) ? 10 ** (values[ix] / 20) : values[ix];
    }
    return linear;
}


// A resonator's coefficients (the former lib/klatt.js, at the reference rate) - [A, B, C]
function resonator(center, bandwidth) {
    const radius = Math.exp(-Math.PI * bandwidth / REF_RATE);
    const coefC = -radius * radius;
    const coefB = 2 * radius * Math.cos(2 * Math.PI * center / REF_RATE);
    return [1 - coefB - coefC, coefB, coefC];
}


// A biquad's complex response at (cos w, sin w, cos 2w, sin 2w) - returns the real part; the imaginary part is left in
// biquadIm (no allocation - these run per harmonic and per noise bin)
let biquadIm = 0;
function biquad(coefs, cos1, sin1, cos2, sin2) {
    // (indexed reads - array destructuring is about 10% slower in this per-harmonic, per-bin path)
    /* eslint-disable prefer-destructuring */
    const b0 = coefs[0];
    const b1 = coefs[1];
    const b2 = coefs[2];
    const a1 = coefs[3];
    const a2 = coefs[4];
    /* eslint-enable prefer-destructuring */
    const numRe = b0 + b1 * cos1 + b2 * cos2;
    const numIm = -(b1 * sin1 + b2 * sin2);
    const denRe = 1 + a1 * cos1 + a2 * cos2;
    const denIm = -(a1 * sin1 + a2 * sin2);
    const den = denRe * denRe + denIm * denIm;
    biquadIm = (numIm * denRe - numRe * denIm) / den;
    return (numRe * denRe + numIm * denIm) / den;
}


// The source shelf's biquad coefficients (the former lib/klatt.js HighShelf at SHELF_HZ)
function shelfCoefficients(gainDb) {
    const amp = 10 ** (gainDb / 40);
    const w0 = 2 * Math.PI * SHELF_HZ / REF_RATE;
    const alpha = Math.sin(w0) / 2 * Math.sqrt(2);
    const cos = Math.cos(w0);
    const sqrtA = 2 * Math.sqrt(amp) * alpha;
    const a0 = (amp + 1) - (amp - 1) * cos + sqrtA;
    return [
        amp * ((amp + 1) + (amp - 1) * cos + sqrtA) / a0, -2 * amp * ((amp - 1) + (amp + 1) * cos) / a0,
        amp * ((amp + 1) + (amp - 1) * cos - sqrtA) / a0, 2 * ((amp - 1) - (amp + 1) * cos) / a0,
        ((amp + 1) - (amp - 1) * cos - sqrtA) / a0
    ];
}


// The frication bandpass's coefficients (the former lib/klatt.js Bandpass)
function bandpassCoefficients(center, bandwidth) {
    const w0 = 2 * Math.PI * Math.min(center, 0.45 * REF_RATE) / REF_RATE;
    const alpha = Math.sin(w0) / (2 * Math.max(0.3, center / bandwidth));
    const a0 = 1 + alpha;
    return [alpha / a0, 0, -alpha / a0, -2 * Math.cos(w0) / a0, (1 - alpha) / a0];
}


// The frication high-pass's coefficients (the former lib/klatt.js HighPass)
function highpassCoefficients(cutoff) {
    const w0 = 2 * Math.PI * Math.min(cutoff, 0.45 * REF_RATE) / REF_RATE;
    const alpha = Math.sin(w0) / Math.sqrt(2);
    const cos = Math.cos(w0);
    const a0 = 1 + alpha;
    return [(1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0];
}


// The formants' (frequency, bandwidth) parameter indexes, and the tract's pole count
const FORMANT_IX = [1, 2, 3, 4, 5, 6].map((kx) => [IX[`f${kx}`], IX[`b${kx}`]]);
const POLE_COUNT = FORMANT_IX.length + 1 + HIGHER_POLES;


// A frame's filters (from linear parameter values) - the tract's poles (their B and C coefficients) and zero, the
// shelf, and the frication filter
function frameFilters(linear) {
    const poleB = new Float64Array(POLE_COUNT);
    const poleC = new Float64Array(POLE_COUNT);
    let pole = 0;
    let gain = 1;
    const addPole = ([coefA, coefB, coefC]) => {
        gain *= coefA;
        poleB[pole] = coefB;
        poleC[pole] = coefC;
        pole++;
    };
    const zero = resonator(linear[IX.fnz], linear[IX.bnz]);
    gain /= zero[0];
    for (const [frequencyIx, bandwidthIx] of FORMANT_IX) {
        addPole(resonator(linear[frequencyIx], linear[bandwidthIx]));
    }
    addPole(resonator(linear[IX.fnp], linear[IX.bnp]));
    for (let kx = 1; kx <= HIGHER_POLES; kx++) {
        const bandwidth = HIGHER_POLE_BANDWIDTH + HIGHER_POLE_BANDWIDTH_STEP * (kx - 1);
        addPole(resonator(linear[IX.f6] * (1 + kx * HIGHER_POLE_SPACING), bandwidth));
    }
    return {
        poleB, poleC, zero, gain, 'shelf': shelfCoefficients(linear[IX.shelf]),
        'bp1': bandpassCoefficients(linear[IX.fp1], linear[IX.bp1]), 'bp2': bandpassCoefficients(linear[IX.fp2], linear[IX.bp2]),
        'hp': highpassCoefficients(linear[IX.fhp])
    };
}


// The tract's complex response at (cos w, sin w) of the reference rate - returns the real part; the imaginary part is
// left in tractIm
let tractIm = 0;
function tractResponse(filters, cos1, sin1) {
    const cos2 = 2 * cos1 * cos1 - 1;
    const sin2 = 2 * sin1 * cos1;
    // The product of the poles' denominators 1 - B e^-jw - C e^-2jw
    let prodRe = 1;
    let prodIm = 0;
    const {poleB, poleC} = filters;
    for (let pole = 0; pole < poleB.length; pole++) {
        const coefB = poleB[pole];
        const coefC = poleC[pole];
        const denRe = 1 - coefB * cos1 - coefC * cos2;
        const denIm = coefB * sin1 + coefC * sin2;
        const nextRe = prodRe * denRe - prodIm * denIm;
        prodIm = prodRe * denIm + prodIm * denRe;
        prodRe = nextRe;
    }
    // (indexed reads, as in biquad)
    /* eslint-disable prefer-destructuring */
    const zeroB = filters.zero[1];
    const zeroC = filters.zero[2];
    /* eslint-enable prefer-destructuring */
    const zeroRe = 1 - zeroB * cos1 - zeroC * cos2;
    const zeroIm = zeroB * sin1 + zeroC * sin2;
    // gain * zero / product
    const prodAbs = prodRe * prodRe + prodIm * prodIm;
    const re = (zeroRe * prodRe + zeroIm * prodIm) / prodAbs;
    tractIm = filters.gain * ((zeroIm * prodRe - zeroRe * prodIm) / prodAbs);
    return filters.gain * re;
}


// The harmonic taper at a frequency for a sample rate
function harmonicTaper(frequency, sampleRate) {
    const start = HARMONIC_TAPER_START * sampleRate;
    const end = HARMONIC_TAPER_END * sampleRate;
    const position = Math.max(0, Math.min(1, (frequency - start) / (end - start)));
    return 0.5 + 0.5 * Math.cos(Math.PI * position);
}


// The channel equalizer's linear gain at a frequency, for gains eq (dB, EQ_BAND_COUNT of them) - 1 if eq is null
function eqGain(eq, frequency) {
    if (eq === null) {
        return 1;
    }
    const step = Math.log(EQ_HIGH_HZ / EQ_LOW_HZ) / (EQ_BAND_COUNT - 1);
    const position = Math.max(0, Math.min(EQ_BAND_COUNT - 1, Math.log(Math.max(1, frequency) / EQ_LOW_HZ) / step));
    const low = Math.min(EQ_BAND_COUNT - 2, Math.floor(position));
    const weight = position - low;
    return 10 ** ((eq[low] * (1 - weight) + eq[low + 1] * weight) / 20);
}


// A frame's complex harmonic amplitudes - {re, im} Float64Arrays indexed by harmonic number (1 ... count)
function harmonicAmplitudes(linear, filters, sampleRate, eq) {
    const av = linear[IX.av];
    const f0 = Math.max(F0_MIN, linear[IX.f0]);
    const count = av <= VOICING_MIN ? 0 : Math.min(LF_HARMONICS - 1, Math.floor(HARMONIC_TAPER_END * sampleRate / f0));
    const re = new Float64Array(count + 1);
    const im = new Float64Array(count + 1);
    if (count === 0) {
        return {re, im, count};
    }

    // The LF coefficients, interpolated in Rd
    const grid = (Math.max(LF_RD_MIN, Math.min(LF_RD_MAX, linear[IX.rd])) - LF_RD_MIN) / LF_RD_STEP;
    const low = Math.max(0, Math.min(LF_RD_COUNT - 2, Math.floor(grid)));
    const weight = grid - low;
    const rowLow = lfHarmonics(low);
    const rowHigh = lfHarmonics(low + 1);
    for (let kx = 1; kx <= count; kx++) {
        const frequency = kx * f0;
        const omega = 2 * Math.PI * frequency / REF_RATE;
        const cos1 = Math.cos(omega);
        const sin1 = Math.sin(omega);
        const tractRe = tractResponse(filters, cos1, sin1);
        const tractImag = tractIm;
        const shelfRe = biquad(filters.shelf, cos1, sin1, 2 * cos1 * cos1 - 1, 2 * sin1 * cos1);
        const shelfIm = biquadIm;
        const scale = 2 * av * harmonicTaper(frequency, sampleRate) * eqGain(eq, frequency);
        const filterRe = scale * (tractRe * shelfRe - tractImag * shelfIm);
        const filterIm = scale * (tractRe * shelfIm + tractImag * shelfRe);
        const sourceRe = rowLow.re[kx] * (1 - weight) + rowHigh.re[kx] * weight;
        const sourceIm = rowLow.im[kx] * (1 - weight) + rowHigh.im[kx] * weight;
        re[kx] = sourceRe * filterRe - sourceIm * filterIm;
        im[kx] = sourceRe * filterIm + sourceIm * filterRe;
    }
    return {re, im, count};
}


// A growable sample buffer addressed by absolute sample index, trimmable from the front
class SampleBuffer {
    constructor() {
        this.data = new Float64Array(8192);
        this.base = 0;
    }

    ensure(index) {
        if (index - this.base >= this.data.length) {
            const data = new Float64Array(Math.max(2 * this.data.length, index - this.base + 1));
            data.set(this.data);
            this.data = data;
        }
    }

    // Drop the samples before an index
    trim(index) {
        const drop = index - this.base;
        if (drop > 65536) {
            this.data.copyWithin(0, drop);
            this.data.fill(0, this.data.length - drop);
            this.base = index;
        }
    }
}


// Create a streaming renderer - push(frame) frames (objects or arrays of parameter values) as they come, then end().
// Blocks of up to blockSize samples (Float32Array) go to onBlock as soon as they're final. end() returns the total
// sample count. eq is the optional channel equalizer's gains (dB, EQ_BAND_COUNT of them - picovox_train/params.py).
export function createRenderer({sampleRate = 44100, seed = 1, onBlock = null, blockSize = 2048, eq = null} = {}) {
    const fftSize = noiseFftSize(sampleRate);
    const hop = fftSize / 4;
    const half = fftSize / 2;
    const noiseScale = Math.sqrt(sampleRate / REF_RATE);
    const seeds = {
        'aspiration': streamSeed(seed, STREAM_ASPIRATION), 'frication': streamSeed(seed, STREAM_FRICATION),
        'jitter': streamSeed(seed, STREAM_JITTER), 'shimmer': streamSeed(seed, STREAM_SHIMMER)
    };
    const window = new Float64Array(fftSize);
    for (let ix = 0; ix < fftSize; ix++) {
        window[ix] = 0.5 - 0.5 * Math.cos(2 * Math.PI * ix / fftSize);
    }
    // The noise frequency bins' (cos w, sin w) at the reference rate, and the aspiration lowpass's magnitude
    const binCount = half + 1;
    const binCos = new Float64Array(binCount);
    const binSin = new Float64Array(binCount);
    const aspirationShape = new Float64Array(binCount);
    const binEq = new Float64Array(binCount);
    for (let bin = 0; bin < binCount; bin++) {
        const omega = 2 * Math.PI * (bin * sampleRate / fftSize) / REF_RATE;
        binCos[bin] = Math.cos(omega);
        binSin[bin] = Math.sin(omega);
        aspirationShape[bin] = 0.5 / Math.sqrt(1.25 - binCos[bin]);
        binEq[bin] = eqGain(eq, bin * sampleRate / fftSize);
    }

    // The frames (values and linear values), and per-frame harmonic amplitudes, by absolute frame index
    const frames = [];
    const linears = [];
    let frameBase = 0;
    const amplitudes = new Map();
    const frameAmplitudes = (index) => {
        if (!amplitudes.has(index)) {
            const linear = linears[index - frameBase];
            amplitudes.set(index, harmonicAmplitudes(linear, frameFilters(linear), sampleRate, eq));
        }
        return amplitudes.get(index);
    };

    // The frames' voicing tables - one period of each frame's voicing (the harmonics' sum over the glottal phase) at
    // size points (a power of 2, about 4 times the harmonics; 3 points of wrap - one before, two after - for the cubic
    // interpolation), two frames' at once from one inverse FFT (frame 2n's the real part, 2n + 1's the imaginary)
    const tables = new Map();
    const frameTable = (index) => {
        if (!tables.has(index)) {
            const first = index - (index & 1);
            const ampFirst = frameAmplitudes(first);
            const ampSecond = first + 1 < frameBase + frames.length ? frameAmplitudes(first + 1) : {'count': 0};
            const count = Math.max(ampFirst.count, ampSecond.count);
            if (count === 0) {
                tables.set(first, null);
                tables.set(first + 1, null);
            } else {
                let size = 32;
                while (size < 8 * count) {
                    size *= 2;
                }
                // Hermitian spectra (each table real): H[k] = X[k] / 2, H[size - k] = conj(X[k]) / 2; packed as
                // H1 + i H2, so the inverse transform is t1 + i t2
                const specRe = new Float64Array(size);
                const specIm = new Float64Array(size);
                for (let kx = 1; kx <= ampFirst.count; kx++) {
                    specRe[kx] += 0.5 * ampFirst.re[kx];
                    specIm[kx] += 0.5 * ampFirst.im[kx];
                    specRe[size - kx] += 0.5 * ampFirst.re[kx];
                    specIm[size - kx] -= 0.5 * ampFirst.im[kx];
                }
                for (let kx = 1; kx <= ampSecond.count; kx++) {
                    // i H2: i (a + ib) / 2 = (-b + ia) / 2 at k; i (a - ib) / 2 = (b + ia) / 2 at size - k
                    specRe[kx] -= 0.5 * ampSecond.im[kx];
                    specIm[kx] += 0.5 * ampSecond.re[kx];
                    specRe[size - kx] += 0.5 * ampSecond.im[kx];
                    specIm[size - kx] += 0.5 * ampSecond.re[kx];
                }
                fft(specRe, specIm, true);
                const tableFirst = new Float64Array(size + 3);
                const tableSecond = new Float64Array(size + 3);
                for (let ix = 0; ix < size; ix++) {
                    tableFirst[ix + 1] = specRe[ix];
                    tableSecond[ix + 1] = specIm[ix];
                }
                for (const table of [tableFirst, tableSecond]) {
                    [table[0], table[size + 1], table[size + 2]] = [table[size], table[1], table[2]];
                }
                tables.set(first, ampFirst.count > 0 ? {size, 'table': tableFirst} : null);
                tables.set(first + 1, ampSecond.count > 0 ? {size, 'table': tableSecond} : null);
            }
        }
        return tables.get(index);
    };

    // The generation state
    const voicing = new SampleBuffer();
    const aspiration = new SampleBuffer();
    const frication = new SampleBuffer();
    const noiseOut = new SampleBuffer();
    let generated = 0;
    let phase0 = 0;
    let phase = 0;
    let jitterCycle = -1;
    let jitterDraw = 0;
    let shimmerCycle = -1;
    let shimmerDraws = [0, 0];
    let noiseFrames = 0;
    let emitted = 0;
    let total = null;
    let block = new Float32Array(blockSize);
    let blockLength = 0;

    // Generate the voicing and noise inputs of samples [start, end) - frame by frame, each frame's values loaded once,
    // the buffers written directly, the noise draws inline (rng.js gaussAt's arithmetic exactly)
    const generateSamples = (start, end) => {
        voicing.ensure(end - 1);
        aspiration.ensure(end - 1);
        frication.ensure(end - 1);
        const voicingData = voicing.data;
        const aspirationData = aspiration.data;
        const fricationData = frication.data;
        const voicingBase = voicing.base;
        const aspirationBase = aspiration.base;
        const fricationBase = frication.base;
        const aspirationSeed = seeds.aspiration;
        const fricationSeed = seeds.frication;
        let index = start;
        while (index < end) {
            const frameIndex = Math.floor(index * FRAME_RATE / sampleRate);
            // The frame's last sample + 1 - the first index whose position (as the samples compute it) reaches the next
            // frame, searched from just before the boundary
            let frameEnd = Math.max(index + 1, Math.ceil((frameIndex + 1) * sampleRate / FRAME_RATE) - 1);
            while (Math.floor(frameEnd * FRAME_RATE / sampleRate) <= frameIndex) {
                frameEnd++;
            }
            const stop = Math.min(end, frameEnd);
            const valuesA = frames[frameIndex - frameBase];
            const valuesB = frames[frameIndex + 1 - frameBase];
            const linearA = linears[frameIndex - frameBase];
            const linearB = linears[frameIndex + 1 - frameBase];
            const f0A = valuesA[IX_F0];
            const f0B = valuesB[IX_F0];
            const jitterA = valuesA[IX_JITTER];
            const jitterB = valuesB[IX_JITTER];
            const rdA = valuesA[IX_RD];
            const rdB = valuesB[IX_RD];
            const shimmerA = linearA[IX_SHIMMER];
            const shimmerB = linearB[IX_SHIMMER];
            const avA = linearA[IX_AV];
            const avB = linearB[IX_AV];
            const ahA = linearA[IX_AH];
            const ahB = linearB[IX_AH];
            const afA = linearA[IX_AF];
            const afB = linearB[IX_AF];
            const entryA = frameTable(frameIndex);
            const entryB = frameTable(frameIndex + 1);
            const tableA = entryA === null ? null : entryA.table;
            const tableB = entryB === null ? null : entryB.table;
            const sizeA = entryA === null ? 0 : entryA.size;
            const sizeB = entryB === null ? 0 : entryB.size;
            const voiced = entryA !== null || entryB !== null;
            for (; index < stop; index++) {
                const position = index * FRAME_RATE / sampleRate;
                const weight = position - frameIndex;
                const f0 = Math.max(F0_MIN, f0A * (1 - weight) + f0B * weight);
                const jitter = jitterA * (1 - weight) + jitterB * weight;
                const rd = rdA * (1 - weight) + rdB * weight;

                // The glottal cycle - jitter by the unperturbed phase's cycle, shimmer by the perturbed phase's
                const cycle0 = Math.floor(phase0);
                if (cycle0 !== jitterCycle) {
                    jitterCycle = cycle0;
                    jitterDraw = gaussAt(seeds.jitter, cycle0);
                }
                const frequency = f0 / Math.max(0.5, 1 + jitter * jitterDraw);
                const cycle = Math.floor(phase);
                const fraction = phase - cycle;
                if (cycle !== shimmerCycle) {
                    shimmerCycle = cycle;
                    shimmerDraws = [gaussAt(seeds.shimmer, cycle), gaussAt(seeds.shimmer, cycle + 1)];
                }
                const shimmerDraw = shimmerDraws[0] * (1 - fraction) + shimmerDraws[1] * fraction;
                const isOpen = fraction < Math.min(0.95, 0.45 + 0.17 * rd);
                phase0 += f0 / sampleRate;
                phase += frequency / sampleRate;

                // The harmonics, the amplitudes interpolated between the frames
                let sum = 0;
                if (voiced) {
                    // Each frame's table at the phase - cubic (Catmull-Rom) interpolation of the four points around it
                    let sumA = 0;
                    let sumB = 0;
                    if (tableA !== null) {
                        const point = fraction * sizeA;
                        const base = Math.floor(point);
                        const tt = point - base;
                        const p0 = tableA[base];
                        const p1 = tableA[base + 1];
                        const p2 = tableA[base + 2];
                        const p3 = tableA[base + 3];
                        sumA = p1 + tt * (0.5 * (p2 - p0) + tt * (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3 +
                            tt * (1.5 * (p1 - p2) + 0.5 * (p3 - p0))));
                    }
                    if (tableB !== null) {
                        const point = fraction * sizeB;
                        const base = Math.floor(point);
                        const tt = point - base;
                        const p0 = tableB[base];
                        const p1 = tableB[base + 1];
                        const p2 = tableB[base + 2];
                        const p3 = tableB[base + 3];
                        sumB = p1 + tt * (0.5 * (p2 - p0) + tt * (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3 +
                            tt * (1.5 * (p1 - p2) + 0.5 * (p3 - p0))));
                    }
                    const shimmerAmount = shimmerA * (1 - weight) + shimmerB * weight;
                    sum = (sumA * (1 - weight) + sumB * weight) * Math.max(0, 1 + shimmerAmount * shimmerDraw);
                }
                voicingData[index - voicingBase] = sum;

                // The noise inputs - the aspiration pulsing with the glottal cycle; two Gaussian draws, each the sum of
                // six uniform hashes (rng.js gaussAt and uniformAt, inline)
                const av = avA * (1 - weight) + avB * weight;
                const modulation = isOpen ? 1 : 1 - NOISE_MODULATION * Math.min(1, 2 * av);
                const ah = ahA * (1 - weight) + ahB * weight;
                const af = afA * (1 - weight) + afB * weight;
                let sumAspiration = 0;
                let sumFrication = 0;
                for (let offset = 0; offset < 6; offset++) {
                    const counter = Math.imul(((index * 6 + offset) + 1) >>> 0, 0x6D2B79F5);
                    let mixed = (aspirationSeed + counter) >>> 0;
                    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
                    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
                    sumAspiration += ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
                    let mixedF = (fricationSeed + counter) >>> 0;
                    mixedF = Math.imul(mixedF ^ (mixedF >>> 15), mixedF | 1);
                    mixedF ^= mixedF + Math.imul(mixedF ^ (mixedF >>> 7), mixedF | 61);
                    sumFrication += ((mixedF ^ (mixedF >>> 14)) >>> 0) / 4294967296;
                }
                aspirationData[index - aspirationBase] = (sumAspiration - 3) / SQRT_HALF * noiseScale * ah * modulation;
                fricationData[index - fricationBase] = (sumFrication - 3) / SQRT_HALF * noiseScale * af;
            }
        }
    };

    // Filter one noise frame and overlap-add it
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    const outRe = new Float64Array(fftSize);
    const outIm = new Float64Array(fftSize);
    const gainAspiration = new Float64Array(binCount);
    const gainFrication = new Float64Array(binCount);
    const noiseLinear = new Float64Array(NAMES.length);
    const poleK0 = new Float64Array(POLE_COUNT);
    const poleK1 = new Float64Array(POLE_COUNT);
    const poleK2 = new Float64Array(POLE_COUNT);
    const filterNoiseFrame = (frameNumber, frameCount) => {
        // The parameters at the frame's center
        const position = frameNumber * hop * FRAME_RATE / sampleRate;
        const frameIndex = Math.max(0, Math.min(frameCount - 2, Math.floor(position)));
        const weight = Math.max(0, Math.min(1, position - frameIndex));
        const linearA = linears[frameIndex - frameBase];
        const linearB = linears[Math.min(frameIndex + 1, frameCount - 1) - frameBase];
        const linear = noiseLinear;
        for (let ix = 0; ix < NAMES.length; ix++) {
            linear[ix] = linearA[ix] * (1 - weight) + linearB[ix] * weight;
        }
        const filters = frameFilters(linear);

        // The windowed noise - aspiration (real) and frication (imaginary) in one transform
        const start = frameNumber * hop - half;
        let energyAspiration = 0;
        let energyFrication = 0;
        const aspirationData = aspiration.data;
        const fricationData = frication.data;
        const aspirationBase = aspiration.base;
        const fricationBase = frication.base;
        const limit = total === null ? Infinity : total;
        for (let ix = 0; ix < fftSize; ix++) {
            const sample = start + ix;
            if (sample >= 0 && sample < limit) {
                // (every sample of a noise frame is generated before it's filtered, and kept until it's emitted)
                const aValue = aspirationData[sample - aspirationBase] * window[ix];
                const fValue = fricationData[sample - fricationBase] * window[ix];
                re[ix] = aValue;
                im[ix] = fValue;
                energyAspiration += aValue * aValue;
                energyFrication += fValue * fValue;
            } else {
                re[ix] = 0;
                im[ix] = 0;
            }
        }
        // Silent noise (below about -100 dB) isn't filtered
        const hasAspiration = energyAspiration > 1e-10 * fftSize;
        const hasFrication = energyFrication > 1e-10 * fftSize;
        if (!hasAspiration && !hasFrication) {
            return;
        }
        fft(re, im);

        // The filters' magnitudes (the spectral shapes - the levels are in the noise) per bin. A real biquad's squared
        // magnitude is k0 + k1 cos w + k2 cos 2w, so the tract (its poles and zero) and the high-pass need no complex
        // arithmetic; the bandpasses' responses are complex (they're summed)
        if (hasAspiration) {
            const {poleB, poleC, zero} = filters;
            for (let pole = 0; pole < POLE_COUNT; pole++) {
                const coefB = poleB[pole];
                const coefC = poleC[pole];
                poleK0[pole] = 1 + coefB * coefB + coefC * coefC;
                poleK1[pole] = 2 * coefB * (coefC - 1);
                poleK2[pole] = -2 * coefC;
            }
            const [, zeroB, zeroC] = zero;
            const zeroK0 = 1 + zeroB * zeroB + zeroC * zeroC;
            const zeroK1 = 2 * zeroB * (zeroC - 1);
            const zeroK2 = -2 * zeroC;
            const {gain} = filters;
            for (let bin = 0; bin < binCount; bin++) {
                const cos1 = binCos[bin];
                const cos2 = 2 * cos1 * cos1 - 1;
                let denominator = 1;
                for (let pole = 0; pole < POLE_COUNT; pole++) {
                    denominator *= poleK0[pole] + poleK1[pole] * cos1 + poleK2[pole] * cos2;
                }
                const numerator = zeroK0 + zeroK1 * cos1 + zeroK2 * cos2;
                gainAspiration[bin] = binEq[bin] * aspirationShape[bin] * Math.abs(gain) * Math.sqrt(numerator / denominator);
            }
        } else {
            gainAspiration.fill(0);
        }
        if (hasFrication) {
            const [b10, b11, b12, a11, a12] = filters.bp1;
            const [b20, b21, b22, a21, a22] = filters.bp2;
            const [h0, h1, h2, ha1, ha2] = filters.hp;
            const hpN0 = h0 * h0 + h1 * h1 + h2 * h2;
            const hpN1 = 2 * (h0 * h1 + h1 * h2);
            const hpN2 = 2 * h0 * h2;
            const hpD0 = 1 + ha1 * ha1 + ha2 * ha2;
            const hpD1 = 2 * (ha1 + ha1 * ha2);
            const hpD2 = 2 * ha2;
            const ab = linear[IX.ab];
            const ap1 = linear[IX.ap1];
            const ap2 = linear[IX.ap2];
            for (let bin = 0; bin < binCount; bin++) {
                const cos1 = binCos[bin];
                const sin1 = binSin[bin];
                const cos2 = 2 * cos1 * cos1 - 1;
                const sin2 = 2 * sin1 * cos1;
                // (a biquad's response (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2) at z = e^jw)
                const n1Re = b10 + b11 * cos1 + b12 * cos2;
                const n1Im = -(b11 * sin1 + b12 * sin2);
                const d1Re = 1 + a11 * cos1 + a12 * cos2;
                const d1Im = -(a11 * sin1 + a12 * sin2);
                const d1 = d1Re * d1Re + d1Im * d1Im;
                const bp1Re = (n1Re * d1Re + n1Im * d1Im) / d1;
                const bp1Im = (n1Im * d1Re - n1Re * d1Im) / d1;
                const n2Re = b20 + b21 * cos1 + b22 * cos2;
                const n2Im = -(b21 * sin1 + b22 * sin2);
                const d2Re = 1 + a21 * cos1 + a22 * cos2;
                const d2Im = -(a21 * sin1 + a22 * sin2);
                const d2 = d2Re * d2Re + d2Im * d2Im;
                const bp2Re = (n2Re * d2Re + n2Im * d2Im) / d2;
                const bp2Im = (n2Im * d2Re - n2Re * d2Im) / d2;
                const sumRe = ab + ap1 * (bp1Re * bp1Re - bp1Im * bp1Im) + ap2 * (bp2Re * bp2Re - bp2Im * bp2Im);
                const sumIm = ap1 * 2 * bp1Re * bp1Im + ap2 * 2 * bp2Re * bp2Im;
                const highpass = (hpN0 + hpN1 * cos1 + hpN2 * cos2) / (hpD0 + hpD1 * cos1 + hpD2 * cos2);
                gainFrication[bin] = binEq[bin] * Math.sqrt(sumRe * sumRe + sumIm * sumIm) * highpass;
            }
        } else {
            gainFrication.fill(0);
        }

        // Separate the two spectra (Xa = (Z[k] + Z*[N-k]) / 2, Xf = (Z[k] - Z*[N-k]) / 2j), filter, and recombine
        for (let bin = 0; bin < fftSize; bin++) {
            const mirror = (fftSize - bin) % fftSize;
            const gainBin = bin <= half ? bin : fftSize - bin;
            const aRe = 0.5 * (re[bin] + re[mirror]);
            const aIm = 0.5 * (im[bin] - im[mirror]);
            const fRe = 0.5 * (im[bin] + im[mirror]);
            const fIm = -0.5 * (re[bin] - re[mirror]);
            outRe[bin] = gainAspiration[gainBin] * aRe + gainFrication[gainBin] * fRe;
            outIm[bin] = gainAspiration[gainBin] * aIm + gainFrication[gainBin] * fIm;
        }
        fft(outRe, outIm, true);
        noiseOut.ensure(start + fftSize - 1);
        const outData = noiseOut.data;
        const outBase = noiseOut.base;
        for (let ix = 0; ix < fftSize; ix++) {
            outData[start + ix - outBase] += outRe[ix] / fftSize * window[ix];
        }
    };

    // The windows' squared sum at a sample, over the noise frames 0 ... frameLimit - 1
    const windowSum = (index, frameLimit) => {
        let sum = 0;
        const first = Math.max(0, Math.floor((index - half) / hop) + 1);
        const last = Math.min(frameLimit - 1, Math.floor((index + half) / hop));
        for (let frameNumber = first; frameNumber <= last; frameNumber++) {
            const offset = index - frameNumber * hop + half;
            if (offset >= 0 && offset < fftSize) {
                sum += window[offset] * window[offset];
            }
        }
        return sum;
    };

    // The windows' squared sum in the steady state (every noise frame over a sample present) by the sample's offset in a hop
    // - windowSum's terms in its order
    const steadySum = new Float64Array(hop);
    for (let offset = 0; offset < hop; offset++) {
        steadySum[offset] = windowSum(fftSize + offset, Number.MAX_SAFE_INTEGER);
    }

    // Deliver final samples in blocks
    const emit = (until, frameLimit) => {
        const noiseData = noiseOut.data;
        const noiseBase = noiseOut.base;
        const voicingData = voicing.data;
        const voicingBase = voicing.base;
        // (the steady state: every frame over the sample - from frame 1 to below frameLimit)
        const steadyEnd = frameLimit === Number.MAX_SAFE_INTEGER ? Infinity : (frameLimit - 1) * hop - half;
        for (; emitted < until; emitted++) {
            const noiseOffset = emitted - noiseBase;
            const noiseValue = noiseOffset >= 0 && noiseOffset < noiseData.length ? noiseData[noiseOffset] : 0;
            const voicingValue = voicingData[emitted - voicingBase];
            const sum = emitted >= half && emitted < steadyEnd ? steadySum[emitted % hop] : windowSum(emitted, frameLimit);
            const noise = noiseValue / Math.max(1e-3, sum);
            block[blockLength++] = voicingValue + noise;
            if (blockLength === blockSize) {
                if (onBlock !== null) {
                    onBlock(block);
                }
                block = new Float32Array(blockSize);
                blockLength = 0;
            }
        }
        // Drop what's no longer needed
        const keep = Math.max(0, emitted - fftSize);
        for (const buffer of [voicing, aspiration, frication, noiseOut]) {
            buffer.trim(keep);
        }
        const keepFrame = Math.max(0, Math.floor(Math.min(generated, noiseFrames * hop) * FRAME_RATE / sampleRate) - 2);
        if (keepFrame - frameBase > 1024) {
            frames.splice(0, keepFrame - frameBase);
            linears.splice(0, keepFrame - frameBase);
            for (const key of amplitudes.keys()) {
                if (key < keepFrame) {
                    amplitudes.delete(key);
                }
            }
            for (const key of tables.keys()) {
                if (key < keepFrame - 1) {
                    tables.delete(key);
                }
            }
            frameBase = keepFrame;
        }
    };

    // Advance as far as the frames so far allow
    const advance = () => {
        const frameCount = frameBase + frames.length;
        // (the samples whose next two frames are known - those before frame frameCount - 2: a frame's voicing table is
        // made with its pair's, so both are pushed first - the same tables however the frames are pushed)
        let until = generated;
        while (Math.floor(until * FRAME_RATE / sampleRate) + 2 < frameCount) {
            until++;
        }
        if (until > generated) {
            generateSamples(generated, until);
            generated = until;
        }
        while (noiseFrames * hop + half <= generated) {
            filterNoiseFrame(noiseFrames, frameCount);
            noiseFrames++;
        }
        // A sample is final when every noise frame over it is done
        const finalUntil = Math.max(0, noiseFrames * hop - half);
        emit(Math.min(finalUntil, generated), Number.MAX_SAFE_INTEGER);
    };

    return {
        sampleRate,

        // Push a frame (or an array of frames)
        push(frameOrFrames) {
            const list = Array.isArray(frameOrFrames) && typeof frameOrFrames[0] === 'object' && frameOrFrames[0] !== null ?
                frameOrFrames : [frameOrFrames];
            for (const frame of list) {
                const values = frameValues(frame);
                frames.push(values);
                linears.push(linearValues(values));
            }
            if (total === null) {
                advance();
            }
        },

        // Finish - render the rest and deliver the last block; returns the total sample count
        end() {
            // (push generated every sample but those of the last frames - up to the last frame, where the total ends)
            const frameCount = frameBase + frames.length;
            total = sampleCount(frameCount, sampleRate);
            if (total > generated) {
                generateSamples(generated, total);
                generated = total;
            }
            const noiseFrameCount = total > 0 ? Math.floor(total / hop) + 1 : 0;
            for (; noiseFrames < noiseFrameCount; noiseFrames++) {
                filterNoiseFrame(noiseFrames, frameCount);
            }
            emit(total, noiseFrameCount);
            if (blockLength > 0 && onBlock !== null) {
                onBlock(block.subarray(0, blockLength));
            }
            blockLength = 0;
            return total;
        }
    };
}


// Render frames (objects or arrays of parameter values) to a Float32Array - onBlock, if given, receives each block as
// it's final
export function render(frames, {sampleRate = 44100, seed = 1, onBlock = null, blockSize = 4096, eq = null} = {}) {
    const blocks = [];
    const renderer = createRenderer({sampleRate, seed, blockSize, eq, 'onBlock': (block) => {
        blocks.push(block.slice());
        if (onBlock !== null) {
            onBlock(block);
        }
    }});
    for (const frame of frames) {
        renderer.push([frame]);
    }
    const total = renderer.end();
    const out = new Float32Array(total);
    let offset = 0;
    for (const block of blocks) {
        out.set(block, offset);
        offset += block.length;
    }
    return out;
}
