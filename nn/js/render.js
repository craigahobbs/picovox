// The parameter-frame renderer - nn/dsynth.py's source-filter synthesizer in JavaScript, for the browser and Node.
// It renders frames of the parameter set (nn/params.py, nn/js/params.js) at any sample rate, streaming: frames are
// pushed as they're predicted, and blocks of samples are delivered as soon as they're final.
//
//     import {render} from './nn/js/render.js';
//     const samples = render(frames, {'sampleRate': 44100, 'seed': 1});   // frames: [{f0, av, ...}, ...] or arrays
//
//     const renderer = createRenderer({'sampleRate': 44100, 'onBlock': (block) => play(block)});
//     renderer.push(frame); ...; renderer.end();
//
// The algorithm is nn/dsynth.py's, sample for sample (with its counter-based noise, nn/js/rng.js):
//
// - Voicing - harmonic synthesis: each frame's complex harmonic amplitudes are the voicing amplitude times the LF
//   pulse's Fourier coefficients (nn/js/lf.js, interpolated in Rd), the source shelf, and the vocal tract's response
//   at k * F0 (lib/klatt.js's filters at the 44.1 kHz reference rate, as analytic frequency responses), tapered near
//   the Nyquist frequency; they're interpolated linearly between frames and summed against the glottal phase, which
//   carries the jitter (a Gaussian per cycle) and the shimmer (Gaussians interpolated across each cycle).
// - Noise - white noise scaled per sample by its level (ah, af interpolated; the aspiration also modulated by the
//   glottal cycle), filtered in the frequency domain: Hann frames of noiseFftSize samples, hop a quarter of that,
//   each spectrum multiplied by the filters' magnitudes (the spectral shapes) at the frame's interpolated
//   parameters, overlap-added and normalized by the windows' squared sum.
// - The spectral correction (optional - a model's correction head, nn/params.py CORRECTION_*): a frame given as an array
//   longer than the parameters carries K band gains (dB) after them (an object frame, as its 'correction' array) - a
//   smooth envelope (linear in dB over log frequency) multiplying the frame's harmonic amplitudes and, interpolated in
//   dB to each noise frame, the noise spectra.

import {
    CORRECTION_HIGH_HZ, CORRECTION_LOW_HZ, DB_NAMES, EQ_BAND_COUNT, EQ_HIGH_HZ, EQ_LOW_HZ, F0_MIN, FLOOR_DB, FRAME_RATE, HARMONIC_TAPER_END,
    HARMONIC_TAPER_START, HIGHER_POLES, HIGHER_POLE_BANDWIDTH, HIGHER_POLE_BANDWIDTH_STEP, HIGHER_POLE_SPACING,
    LF_HARMONICS, LF_RD_MAX, LF_RD_MIN, LF_RD_STEP, NAMES, NOISE_MODULATION, PARAMETERS, REF_RATE, SHELF_HZ
} from './params.js';
import {LF_RD_COUNT, lfHarmonics, openFraction} from './lf.js';
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


// A frame's spectral correction gains (dB) - the values after the parameters of an array frame, or an object frame's
// 'correction' - as a Float64Array, or null if it has none
export function frameCorrection(frame) {
    if (Array.isArray(frame) || ArrayBuffer.isView(frame)) {
        return frame.length > NAMES.length ? Float64Array.from(frame.slice(NAMES.length)) : null;
    }
    return frame.correction === undefined || frame.correction === null ? null : Float64Array.from(frame.correction);
}


// Parameter values with the levels as linear amplitudes (the interpolable form)
function linearValues(values) {
    const linear = new Float64Array(values.length);
    for (let ix = 0; ix < values.length; ix++) {
        linear[ix] = DB_SET.has(NAMES[ix]) ? 10 ** (values[ix] / 20) : values[ix];
    }
    return linear;
}


// A resonator's coefficients (lib/klatt.js, at the reference rate) - [A, B, C]
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
    const b0 = coefs[0];
    const b1 = coefs[1];
    const b2 = coefs[2];
    const a1 = coefs[3];
    const a2 = coefs[4];
    const numRe = b0 + b1 * cos1 + b2 * cos2;
    const numIm = -(b1 * sin1 + b2 * sin2);
    const denRe = 1 + a1 * cos1 + a2 * cos2;
    const denIm = -(a1 * sin1 + a2 * sin2);
    const den = denRe * denRe + denIm * denIm;
    biquadIm = (numIm * denRe - numRe * denIm) / den;
    return (numRe * denRe + numIm * denIm) / den;
}


// The source shelf's biquad coefficients (lib/klatt.js HighShelf at SHELF_HZ)
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


// The frication bandpass's coefficients (lib/klatt.js Bandpass)
function bandpassCoefficients(center, bandwidth) {
    const w0 = 2 * Math.PI * Math.min(center, 0.45 * REF_RATE) / REF_RATE;
    const alpha = Math.sin(w0) / (2 * Math.max(0.3, center / bandwidth));
    const a0 = 1 + alpha;
    return [alpha / a0, 0, -alpha / a0, -2 * Math.cos(w0) / a0, (1 - alpha) / a0];
}


// The frication high-pass's coefficients (lib/klatt.js HighPass)
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
    const zeroB = filters.zero[1];
    const zeroC = filters.zero[2];
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


// The spectral correction's linear gain at a frequency, for band gains (dB) - interpolated linearly in dB over log frequency
function correctionGain(correction, frequency) {
    const bands = correction.length;
    const step = Math.log(CORRECTION_HIGH_HZ / CORRECTION_LOW_HZ) / (bands - 1);
    const position = Math.max(0, Math.min(bands - 1, Math.log(Math.max(1, frequency) / CORRECTION_LOW_HZ) / step));
    const low = Math.min(bands - 2, Math.floor(position));
    const weight = position - low;
    return 10 ** ((correction[low] * (1 - weight) + correction[low + 1] * weight) / 20);
}


// A frame's complex harmonic amplitudes - {re, im} Float64Arrays indexed by harmonic number (1 ... count); correction is
// the frame's spectral correction (dB), or null
function harmonicAmplitudes(linear, filters, sampleRate, eq, correction = null) {
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
        let scale = 2 * av * harmonicTaper(frequency, sampleRate) * eqGain(eq, frequency);
        if (correction !== null) {
            scale *= correctionGain(correction, frequency);
        }
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

    get(index) {
        const offset = index - this.base;
        return offset >= 0 && offset < this.data.length ? this.data[offset] : 0;
    }

    set(index, value) {
        this.ensure(index);
        this.data[index - this.base] = value;
    }

    add(index, value) {
        this.ensure(index);
        this.data[index - this.base] += value;
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
// sample count. eq is the optional channel equalizer's gains (dB, EQ_BAND_COUNT of them - nn/params.py).
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

    // The frames (values, linear values, and spectral corrections), and per-frame harmonic amplitudes, by absolute frame
    // index
    const frames = [];
    const linears = [];
    const corrections = [];
    let frameBase = 0;
    const amplitudes = new Map();
    const frameAmplitudes = (index) => {
        if (!amplitudes.has(index)) {
            const linear = linears[index - frameBase];
            amplitudes.set(index, harmonicAmplitudes(linear, frameFilters(linear), sampleRate, eq, corrections[index - frameBase]));
        }
        return amplitudes.get(index);
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

    // Generate one sample's voicing and noise inputs
    let pairIndex = -1;
    let pairA = null;
    let pairB = null;
    const generate = (index) => {
        const position = index * FRAME_RATE / sampleRate;
        const frameIndex = Math.floor(position);
        const weight = position - frameIndex;
        const valuesA = frames[frameIndex - frameBase];
        const valuesB = frames[frameIndex + 1 - frameBase];
        const linearA = linears[frameIndex - frameBase];
        const linearB = linears[frameIndex + 1 - frameBase];
        const f0 = Math.max(F0_MIN, valuesA[IX_F0] * (1 - weight) + valuesB[IX_F0] * weight);
        const jitter = valuesA[IX_JITTER] * (1 - weight) + valuesB[IX_JITTER] * weight;
        const rd = valuesA[IX_RD] * (1 - weight) + valuesB[IX_RD] * weight;

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
        const isOpen = fraction < openFraction(rd);
        phase0 += f0 / sampleRate;
        phase += frequency / sampleRate;

        // The harmonics, the amplitudes interpolated between the frames (the frames' amplitudes kept while the frame is)
        if (frameIndex !== pairIndex) {
            pairIndex = frameIndex;
            pairA = frameAmplitudes(frameIndex);
            pairB = frameAmplitudes(frameIndex + 1);
        }
        const ampA = pairA;
        const ampB = pairB;
        let sum = 0;
        if (ampA.count > 0 || ampB.count > 0) {
            const angle = 2 * Math.PI * fraction;
            const stepRe = Math.cos(angle);
            const stepIm = Math.sin(angle);
            let zRe = stepRe;
            let zIm = stepIm;
            let sumA = 0;
            let sumB = 0;
            const reA = ampA.re;
            const imA = ampA.im;
            const reB = ampB.re;
            const imB = ampB.im;
            const both = Math.min(ampA.count, ampB.count);
            const count = Math.max(ampA.count, ampB.count);
            let kx = 1;
            for (; kx <= both; kx++) {
                sumA += reA[kx] * zRe - imA[kx] * zIm;
                sumB += reB[kx] * zRe - imB[kx] * zIm;
                const nextRe = zRe * stepRe - zIm * stepIm;
                zIm = zRe * stepIm + zIm * stepRe;
                zRe = nextRe;
            }
            if (ampA.count > both) {
                for (; kx <= count; kx++) {
                    sumA += reA[kx] * zRe - imA[kx] * zIm;
                    const nextRe = zRe * stepRe - zIm * stepIm;
                    zIm = zRe * stepIm + zIm * stepRe;
                    zRe = nextRe;
                }
            } else {
                for (; kx <= count; kx++) {
                    sumB += reB[kx] * zRe - imB[kx] * zIm;
                    const nextRe = zRe * stepRe - zIm * stepIm;
                    zIm = zRe * stepIm + zIm * stepRe;
                    zRe = nextRe;
                }
            }
            const shimmerAmount = linearA[IX_SHIMMER] * (1 - weight) + linearB[IX_SHIMMER] * weight;
            sum = (sumA * (1 - weight) + sumB * weight) * Math.max(0, 1 + shimmerAmount * shimmerDraw);
        }
        voicing.set(index, sum);

        // The noise inputs - the aspiration pulsing with the glottal cycle
        const av = linearA[IX_AV] * (1 - weight) + linearB[IX_AV] * weight;
        const modulation = isOpen ? 1 : 1 - NOISE_MODULATION * Math.min(1, 2 * av);
        const ah = linearA[IX_AH] * (1 - weight) + linearB[IX_AH] * weight;
        const af = linearA[IX_AF] * (1 - weight) + linearB[IX_AF] * weight;
        aspiration.set(index, gaussAt(seeds.aspiration, index) * noiseScale * ah * modulation);
        frication.set(index, gaussAt(seeds.frication, index) * noiseScale * af);
    };

    // Filter one noise frame and overlap-add it
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    const outRe = new Float64Array(fftSize);
    const outIm = new Float64Array(fftSize);
    const gainAspiration = new Float64Array(binCount);
    const gainFrication = new Float64Array(binCount);
    const noiseLinear = new Float64Array(NAMES.length);
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

        // The spectral correction at the frame's center (interpolated in dB; a frame without one is 0 dB)
        const correctionA = corrections[frameIndex - frameBase];
        const correctionB = corrections[Math.min(frameIndex + 1, frameCount - 1) - frameBase];
        let correction = null;
        if (correctionA !== null || correctionB !== null) {
            const bands = (correctionA ?? correctionB).length;
            correction = new Float64Array(bands);
            for (let band = 0; band < bands; band++) {
                correction[band] = (correctionA?.[band] ?? 0) * (1 - weight) + (correctionB?.[band] ?? 0) * weight;
            }
        }

        // The windowed noise - aspiration (real) and frication (imaginary) in one transform
        const start = frameNumber * hop - half;
        let energyAspiration = 0;
        let energyFrication = 0;
        for (let ix = 0; ix < fftSize; ix++) {
            const sample = start + ix;
            const inside = sample >= 0 && (total === null || sample < total);
            re[ix] = inside ? aspiration.get(sample) * window[ix] : 0;
            im[ix] = inside ? frication.get(sample) * window[ix] : 0;
            energyAspiration += re[ix] * re[ix];
            energyFrication += im[ix] * im[ix];
        }
        // Silent noise (below about -100 dB) isn't filtered
        const hasAspiration = energyAspiration > 1e-10 * fftSize;
        const hasFrication = energyFrication > 1e-10 * fftSize;
        if (!hasAspiration && !hasFrication) {
            return;
        }
        fft(re, im);

        // The filters' magnitudes (the spectral shapes - the levels are in the noise) per bin
        for (let bin = 0; bin < binCount; bin++) {
            const cos1 = binCos[bin];
            const sin1 = binSin[bin];
            const cos2 = 2 * cos1 * cos1 - 1;
            const sin2 = 2 * sin1 * cos1;
            if (hasAspiration) {
                const tractRe = tractResponse(filters, cos1, sin1);
                gainAspiration[bin] = binEq[bin] * aspirationShape[bin] * Math.sqrt(tractRe * tractRe + tractIm * tractIm);
            } else {
                gainAspiration[bin] = 0;
            }
            if (hasFrication) {
                const bp1Re = biquad(filters.bp1, cos1, sin1, cos2, sin2);
                const bp1Im = biquadIm;
                const bp2Re = biquad(filters.bp2, cos1, sin1, cos2, sin2);
                const bp2Im = biquadIm;
                const sumRe = linear[IX.ab] + linear[IX.ap1] * (bp1Re * bp1Re - bp1Im * bp1Im) +
                    linear[IX.ap2] * (bp2Re * bp2Re - bp2Im * bp2Im);
                const sumIm = linear[IX.ap1] * 2 * bp1Re * bp1Im + linear[IX.ap2] * 2 * bp2Re * bp2Im;
                const hpRe = biquad(filters.hp, cos1, sin1, cos2, sin2);
                const highpass = hpRe * hpRe + biquadIm * biquadIm;
                gainFrication[bin] = binEq[bin] * Math.sqrt(sumRe * sumRe + sumIm * sumIm) * highpass;
            } else {
                gainFrication[bin] = 0;
            }
            if (correction !== null) {
                const gain = correctionGain(correction, bin * sampleRate / fftSize);
                gainAspiration[bin] *= gain;
                gainFrication[bin] *= gain;
            }
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
        for (let ix = 0; ix < fftSize; ix++) {
            noiseOut.add(start + ix, outRe[ix] / fftSize * window[ix]);
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

    // Deliver final samples in blocks
    const emit = (until, frameLimit) => {
        for (; emitted < until; emitted++) {
            const noise = noiseOut.get(emitted) / Math.max(1e-3, windowSum(emitted, frameLimit));
            block[blockLength++] = voicing.get(emitted) + noise;
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
            corrections.splice(0, keepFrame - frameBase);
            for (const key of amplitudes.keys()) {
                if (key < keepFrame) {
                    amplitudes.delete(key);
                }
            }
            frameBase = keepFrame;
        }
    };

    // Advance as far as the frames so far allow
    const advance = () => {
        const frameCount = frameBase + frames.length;
        while (Math.floor(generated * FRAME_RATE / sampleRate) + 1 < frameCount) {
            generate(generated);
            generated++;
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
                corrections.push(frameCorrection(frame));
            }
            if (total === null) {
                advance();
            }
        },

        // Finish - render the rest and deliver the last block; returns the total sample count
        end() {
            const frameCount = frameBase + frames.length;
            total = sampleCount(frameCount, sampleRate);
            while (generated < total) {
                generate(generated);
                generated++;
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
