// A Klatt-style formant synthesizer - a glottal source and aspiration through a cascade of formant resonators
// (with a nasal pole-zero pair), and frication noise through a parallel bank of formant resonators
//
// The synthesizer runs on parameter frames (every frameMs), linearly interpolated per sample. Each frame has:
//
// - f0 - the fundamental frequency (Hz)
// - av - the voicing amplitude (linear, 0 to 1)
// - ah - the aspiration amplitude (linear)
// - af - the frication amplitude (linear)
// - rd - the LF voice quality (0.3 pressed to 2.7 breathy)
// - f1-f6, b1-b6 - the formant frequencies and bandwidths (Hz)
// - fnp, bnp, fnz, bnz - the nasal pole and zero frequencies and bandwidths (Hz) - equal pole and zero cancel
// - fp1, bp1, ap1, fp2, bp2, ap2 - the frication noise's two resonances (fourth-order): frequencies, bandwidths (Hz),
//   and amplitudes
// - ab - the frication noise's flat (bypass) amplitude
// - fhp - the frication noise's high-pass cutoff (Hz) - a steep (24 dB/octave) low-frequency edge
// - jit, shim - the jitter and shimmer multipliers (of the jitter and shimmer options)
// - ob1 - the F1 bandwidth increase (Hz) while the glottis is open (subglottal coupling)
//
// Missing frame parameters take their defaultFrame values.

import {lfPulse} from './glottal.js';


// A second-order digital resonator (Klatt) - unity gain at DC
class Resonator {
    constructor() {
        this.y1 = 0;
        this.y2 = 0;
        this.coefA = 1;
        this.coefB = 0;
        this.coefC = 0;
    }

    set(frequency, bandwidth, sampleRate) {
        const radius = Math.exp(-Math.PI * bandwidth / sampleRate);
        this.coefC = -radius * radius;
        this.coefB = 2 * radius * Math.cos(2 * Math.PI * frequency / sampleRate);
        this.coefA = 1 - this.coefB - this.coefC;
    }

    step(input) {
        const output = this.coefA * input + this.coefB * this.y1 + this.coefC * this.y2;
        this.y2 = this.y1;
        this.y1 = output;
        return output;
    }
}


// A second-order digital antiresonator (a zero pair) - unity gain at DC
class Antiresonator {
    constructor() {
        this.x1 = 0;
        this.x2 = 0;
        this.coefA = 1;
        this.coefB = 0;
        this.coefC = 0;
    }

    set(frequency, bandwidth, sampleRate) {
        const radius = Math.exp(-Math.PI * bandwidth / sampleRate);
        const poleC = -radius * radius;
        const poleB = 2 * radius * Math.cos(2 * Math.PI * frequency / sampleRate);
        const poleA = 1 - poleB - poleC;
        this.coefA = 1 / poleA;
        this.coefB = -poleB / poleA;
        this.coefC = -poleC / poleA;
    }

    step(input) {
        const output = this.coefA * input + this.coefB * this.x1 + this.coefC * this.x2;
        this.x2 = this.x1;
        this.x1 = input;
        return output;
    }
}


// A bandpass filter with a 0 dB peak (RBJ biquad) - for shaping frication noise
export class Bandpass {
    constructor() {
        this.x1 = 0;
        this.x2 = 0;
        this.y1 = 0;
        this.y2 = 0;
        this.b0 = 0;
        this.b2 = 0;
        this.a1 = 0;
        this.a2 = 0;
    }

    set(frequency, bandwidth, sampleRate) {
        const w0 = 2 * Math.PI * Math.min(frequency, 0.45 * sampleRate) / sampleRate;
        const alpha = Math.sin(w0) / (2 * Math.max(0.3, frequency / bandwidth));
        const a0 = 1 + alpha;
        this.b0 = alpha / a0;
        this.b2 = -alpha / a0;
        this.a1 = -2 * Math.cos(w0) / a0;
        this.a2 = (1 - alpha) / a0;
    }

    step(input) {
        const output = this.b0 * input + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
        this.x2 = this.x1;
        this.x1 = input;
        this.y2 = this.y1;
        this.y1 = output;
        return output;
    }
}


// A Butterworth high-pass biquad (RBJ)
export class HighPass {
    constructor() {
        this.x1 = 0;
        this.x2 = 0;
        this.y1 = 0;
        this.y2 = 0;
        this.set(500, 44100);
    }

    set(frequency, sampleRate) {
        const w0 = 2 * Math.PI * Math.min(frequency, 0.45 * sampleRate) / sampleRate;
        const alpha = Math.sin(w0) / Math.sqrt(2);
        const cos = Math.cos(w0);
        const a0 = 1 + alpha;
        this.b0 = (1 + cos) / 2 / a0;
        this.b1 = -(1 + cos) / a0;
        this.b2 = (1 + cos) / 2 / a0;
        this.a1 = -2 * cos / a0;
        this.a2 = (1 - alpha) / a0;
    }

    step(input) {
        const output = this.b0 * input + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
        this.x2 = this.x1;
        this.x1 = input;
        this.y2 = this.y1;
        this.y1 = output;
        return output;
    }
}


// An RBJ high-shelf biquad - for the voice source's spectral tilt
export class HighShelf {
    constructor(frequency, gainDb, sampleRate) {
        const amp = 10 ** (gainDb / 40);
        const w0 = 2 * Math.PI * frequency / sampleRate;
        const alpha = Math.sin(w0) / 2 * Math.sqrt(2);
        const cos = Math.cos(w0);
        const sqrtA = 2 * Math.sqrt(amp) * alpha;
        const a0 = (amp + 1) - (amp - 1) * cos + sqrtA;
        this.b0 = amp * ((amp + 1) + (amp - 1) * cos + sqrtA) / a0;
        this.b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos) / a0;
        this.b2 = amp * ((amp + 1) + (amp - 1) * cos - sqrtA) / a0;
        this.a1 = 2 * ((amp - 1) - (amp + 1) * cos) / a0;
        this.a2 = ((amp + 1) - (amp - 1) * cos - sqrtA) / a0;
        this.x1 = 0;
        this.x2 = 0;
        this.y1 = 0;
        this.y2 = 0;
    }

    step(input) {
        const output = this.b0 * input + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
        this.x2 = this.x1;
        this.x1 = input;
        this.y2 = this.y1;
        this.y1 = output;
        return output;
    }
}


// A seeded random number generator (mulberry32), so syntheses are repeatable
export function makeRandom(seed) {
    let state = seed >>> 0;
    const uniform = () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = state;
        mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
    const gauss = () => {
        let sum = 0;
        for (let ix = 0; ix < 6; ix++) {
            sum += uniform();
        }
        return (sum - 3) / Math.sqrt(0.5);
    };
    return {uniform, gauss};
}


// The default frame parameters
export const defaultFrame = {
    'f0': 100, 'av': 0, 'ah': 0, 'af': 0, 'rd': 1,
    'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': 3500, 'f5': 4500, 'f6': 5500,
    'b1': 80, 'b2': 90, 'b3': 150, 'b4': 250, 'b5': 300, 'b6': 1000,
    'fnp': 270, 'bnp': 100, 'fnz': 270, 'bnz': 100,
    'fp1': 3000, 'bp1': 1000, 'ap1': 0, 'fp2': 6000, 'bp2': 2000, 'ap2': 0, 'ab': 0, 'fhp': 500,
    'jit': 1, 'shim': 1, 'ob1': 0
};


// The frame parameter names, interpolated per sample
const frameKeys = Object.keys(defaultFrame);


// Synthesize parameter frames - returns a Float32Array at the sample rate. The options are the frame period (ms),
// the sample rate, the random seed, and the voice's jitter (fraction), shimmer (fraction), and glottal noise modulation
// (0 to 1 - how much the aspiration pulses with the glottal cycle), the voicing's spectral tilt correction - a high shelf
// (sourceShelfHz, sourceShelfDb), the breath noise's high shelf while voicing (noiseShelfDb, at sourceShelfHz - the
// glottal turbulence that masks the upper harmonics), and the fixed higher formants above F6 - [[frequency, bandwidth],
// ...]. A real vocal tract resonates every ~1 kHz; without these higher poles, the cascade's response falls 12 dB/octave
// for each formant passed. For streaming, onBlock(samples) is called with each block of blockSize samples (a view of the
// output) as soon as it is synthesized.
export function synthesize(framesIn, {frameMs = 5, sampleRate = 44100, seed = 1, jitter = 0.006, shimmer = 0.03, noiseModulation = 0.7,
    sourceShelfHz = 1500, sourceShelfDb = 0, noiseShelfDb = 0, higherFormants = [], onBlock = null, blockSize = 2048} = {}) {
    const random = makeRandom(seed);
    const frames = framesIn.map((frame) => ({...defaultFrame, ...frame}));
    const samplesPerFrame = frameMs * sampleRate / 1000;
    const totalSamples = Math.floor((frames.length - 1) * samplesPerFrame);
    const out = new Float32Array(totalSamples);

    // The cascade (voicing and aspiration) and parallel (frication) resonators
    const nasalZero = new Antiresonator();
    const nasalPole = new Resonator();
    const cascade = [new Resonator(), new Resonator(), new Resonator(), new Resonator(), new Resonator(), new Resonator()];
    const f1Closed = new Resonator();
    const f1Open = new Resonator();
    const higher = higherFormants.filter(([frequency]) => frequency < 0.45 * sampleRate).map(([frequency, bandwidth]) => {
        const resonator = new Resonator();
        resonator.set(frequency, bandwidth, sampleRate);
        return resonator;
    });
    const sourceShelf = new HighShelf(sourceShelfHz, sourceShelfDb, sampleRate);
    const noiseShelf = new HighShelf(sourceShelfHz, noiseShelfDb, sampleRate);
    const frication1 = new Bandpass();
    const frication1b = new Bandpass();
    const frication2 = new Bandpass();
    const frication2b = new Bandpass();
    const fricationHighPass1 = new HighPass();
    const fricationHighPass2 = new HighPass();

    // The glottal period state
    let pulse = null;
    let periodLength = 0;
    let periodTime = 0;
    let pulseGain = 1;
    let openFraction = 0.6;

    // A gently low-passed (~5 kHz) noise source for the aspiration
    let aspirationLow = 0;
    const current = {};
    for (let ix = 0; ix < totalSamples; ix++) {
        // Interpolate the frame parameters, updating the filters every 16 samples
        const framePosition = ix / samplesPerFrame;
        const frameIndex = Math.floor(framePosition);
        const fraction = framePosition - frameIndex;
        const frameA = frames[frameIndex];
        const frameB = frames[frameIndex + 1];
        for (const key of frameKeys) {
            current[key] = frameA[key] + (frameB[key] - frameA[key]) * fraction;
        }
        if (ix % 16 === 0) {
            nasalZero.set(current.fnz, current.bnz, sampleRate);
            nasalPole.set(current.fnp, current.bnp, sampleRate);
            for (let kx = 0; kx < 6; kx++) {
                cascade[kx].set(current[`f${kx + 1}`], current[`b${kx + 1}`], sampleRate);
            }
            f1Closed.set(current.f1, current.b1, sampleRate);
            f1Open.set(current.f1, current.b1 + current.ob1, sampleRate);
            frication1.set(current.fp1, current.bp1, sampleRate);
            frication1b.set(current.fp1, current.bp1, sampleRate);
            frication2.set(current.fp2, current.bp2, sampleRate);
            frication2b.set(current.fp2, current.bp2, sampleRate);
            fricationHighPass1.set(current.fhp, sampleRate);
            fricationHighPass2.set(current.fhp, sampleRate);
        }

        // The glottal source - a new LF pulse each period, with jitter and shimmer
        const period = 1 / Math.max(40, current.f0);
        if (pulse === null || periodTime >= periodLength) {
            periodTime = pulse === null ? 0 : periodTime - periodLength;
            periodLength = period * (1 + jitter * current.jit * random.gauss());
            pulse = lfPulse(periodLength, current.rd);
            pulseGain = Math.max(0, 1 + shimmer * current.shim * random.gauss());
            openFraction = Math.min(0.95, 0.45 + 0.17 * current.rd);
        }
        const voicing = sourceShelf.step(current.av * pulseGain * pulse(periodTime));
        const isOpen = periodTime < openFraction * periodLength;
        periodTime += 1 / sampleRate;

        // Aspiration - noise pulsing with the glottal opening while voicing (steady when the glottis is open)
        aspirationLow = 0.5 * aspirationLow + 0.5 * random.gauss();
        const modulation = noiseModulation * Math.min(1, 2 * current.av);
        // While voicing, the breath noise is brightened by its own shelf (voiceless aspiration keeps its spectrum)
        const breath = current.ah * aspirationLow * (isOpen ? 1 : 1 - modulation);
        const aspiration = breath + Math.min(1, 2 * current.av) * (noiseShelf.step(breath) - breath);

        // The cascade branch
        let signal = voicing + aspiration;
        signal = nasalPole.step(nasalZero.step(signal));
        for (let kx = higher.length - 1; kx >= 0; kx--) {
            signal = higher[kx].step(signal);
        }
        for (let kx = 5; kx >= 1; kx--) {
            signal = cascade[kx].step(signal);
        }
        // F1 - its bandwidth widened while the glottis is open
        const [f1] = cascade;
        const coefficients = isOpen ? f1Open : f1Closed;
        f1.coefA = coefficients.coefA;
        f1.coefB = coefficients.coefB;
        f1.coefC = coefficients.coefC;
        signal = f1.step(signal);

        // The frication branch - noise through two resonances, plus a flat bypass
        const noise = current.af * random.gauss();
        const fricationSum = current.ab * noise + current.ap1 * frication1b.step(frication1.step(noise)) +
            current.ap2 * frication2b.step(frication2.step(noise));
        const fricative = fricationHighPass2.step(fricationHighPass1.step(fricationSum));
        out[ix] = signal + fricative;

        // Stream each finished block
        if (onBlock !== null && ((ix + 1) % blockSize === 0 || ix === totalSamples - 1)) {
            onBlock(out.subarray(ix + 1 - ((ix % blockSize) + 1), ix + 1));
        }
    }
    return out;
}


// Normalize samples to a peak level
export function normalize(samples, peak = 0.9) {
    let max = 0;
    for (const sample of samples) {
        max = Math.max(max, Math.abs(sample));
    }
    const scale = max > 0 ? peak / max : 1;
    return samples.map((sample) => sample * scale);
}
