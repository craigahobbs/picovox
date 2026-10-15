import {Bandpass, HighPass, HighShelf, defaultFrame, makeRandom, normalize, synthesize} from '../lib/klatt.js';
import {decimate, formants, hann, pitchTrack} from '../lib/analysis.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A filter's magnitude response at a frequency, from its impulse response
function magnitude(filter, frequency, sampleRate, length = 8192) {
    let re = 0;
    let im = 0;
    for (let ix = 0; ix < length; ix++) {
        const response = filter.step(ix === 0 ? 1 : 0);
        const angle = -2 * Math.PI * frequency * ix / sampleRate;
        re += response * Math.cos(angle);
        im += response * Math.sin(angle);
    }
    return Math.hypot(re, im);
}


// The root-mean-square of samples
function rms(samples) {
    let sum = 0;
    for (const sample of samples) {
        sum += sample * sample;
    }
    return Math.sqrt(sum / samples.length);
}


// Steady vowel frames
function vowelFrames(count, frame = {}) {
    return new Array(count).fill(null).map(() => ({'av': 0.5, 'f0': 120, ...frame}));
}


test('Bandpass, unity peak at the center frequency', () => {
    const sampleRate = 16000;
    const make = (frequency, bandwidth) => {
        const filter = new Bandpass();
        filter.set(frequency, bandwidth, sampleRate);
        return filter;
    };
    assert.ok(Math.abs(magnitude(make(2000, 400), 2000, sampleRate) - 1) < 1e-6);
    assert.ok(magnitude(make(2000, 400), 1000, sampleRate) < 0.3);
    assert.ok(magnitude(make(2000, 400), 4000, sampleRate) < 0.3);
    assert.ok(magnitude(make(2000, 400), 0, sampleRate) < 1e-9);

    // The center frequency is limited to 0.45 of the sample rate, the Q to at least 0.3
    assert.ok(Math.abs(magnitude(make(9000, 1000), 0.45 * sampleRate, sampleRate) - 1) < 1e-6);
    assert.ok(Math.abs(magnitude(make(1000, 100000), 1000, sampleRate) - 1) < 1e-6);
    const broad = make(1000, 100000);
    const broadQ = make(1000, 1000 / 0.3);
    assert.deepEqual([broad.b0, broad.a1, broad.a2], [broadQ.b0, broadQ.a1, broadQ.a2]);
});


test('HighPass, Butterworth response', () => {
    const sampleRate = 16000;

    // The default cutoff is 500 Hz at 44.1 kHz
    const defaultFilter = new HighPass();
    assert.ok(Math.abs(magnitude(defaultFilter, 500, 44100) - Math.SQRT1_2) < 1e-6);

    // -3 dB at the cutoff, zero at DC, unity at Nyquist, -12 dB/octave below the cutoff
    const filter = new HighPass();
    filter.set(1000, sampleRate);
    assert.ok(Math.abs(magnitude(filter, 1000, sampleRate) - Math.SQRT1_2) < 1e-6);
    filter.set(1000, sampleRate);
    assert.ok(magnitude(filter, 0, sampleRate) < 1e-9);
    filter.set(1000, sampleRate);
    assert.ok(Math.abs(magnitude(filter, 8000, sampleRate) - 1) < 1e-9);
    filter.set(1000, sampleRate);
    const octaveBelow = magnitude(filter, 250, sampleRate);
    assert.ok(Math.abs(20 * Math.log10(octaveBelow) + 24) < 0.5);
});


test('HighShelf, gains', () => {
    const sampleRate = 16000;
    for (const gainDb of [-12, 0, 6]) {
        // Unity at DC, the shelf gain at Nyquist, and half the gain (in dB) at the shelf frequency
        assert.ok(Math.abs(magnitude(new HighShelf(1500, gainDb, sampleRate), 0, sampleRate) - 1) < 1e-6);
        const high = magnitude(new HighShelf(1500, gainDb, sampleRate), 8000, sampleRate);
        assert.ok(Math.abs(20 * Math.log10(high) - gainDb) < 1e-6);
        const middle = magnitude(new HighShelf(1500, gainDb, sampleRate), 1500, sampleRate);
        assert.ok(Math.abs(20 * Math.log10(middle) - gainDb / 2) < 1e-6);
    }

    // 0 dB is the identity
    const flat = new HighShelf(1500, 0, sampleRate);
    for (const value of [1, -0.5, 0.25, 0, 3]) {
        assert.ok(Math.abs(flat.step(value) - value) < 1e-12);
    }
});


test('makeRandom', () => {
    // Repeatable per seed
    const first = makeRandom(42);
    const second = makeRandom(42);
    const other = makeRandom(43);
    const firstValues = Array.from({'length': 10}, () => first.uniform());
    assert.deepEqual(Array.from({'length': 10}, () => second.uniform()), firstValues);
    assert.notDeepEqual(Array.from({'length': 10}, () => other.uniform()), firstValues);

    // The mulberry32 known first value for seed 0
    assert.equal(makeRandom(0).uniform(), 1144304738 / 4294967296);

    // Uniform in [0, 1) with mean 1/2; Gaussian with mean 0 and unit variance
    const random = makeRandom(1);
    let uniformSum = 0;
    let gaussSum = 0;
    let gaussSquares = 0;
    const count = 20000;
    for (let ix = 0; ix < count; ix++) {
        const uniform = random.uniform();
        assert.ok(uniform >= 0 && uniform < 1);
        uniformSum += uniform;
        const gauss = random.gauss();
        assert.ok(Math.abs(gauss) <= 3 / Math.sqrt(0.5));
        gaussSum += gauss;
        gaussSquares += gauss * gauss;
    }
    assert.ok(Math.abs(uniformSum / count - 0.5) < 0.01);
    assert.ok(Math.abs(gaussSum / count) < 0.03);
    assert.ok(Math.abs(gaussSquares / count - 1) < 0.03);
});


test('defaultFrame', () => {
    assert.equal(defaultFrame.f0, 100);
    assert.equal(defaultFrame.av, 0);
    assert.equal(defaultFrame.fnp, defaultFrame.fnz);
    assert.equal(defaultFrame.bnp, defaultFrame.bnz);
});


test('synthesize, length and silence', () => {
    // (frames - 1) frame periods of samples
    assert.equal(synthesize([{}, {}]).length, 220);
    assert.equal(synthesize([{}]).length, 0);
    assert.equal(synthesize([{}, {}, {}], {'frameMs': 10, 'sampleRate': 16000}).length, 320);
    assert.equal(synthesize([{}, {}, {}], {'frameMs': 3, 'sampleRate': 11025}).length, Math.floor(2 * 3 * 11.025));

    // The default frames (no voicing, aspiration, or frication) are silent
    const silence = synthesize([{}, {}, {}]);
    assert.ok(silence instanceof Float32Array);
    assert.ok(silence.every((sample) => sample === 0));
});


test('synthesize, repeatable by seed', () => {
    const frames = vowelFrames(20, {'ah': 0.05, 'af': 0.02, 'ab': 1});
    const first = synthesize(frames, {'sampleRate': 16000});
    const second = synthesize(frames, {'sampleRate': 16000});
    const other = synthesize(frames, {'sampleRate': 16000, 'seed': 2});
    assert.deepEqual(first, second);
    assert.notDeepEqual(first, other);
    assert.ok(first.every((sample) => Number.isFinite(sample)));
});


test('synthesize, pitch and frame interpolation', () => {
    const sampleRate = 16000;

    // A steady 120 Hz voice, without jitter
    const steady = synthesize(vowelFrames(100), {sampleRate, 'jitter': 0, 'shimmer': 0});
    const steadyTrack = pitchTrack(steady, sampleRate).filter((frame) => frame.f0 > 0);
    assert.ok(steadyTrack.length > 20);
    for (const frame of steadyTrack) {
        assert.ok(Math.abs(frame.f0 - 120) < 3, `f0 ${frame.f0}`);
    }

    // A glide from 100 Hz to 200 Hz passes 150 Hz at its middle
    const glide = synthesize([{'av': 0.5, 'f0': 100}, {'av': 0.5, 'f0': 200}], {sampleRate, 'frameMs': 1000, 'jitter': 0, 'shimmer': 0});
    assert.equal(glide.length, sampleRate);
    const glideTrack = pitchTrack(glide, sampleRate);
    const middle = glideTrack.reduce((best, frame) => (Math.abs(frame.time - 0.5) < Math.abs(best.time - 0.5) ? frame : best));
    assert.ok(Math.abs(middle.f0 - 150) < 8, `f0 ${middle.f0}`);
    const early = glideTrack.find((frame) => frame.f0 > 0);
    const late = glideTrack.findLast((frame) => frame.f0 > 0);
    assert.ok(early.f0 < 120 && late.f0 > 180, `f0 ${early.f0} to ${late.f0}`);

    // The voicing amplitude interpolates - the output grows from silence
    const fadeIn = synthesize([{'av': 0, 'f0': 150}, {'av': 1, 'f0': 150}], {sampleRate, 'frameMs': 400, 'jitter': 0, 'shimmer': 0});
    const quarter = fadeIn.length / 4;
    assert.ok(rms(fadeIn.subarray(0, quarter)) < 0.5 * rms(fadeIn.subarray(3 * quarter)));
});


test('synthesize, formant peaks', () => {
    // A vowel's LPC formants are near its F1 and F2
    const sampleRate = 16000;
    const frames = vowelFrames(60, {'f1': 700, 'f2': 1200, 'f3': 2600, 'f4': 3500, 'f5': 4500, 'f6': 5500});
    const samples = synthesize(frames, {sampleRate, 'jitter': 0, 'shimmer': 0});
    const decimated = decimate(samples, 2);
    const frame = decimated.subarray(1000, 1400);
    const window = hann(frame.length);
    const found = formants(frame.map((value, ix) => value * window[ix]), sampleRate / 2, 10);
    for (const expected of [700, 1200, 2600]) {
        const near = found.some((formant) => Math.abs(formant.frequency - expected) < 0.1 * expected);
        assert.ok(near, `F ${expected} in ${JSON.stringify(found)}`);
    }
});


test('synthesize, streaming blocks equal the full output', () => {
    const frames = vowelFrames(30, {'ah': 0.05, 'af': 0.1, 'ap1': 1, 'ap2': 0.5, 'ab': 0.2, 'fnz': 400, 'ob1': 100});
    const full = synthesize(frames, {'sampleRate': 16000});
    for (const blockSize of [100, 512, 2048, full.length, 10000]) {
        const blocks = [];
        const streamed = synthesize(frames, {'sampleRate': 16000, blockSize, 'onBlock': (block) => {
            blocks.push(Float32Array.from(block));
        }});
        assert.deepEqual(streamed, full);
        assert.equal(blocks.length, Math.ceil(full.length / blockSize), `block size ${blockSize}`);
        for (const block of blocks.slice(0, -1)) {
            assert.equal(block.length, blockSize);
        }
        const joined = new Float32Array(full.length);
        let offset = 0;
        for (const block of blocks) {
            joined.set(block, offset);
            offset += block.length;
        }
        assert.equal(offset, full.length);
        assert.deepEqual(joined, full);
    }
});


test('synthesize, higher formants', () => {
    const sampleRate = 16000;
    const frames = vowelFrames(20);
    const options = {sampleRate, 'jitter': 0, 'shimmer': 0};
    const base = synthesize(frames, options);

    // Higher formants at or above 0.45 of the sample rate are ignored
    assert.deepEqual(synthesize(frames, {...options, 'higherFormants': [[7200, 300], [7500, 300]]}), base);

    // A higher formant within range changes the output
    const withHigher = synthesize(frames, {...options, 'higherFormants': [[6500, 300], [7500, 300]]});
    assert.notDeepEqual(withHigher, base);
    assert.equal(withHigher.length, base.length);
});


test('synthesize, source shelf', () => {
    const sampleRate = 16000;
    const frames = vowelFrames(40);
    const options = {sampleRate, 'jitter': 0, 'shimmer': 0};
    const base = synthesize(frames, options);

    // A 0 dB shelf is the default; a high shelf boost adds high-frequency energy
    assert.deepEqual(synthesize(frames, {...options, 'sourceShelfHz': 3000, 'sourceShelfDb': 0}), base);
    const boosted = synthesize(frames, {...options, 'sourceShelfDb': 12});

    // The boost raises the energy above 3 kHz, and leaves the low frequencies
    const highRms = (samples) => {
        const filter = new HighPass();
        filter.set(3000, sampleRate);
        return rms(samples.map((sample) => filter.step(sample)));
    };
    const lowRms = (samples) => {
        let low = 0;
        return rms(samples.map((sample) => {
            low = 0.95 * low + 0.05 * sample;
            return low;
        }));
    };
    assert.ok(highRms(boosted) > 1.5 * highRms(base));
    assert.ok(Math.abs(lowRms(boosted) / lowRms(base) - 1) < 0.05);

    // The breath noise's shelf - it brightens the noise while voicing, and leaves voiceless aspiration alone
    const breathy = vowelFrames(40, {'ah': 0.5});
    const noise = synthesize(breathy, options);
    assert.deepEqual(synthesize(breathy, {...options, 'noiseShelfDb': 0}), noise);
    const brightNoise = synthesize(breathy, {...options, 'noiseShelfDb': 20});
    assert.ok(highRms(brightNoise) > 1.5 * highRms(noise));
    assert.ok(Math.abs(lowRms(brightNoise) / lowRms(noise) - 1) < 0.1);
    const aspirated = vowelFrames(40, {'av': 0, 'ah': 0.5});
    assert.deepEqual(synthesize(aspirated, {...options, 'noiseShelfDb': 20}), synthesize(aspirated, options));
    assert.deepEqual(synthesize(frames, {...options, 'noiseShelfDb': 12}), base);
});


test('synthesize, aspiration and frication', () => {
    const sampleRate = 16000;
    const options = {sampleRate};

    // Aspiration alone is noise; more aspiration is louder
    const aspiration = synthesize(vowelFrames(20, {'av': 0, 'ah': 0.1}), options);
    const louder = synthesize(vowelFrames(20, {'av': 0, 'ah': 0.2}), options);
    assert.ok(rms(aspiration) > 0);
    assert.ok(Math.abs(rms(louder) / rms(aspiration) - 2) < 1e-3);

    // Glottal noise modulation changes voiced aspiration
    const voiced = vowelFrames(20, {'ah': 0.1});
    assert.notDeepEqual(synthesize(voiced, {...options, 'noiseModulation': 0}), synthesize(voiced, options));

    // Frication alone is high-passed noise, without low-frequency energy
    const frication = synthesize(vowelFrames(40, {'av': 0, 'af': 0.5, 'ab': 1, 'fhp': 2000}), options);
    assert.ok(rms(frication) > 0.01);
    let lowSum = 0;
    for (let ix = 0; ix + 16 <= frication.length; ix += 16) {
        lowSum += frication.subarray(ix, ix + 16).reduce((sum, sample) => sum + sample, 0) / 16;
    }
    assert.ok(Math.abs(lowSum / (frication.length / 16)) < 0.01 * rms(frication));

    // Zero frication amplitude with nonzero resonance amplitudes is silent
    const noFrication = synthesize(vowelFrames(10, {'av': 0, 'af': 0, 'ap1': 1, 'ap2': 1, 'ab': 1}), options);
    assert.ok(noFrication.every((sample) => sample === 0));
});


test('synthesize, jitter and shimmer', () => {
    const sampleRate = 16000;
    const frames = vowelFrames(40);
    const clean = synthesize(frames, {sampleRate, 'jitter': 0, 'shimmer': 0});
    const jittered = synthesize(frames, {sampleRate, 'shimmer': 0});
    const shimmered = synthesize(frames, {sampleRate, 'jitter': 0});
    assert.notDeepEqual(jittered, clean);
    assert.notDeepEqual(shimmered, clean);

    // The per-frame jit and shim multipliers of zero turn them off
    assert.deepEqual(synthesize(vowelFrames(40, {'jit': 0, 'shim': 0}), {sampleRate}), clean);
});


test('normalize', () => {
    const samples = Float32Array.from([0.1, -0.5, 0.25]);
    const normalized = normalize(samples);
    assert.ok(normalized instanceof Float32Array);
    assert.ok(Math.abs(normalized[1] + 0.9) < 1e-6);
    assert.ok(Math.abs(normalized[0] - 0.18) < 1e-6);
    assert.ok(Math.abs(normalized[2] - 0.45) < 1e-6);

    // A custom peak
    assert.ok(Math.abs(normalize(samples, 1)[1] + 1) < 1e-7);

    // Silence is unchanged
    assert.deepEqual(normalize(new Float32Array(4)), new Float32Array(4));
    assert.deepEqual(normalize(new Float32Array(0)), new Float32Array(0));

    // The input is not modified
    assert.deepEqual(samples, Float32Array.from([0.1, -0.5, 0.25]));
});
