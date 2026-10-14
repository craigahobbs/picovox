import {decimate, fft, formants, hann, lpc, pitchTrack, polyRoots, spectrogram} from '../lib/analysis.js';
import {strict as assert} from 'node:assert';
import {makeRandom} from '../lib/klatt.js';
import test from 'node:test';


// A direct (O(n^2)) discrete Fourier transform, for reference
function dft(re, im) {
    const size = re.length;
    const outRe = new Float64Array(size);
    const outIm = new Float64Array(size);
    for (let kx = 0; kx < size; kx++) {
        for (let ix = 0; ix < size; ix++) {
            const angle = -2 * Math.PI * kx * ix / size;
            outRe[kx] += re[ix] * Math.cos(angle) - im[ix] * Math.sin(angle);
            outIm[kx] += re[ix] * Math.sin(angle) + im[ix] * Math.cos(angle);
        }
    }
    return [outRe, outIm];
}


// A sinusoid
function sine(frequency, sampleRate, length, amplitude = 1) {
    const samples = new Float32Array(length);
    for (let ix = 0; ix < length; ix++) {
        samples[ix] = amplitude * Math.sin(2 * Math.PI * frequency * ix / sampleRate);
    }
    return samples;
}


// An all-pole filter's response to a pulse train - resonances [[frequency, bandwidth], ...]
function resonances(poles, sampleRate, length, f0) {
    const coeffs = [];
    for (const [frequency, bandwidth] of poles) {
        const radius = Math.exp(-Math.PI * bandwidth / sampleRate);
        coeffs.push([2 * radius * Math.cos(2 * Math.PI * frequency / sampleRate), -radius * radius]);
    }
    let signal = new Float64Array(length);
    const period = Math.round(sampleRate / f0);
    for (let ix = 0; ix < length; ix += period) {
        signal[ix] = 1;
    }
    for (const [feedback1, feedback2] of coeffs) {
        const filtered = new Float64Array(length);
        for (let ix = 0; ix < length; ix++) {
            filtered[ix] = signal[ix] + feedback1 * (ix > 0 ? filtered[ix - 1] : 0) + feedback2 * (ix > 1 ? filtered[ix - 2] : 0);
        }
        signal = filtered;
    }
    return signal;
}


test('fft, matches a direct DFT', () => {
    const random = makeRandom(5);
    for (const size of [1, 2, 4, 8, 32, 128]) {
        const re = new Float64Array(size);
        const im = new Float64Array(size);
        for (let ix = 0; ix < size; ix++) {
            re[ix] = random.gauss();
            im[ix] = random.gauss();
        }
        const [expectedRe, expectedIm] = dft(re, im);
        fft(re, im);
        for (let ix = 0; ix < size; ix++) {
            assert.ok(Math.abs(re[ix] - expectedRe[ix]) < 1e-9, `size ${size}, bin ${ix}`);
            assert.ok(Math.abs(im[ix] - expectedIm[ix]) < 1e-9, `size ${size}, bin ${ix}`);
        }
    }
});


test('fft, impulse and sinusoid', () => {
    // An impulse has a flat spectrum
    const re = new Float64Array(16);
    const im = new Float64Array(16);
    re[0] = 1;
    fft(re, im);
    assert.deepEqual(Array.from(re), new Array(16).fill(1));
    assert.ok(im.every((value) => Math.abs(value) < 1e-15));

    // A cosine at bin 3 has energy only in bins 3 and 13
    const cosRe = new Float64Array(16);
    const cosIm = new Float64Array(16);
    for (let ix = 0; ix < 16; ix++) {
        cosRe[ix] = Math.cos(2 * Math.PI * 3 * ix / 16);
    }
    fft(cosRe, cosIm);
    for (let ix = 0; ix < 16; ix++) {
        const magnitude = Math.hypot(cosRe[ix], cosIm[ix]);
        assert.ok(Math.abs(magnitude - (ix === 3 || ix === 13 ? 8 : 0)) < 1e-12, `bin ${ix}`);
    }
});


test('hann', () => {
    const window = hann(9);
    assert.equal(window.length, 9);
    assert.ok(Math.abs(window[0]) < 1e-15);
    assert.ok(Math.abs(window[8]) < 1e-15);
    assert.ok(Math.abs(window[4] - 1) < 1e-15);
    assert.ok(Math.abs(window[2] - 0.5) < 1e-15);
    // Symmetric
    for (let ix = 0; ix < 9; ix++) {
        assert.ok(Math.abs(window[ix] - window[8 - ix]) < 1e-15);
    }
});


test('spectrogram, sinusoid peak', () => {
    const sampleRate = 16000;
    const samples = sine(1000, sampleRate, 1600);
    const frames = spectrogram(samples, sampleRate);

    // 25 ms windows (400 samples) every 5 ms (80 samples) in 1600 samples
    assert.equal(frames.length, 16);
    assert.equal(frames[0].time, 0);
    assert.ok(Math.abs(frames[1].time - 0.005) < 1e-12);
    for (const frame of frames) {
        assert.equal(frame.db.length, 512);
        // The peak bin is at 1000 Hz (bin 1000 / (16000 / 1024) = 64)
        const peak = frame.db.indexOf(Math.max(...frame.db));
        assert.equal(peak, 64);
    }
});


test('spectrogram, options and silence', () => {
    const frames = spectrogram(new Float32Array(1000), 10000, {'windowMs': 10, 'hopMs': 10, 'fftSize': 128, 'preEmphasis': 0});
    assert.equal(frames.length, 10);
    for (const frame of frames) {
        assert.equal(frame.db.length, 64);
        // Silence is the -120 dB floor
        assert.ok(frame.db.every((value) => Math.abs(value + 120) < 1e-9));
    }

    // Too short for a window
    assert.deepEqual(spectrogram(new Float32Array(100), 10000), []);
});


test('lpc, recovers an autoregressive process', () => {
    // x[n] = 1.3 x[n-1] - 0.6 x[n-2] + noise, so A(z) = 1 - 1.3 z^-1 + 0.6 z^-2
    const random = makeRandom(3);
    const samples = new Float64Array(20000);
    for (let ix = 2; ix < samples.length; ix++) {
        samples[ix] = 1.3 * samples[ix - 1] - 0.6 * samples[ix - 2] + random.gauss();
    }
    const {'a': coefficients, error} = lpc(samples, 4);
    assert.equal(coefficients.length, 5);
    assert.equal(coefficients[0], 1);
    assert.ok(Math.abs(coefficients[1] + 1.3) < 0.03);
    assert.ok(Math.abs(coefficients[2] - 0.6) < 0.03);
    assert.ok(Math.abs(coefficients[3]) < 0.03);
    assert.ok(Math.abs(coefficients[4]) < 0.03);
    // The prediction error is the noise energy (unit variance)
    assert.ok(Math.abs(error / samples.length - 1) < 0.05);
});


test('lpc, silence', () => {
    const {'a': coefficients, error} = lpc(new Float32Array(64), 3);
    assert.deepEqual(Array.from(coefficients), [1, 0, 0, 0]);
    assert.equal(error, 0);
});


test('polyRoots, real and complex roots', () => {
    // 2 (z - 0.5)(z + 0.25) = 2z^2 - 0.5z - 0.25
    const realRoots = polyRoots([2, -0.5, -0.25]).sort((first, second) => first[0] - second[0]);
    assert.ok(Math.abs(realRoots[0][0] + 0.25) < 1e-10 && Math.abs(realRoots[0][1]) < 1e-10);
    assert.ok(Math.abs(realRoots[1][0] - 0.5) < 1e-10 && Math.abs(realRoots[1][1]) < 1e-10);

    // z^3 - 1 - the cube roots of unity
    const unityRoots = polyRoots([1, 0, 0, -1]);
    assert.equal(unityRoots.length, 3);
    for (const [re, im] of unityRoots) {
        assert.ok(Math.abs(Math.hypot(re, im) - 1) < 1e-10);
        const angle = Math.atan2(im, re) * 3 / (2 * Math.PI);
        assert.ok(Math.abs(angle - Math.round(angle)) < 1e-10);
    }

    // z^2 + 1 = 0 - the roots +/- i
    const imaginary = polyRoots([1, 0, 1]).sort((first, second) => first[1] - second[1]);
    assert.ok(Math.abs(imaginary[0][0]) < 1e-10 && Math.abs(imaginary[0][1] + 1) < 1e-10);
    assert.ok(Math.abs(imaginary[1][0]) < 1e-10 && Math.abs(imaginary[1][1] - 1) < 1e-10);
});


test('polyRoots, degenerate polynomials', () => {
    // A constant has no roots
    assert.deepEqual(polyRoots([3]), []);

    // A non-finite polynomial never converges - the iteration limit ends it
    const roots = polyRoots([1, Number.NaN, 1]);
    assert.equal(roots.length, 2);
    assert.ok(roots.every(([re, im]) => Number.isNaN(re) && Number.isNaN(im)));
});


test('formants, synthetic resonances', () => {
    const sampleRate = 10000;
    const signal = resonances([[500, 60], [1500, 90], [2500, 120]], sampleRate, 400, 100);
    const window = hann(signal.length);
    const frame = signal.map((value, ix) => value * window[ix]);
    const result = formants(frame, sampleRate, 8);

    // Each resonance is found near its frequency, with a positive bandwidth under the 600 Hz limit
    for (const expected of [500, 1500, 2500]) {
        const found = result.find((formant) => Math.abs(formant.frequency - expected) < 50);
        assert.ok(found, `formant ${expected} in ${JSON.stringify(result)}`);
        assert.ok(found.bandwidth > 0 && found.bandwidth < 600);
    }

    // Sorted by frequency, within (90, Nyquist - 100)
    for (let ix = 0; ix < result.length; ix++) {
        assert.ok(result[ix].frequency > 90 && result[ix].frequency < sampleRate / 2 - 100);
        if (ix > 0) {
            assert.ok(result[ix].frequency > result[ix - 1].frequency);
        }
    }
});


test('formants, broad resonances are rejected', () => {
    // A resonance far broader than 600 Hz is not a formant
    const sampleRate = 10000;
    const signal = resonances([[1000, 60], [3000, 2000]], sampleRate, 400, 100);
    const result = formants(signal, sampleRate, 4);
    assert.ok(result.some((formant) => Math.abs(formant.frequency - 1000) < 50));
    assert.ok(result.every((formant) => formant.bandwidth < 600));
    assert.ok(!result.some((formant) => Math.abs(formant.frequency - 3000) < 300));
});


test('decimate, length and DC', () => {
    const samples = new Float32Array(1000).fill(0.5);
    const out = decimate(samples, 4);
    assert.ok(out instanceof Float32Array);
    assert.equal(out.length, 250);
    // Unity gain at DC away from the edges, and less at the edge (the kernel hangs off the signal)
    for (let ix = 10; ix < 240; ix++) {
        assert.ok(Math.abs(out[ix] - 0.5) < 1e-6, `sample ${ix}`);
    }
    assert.ok(out[0] < 0.4);
    assert.equal(decimate(samples, 3).length, 333);
});


test('decimate, low-pass', () => {
    const sampleRate = 16000;

    // A low frequency passes, a frequency above the new Nyquist (2000 Hz) is removed
    const passed = decimate(sine(300, sampleRate, 4000), 4, 127);
    const stopped = decimate(sine(3500, sampleRate, 4000), 4, 127);
    const peak = (samples) => Math.max(...samples.subarray(50, samples.length - 50).map(Math.abs));
    assert.ok(Math.abs(peak(passed) - 1) < 0.02);
    assert.ok(peak(stopped) < 0.01);

    // The passed sinusoid is the decimated sinusoid
    const expected = sine(300, sampleRate / 4, passed.length);
    for (let ix = 50; ix < passed.length - 50; ix++) {
        assert.ok(Math.abs(passed[ix] - expected[ix]) < 0.02, `sample ${ix}`);
    }
});


test('pitchTrack, sinusoid', () => {
    const sampleRate = 8000;
    const samples = sine(200, sampleRate, 4000, 0.5);
    const track = pitchTrack(samples, sampleRate);
    // 3 / 70 s windows (343 samples), 10 ms (80 sample) hops, with the maximum lag (115) of look-ahead
    assert.equal(track.length, Math.ceil((4000 - 343 - 115) / 80));
    assert.ok(Math.abs(track[0].time - 171.5 / 8000) < 1e-12);
    for (const frame of track) {
        assert.ok(Math.abs(frame.f0 - 200) < 1, `f0 ${frame.f0}`);
        assert.ok(frame.strength > 0.99);
    }
});


test('pitchTrack, pulse train and options', () => {
    // A pulse train at 125 Hz
    const sampleRate = 8000;
    const samples = new Float32Array(4000);
    for (let ix = 0; ix < samples.length; ix += 64) {
        samples[ix] = 1;
    }
    const track = pitchTrack(samples, sampleRate, {'hopMs': 20, 'minF0': 100, 'maxF0': 300});
    assert.ok(track.length > 0);
    for (const frame of track) {
        assert.equal(frame.f0, 125);
    }
    assert.ok(Math.abs(track[1].time - track[0].time - 0.02) < 1e-12);
});


test('pitchTrack, unvoiced', () => {
    // Silence and noise are unvoiced
    const sampleRate = 8000;
    const silence = pitchTrack(new Float32Array(2000), sampleRate);
    assert.ok(silence.length > 0);
    assert.ok(silence.every((frame) => frame.f0 === 0 && frame.strength === 0));

    const random = makeRandom(7);
    const noise = new Float32Array(4000).map(() => random.gauss());
    const noiseTrack = pitchTrack(noise, sampleRate);
    assert.ok(noiseTrack.every((frame) => frame.f0 === 0 && frame.strength < 0.5));

    // A quiet sinusoid is periodic but below the energy threshold
    const quiet = pitchTrack(sine(200, sampleRate, 2000, 3e-4), sampleRate);
    assert.ok(quiet.every((frame) => frame.f0 === 0 && frame.strength > 0.99));
});
