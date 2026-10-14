// Speech analysis - FFT, spectrograms, LPC, formant tracking, and pitch tracking


// In-place radix-2 FFT of real and imaginary arrays (length a power of 2)
export function fft(re, im) {
    const n = re.length;
    for (let ix = 1, jx = 0; ix < n; ix++) {
        let bit = n >> 1;
        for (; jx & bit; bit >>= 1) {
            jx ^= bit;
        }
        jx ^= bit;
        if (ix < jx) {
            [re[ix], re[jx]] = [re[jx], re[ix]];
            [im[ix], im[jx]] = [im[jx], im[ix]];
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const angle = -2 * Math.PI / len;
        const wRe = Math.cos(angle);
        const wIm = Math.sin(angle);
        for (let start = 0; start < n; start += len) {
            let curRe = 1;
            let curIm = 0;
            for (let kx = 0; kx < len / 2; kx++) {
                const aIx = start + kx;
                const bIx = aIx + len / 2;
                const bRe = re[bIx] * curRe - im[bIx] * curIm;
                const bIm = re[bIx] * curIm + im[bIx] * curRe;
                re[bIx] = re[aIx] - bRe;
                im[bIx] = im[aIx] - bIm;
                re[aIx] += bRe;
                im[aIx] += bIm;
                const nextRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = nextRe;
            }
        }
    }
}


// A Hann window
export function hann(length) {
    const window = new Float64Array(length);
    for (let ix = 0; ix < length; ix++) {
        window[ix] = 0.5 - 0.5 * Math.cos(2 * Math.PI * ix / (length - 1));
    }
    return window;
}


// A magnitude spectrogram (dB) - frames of {time, db: Float64Array of fftSize / 2 bins}
export function spectrogram(samples, sampleRate, {windowMs = 25, hopMs = 5, fftSize = 1024, preEmphasis = 0.97} = {}) {
    const windowLength = Math.round(windowMs * sampleRate / 1000);
    const hop = Math.round(hopMs * sampleRate / 1000);
    const window = hann(windowLength);
    const frames = [];
    for (let start = 0; start + windowLength <= samples.length; start += hop) {
        const re = new Float64Array(fftSize);
        const im = new Float64Array(fftSize);
        for (let ix = 0; ix < windowLength; ix++) {
            const sample = samples[start + ix] - (start + ix > 0 ? preEmphasis * samples[start + ix - 1] : 0);
            re[ix] = sample * window[ix];
        }
        fft(re, im);
        const db = new Float64Array(fftSize / 2);
        for (let ix = 0; ix < fftSize / 2; ix++) {
            db[ix] = 10 * Math.log10(re[ix] * re[ix] + im[ix] * im[ix] + 1e-12);
        }
        frames.push({'time': start / sampleRate, db});
    }
    return frames;
}


// LPC coefficients by autocorrelation and the Levinson-Durbin recursion - returns [1, a1, ..., ap] and the error
export function lpc(frame, order) {
    const r = new Float64Array(order + 1);
    for (let lag = 0; lag <= order; lag++) {
        let sum = 0;
        for (let ix = lag; ix < frame.length; ix++) {
            sum += frame[ix] * frame[ix - lag];
        }
        r[lag] = sum;
    }
    r[0] *= 1 + 1e-9;
    const a = new Float64Array(order + 1);
    a[0] = 1;
    let error = r[0];
    if (error <= 0) {
        return {a, error: 0};
    }
    for (let ix = 1; ix <= order; ix++) {
        let acc = r[ix];
        for (let jx = 1; jx < ix; jx++) {
            acc += a[jx] * r[ix - jx];
        }
        const k = -acc / error;
        const prev = a.slice();
        for (let jx = 1; jx < ix; jx++) {
            a[jx] = prev[jx] + k * prev[ix - jx];
        }
        a[ix] = k;
        error *= 1 - k * k;
    }
    return {a, error};
}


// The complex roots of a real polynomial (coefficients highest power first) by the Durand-Kerner method
export function polyRoots(coeffs) {
    const n = coeffs.length - 1;
    const lead = coeffs[0];
    const c = coeffs.map((value) => value / lead);
    let roots = [];
    for (let ix = 0; ix < n; ix++) {
        const angle = 2 * Math.PI * ix / n + 0.4;
        roots.push([0.9 * Math.cos(angle), 0.9 * Math.sin(angle)]);
    }
    for (let iteration = 0; iteration < 500; iteration++) {
        let delta = 0;
        const next = [];
        for (let ix = 0; ix < n; ix++) {
            const [zr, zi] = roots[ix];
            // Evaluate the polynomial at z
            let pr = 1;
            let pi = 0;
            for (let kx = 1; kx <= n; kx++) {
                const tr = pr * zr - pi * zi + c[kx];
                pi = pr * zi + pi * zr;
                pr = tr;
            }
            // Divide by the product of (z - other roots)
            let dr = 1;
            let di = 0;
            for (let jx = 0; jx < n; jx++) {
                if (jx !== ix) {
                    const ar = zr - roots[jx][0];
                    const ai = zi - roots[jx][1];
                    const tr = dr * ar - di * ai;
                    di = dr * ai + di * ar;
                    dr = tr;
                }
            }
            const denom = dr * dr + di * di || 1e-30;
            const qr = (pr * dr + pi * di) / denom;
            const qi = (pi * dr - pr * di) / denom;
            next.push([zr - qr, zi - qi]);
            delta = Math.max(delta, Math.abs(qr) + Math.abs(qi));
        }
        roots = next;
        if (delta < 1e-10) {
            break;
        }
    }
    return roots;
}


// Formant candidates of a frame - [{frequency, bandwidth}] sorted by frequency, from the LPC polynomial's roots
export function formants(frame, sampleRate, order) {
    const {a} = lpc(frame, order);
    const roots = polyRoots(Array.from(a));
    const result = [];
    for (const [re, im] of roots) {
        if (im <= 0) {
            continue;
        }
        const frequency = Math.atan2(im, re) * sampleRate / (2 * Math.PI);
        const bandwidth = -Math.log(Math.hypot(re, im)) * sampleRate / Math.PI;
        if (frequency > 90 && frequency < sampleRate / 2 - 100 && bandwidth < 600) {
            result.push({frequency, bandwidth});
        }
    }
    return result.sort((first, second) => first.frequency - second.frequency);
}


// Resample by an integer factor down, with a windowed-sinc low-pass filter
export function decimate(samples, factor, taps = 63) {
    const cutoff = 0.45 / factor;
    const kernel = new Float64Array(taps);
    const half = (taps - 1) / 2;
    let sum = 0;
    for (let ix = 0; ix < taps; ix++) {
        const x = ix - half;
        const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
        kernel[ix] = sinc * (0.54 - 0.46 * Math.cos(2 * Math.PI * ix / (taps - 1)));
        sum += kernel[ix];
    }
    const out = new Float32Array(Math.floor(samples.length / factor));
    for (let ox = 0; ox < out.length; ox++) {
        const center = ox * factor;
        let acc = 0;
        for (let ix = 0; ix < taps; ix++) {
            const sx = center + ix - half;
            if (sx >= 0 && sx < samples.length) {
                acc += samples[sx] * kernel[ix];
            }
        }
        out[ox] = acc / sum;
    }
    return out;
}


// Pitch (F0) track by normalized autocorrelation - [{time, f0 (0 if unvoiced), strength}]
export function pitchTrack(samples, sampleRate, {hopMs = 10, minF0 = 70, maxF0 = 400} = {}) {
    const hop = Math.round(hopMs * sampleRate / 1000);
    const windowLength = Math.round(3 * sampleRate / minF0);
    const minLag = Math.floor(sampleRate / maxF0);
    const maxLag = Math.ceil(sampleRate / minF0);
    const track = [];
    for (let start = 0; start + windowLength + maxLag < samples.length; start += hop) {
        let energy = 0;
        for (let ix = 0; ix < windowLength; ix++) {
            energy += samples[start + ix] * samples[start + ix];
        }
        let bestLag = 0;
        let best = 0;
        for (let lag = minLag; lag <= maxLag; lag++) {
            let acc = 0;
            let energyLag = 0;
            for (let ix = 0; ix < windowLength; ix++) {
                acc += samples[start + ix] * samples[start + ix + lag];
                energyLag += samples[start + ix + lag] * samples[start + ix + lag];
            }
            const value = acc / Math.sqrt(energy * energyLag + 1e-12);
            if (value > best) {
                best = value;
                bestLag = lag;
            }
        }
        const voiced = best > 0.5 && energy > 1e-4 * windowLength * 0.001;
        track.push({'time': (start + windowLength / 2) / sampleRate, 'f0': voiced ? sampleRate / bestLag : 0, 'strength': best});
    }
    return track;
}
