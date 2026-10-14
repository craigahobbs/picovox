// Spectral slices - the average spectrum (dB, 500 Hz bands to 8 kHz) of time ranges of WAV files
//
// node tools/slices.js file.wav start-end [start-end ...]   (seconds)

import {readWav} from '../lib/wav.js';
import {fft, hann} from '../lib/analysis.js';

const [path, ...ranges] = process.argv.slice(2);
const {samples, sampleRate} = readWav(path);
for (const range of ranges) {
    const [start, end] = range.split('-').map(Number);
    const n = 512;
    const window = hann(n);
    const power = new Float64Array(n / 2);
    let count = 0;
    for (let at = Math.floor(start * sampleRate); at + n <= Math.floor(end * sampleRate) + n / 2; at += n / 4) {
        const re = new Float64Array(n);
        const im = new Float64Array(n);
        for (let ix = 0; ix < n; ix++) {
            re[ix] = (samples[at + ix] || 0) * window[ix];
        }
        fft(re, im);
        for (let ix = 0; ix < n / 2; ix++) {
            power[ix] += re[ix] * re[ix] + im[ix] * im[ix];
        }
        count++;
    }
    const bands = [];
    for (let low = 0; low < 8000; low += 500) {
        let sum = 0;
        for (let ix = Math.floor(low / sampleRate * n); ix < Math.floor((low + 500) / sampleRate * n); ix++) {
            sum += power[ix];
        }
        bands.push(10 * Math.log10(sum / count + 1e-12));
    }
    console.log(range.padEnd(12), bands.map((db) => db.toFixed(0).padStart(4)).join(''));
}
