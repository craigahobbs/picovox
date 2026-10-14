// Print the harmonic envelope (dB relative to the fundamental) of sustained synthetic vowels

import {fft, hann} from '../lib/analysis.js';
import process from 'node:process';
import {synthesize} from '../lib/klatt.js';

export function envelope(frame, options = {}, points = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000]) {
    const samples = synthesize(Array.from({'length': 200}, () => frame), {'jitter': 0, 'shimmer': 0, ...options});
    const fftSize = 16384;
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    const window = hann(fftSize);
    for (let ix = 0; ix < fftSize; ix++) {
        re[ix] = samples[20000 + ix] * window[ix];
    }
    fft(re, im);
    const db = (hz) => {
        const bin = Math.round(hz / 44100 * fftSize);
        let max = 0;
        for (let kx = bin - 40; kx <= bin + 40; kx++) {
            max = Math.max(max, re[kx] * re[kx] + im[kx] * im[kx]);
        }
        return 10 * Math.log10(max);
    };
    const reference = db(frame.f0);
    return points.map((hz) => [hz, db(hz) - reference]);
}

if (process.argv[1] && process.argv[1].endsWith('envelope.js')) {
    const higherFormants = [[6500, 500], [7500, 600], [8500, 700], [9500, 800], [10500, 900], [11500, 1000], [12500, 1100]];
    const base = {'f0': 110, 'av': 1, 'rd': 0.85, 'f1': 740, 'f2': 1180, 'f3': 2450, 'f4': 3300, 'f5': 4200, 'f6': 5500, 'b1': 100};
    const cases = [
        ['AA', {}, base],
        ['AA shelf 14', {'sourceShelfDb': 14}, base],
        ['neutral', {}, {...base, 'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': 3500, 'f5': 4500, 'f6': 5500}],
        ['neutral rd 0.3', {}, {...base, 'rd': 0.3, 'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': 3500, 'f5': 4500, 'f6': 5500}],
        ['neutral +higher', {higherFormants},
            {...base, 'rd': 1.0, 'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': 3500, 'f5': 4500, 'f6': 5500, 'b4': 200, 'b5': 300, 'b6': 400}],
        ['AA +higher', {higherFormants},
            {...base, 'rd': 1.0, 'b4': 200, 'b5': 300, 'b6': 400}]
    ];
    for (const [label, options, frame] of cases) {
        process.stdout.write(`${label.padEnd(16)} ${envelope(frame, options).map(([hz, db]) => `${hz}:${db.toFixed(0)}`).join(' ')}\n`);
    }
}
