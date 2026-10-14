// Long-term average spectrum of the voiced frames of WAV files, in bands (dB relative to the 0-500 Hz band)
//
// node tools/ltas.js a.wav [b.wav ...]

import {readWav} from '../lib/wav.js';
import {spectrogram, pitchTrack} from '../lib/analysis.js';


export function ltas(path, {voiced = true} = {}) {
    const {samples, sampleRate} = readWav(path);
    const frames = spectrogram(samples, sampleRate, {'fftSize': 2048, 'windowMs': 30, 'hopMs': 10, 'preEmphasis': 0});
    const pitch = pitchTrack(samples, sampleRate, {'hopMs': 10});
    const bands = [[0, 500], [500, 1000], [1000, 2000], [2000, 3000], [3000, 4000], [4000, 6000], [6000, 8000]];
    const power = bands.map(() => 0);
    const powerVoiced = bands.map(() => 0);
    let countVoiced = 0;
    const binHz = sampleRate / 2048;
    let count = 0;
    frames.forEach((frame, ix) => {
        const point = pitch[Math.min(pitch.length - 1, ix)];
        const isVoiced = point && point.f0 > 0;
        const frameEnergy = frame.db.reduce((sum, value) => sum + Math.pow(10, value / 10), 0);
        if (isVoiced) {
            countVoiced++;
        }
        if (isVoiced === voiced && frameEnergy > 1e-3) {
            count++;
        }
        bands.forEach(([low, high], ixBand) => {
            for (let bin = Math.ceil(low / binHz); bin < Math.min(frame.db.length, high / binHz); bin++) {
                const value = Math.pow(10, frame.db[bin] / 10);
                if (isVoiced) {
                    powerVoiced[ixBand] += value;
                }
                if (isVoiced === voiced && frameEnergy > 1e-3) {
                    power[ixBand] += value;
                }
            }
        });
    });
    const reference = 10 * Math.log10(powerVoiced[0] / countVoiced + 1e-20);
    const db = power.map((value) => 10 * Math.log10(value / count + 1e-20));
    return {bands, 'db': db.map((value) => value - reference)};
}


if (process.argv[1] && process.argv[1].endsWith('ltas.js')) {
    const voiced = !process.argv.includes('--unvoiced');
    for (const path of process.argv.slice(2).filter((arg) => !arg.startsWith('--'))) {
        const {bands, db} = ltas(path, {voiced});
        console.log(path.padEnd(28), bands.map(([low, high], ix) => `${low / 1000}-${high / 1000}k:${db[ix].toFixed(1).padStart(6)}`).join(' '));
    }
}
