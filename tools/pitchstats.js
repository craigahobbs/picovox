// Pitch statistics of WAV files - F0 range, variability (semitones), and the rate of pitch movement
//
// node tools/pitchstats.js a.wav [b.wav ...]

import {readWav} from '../lib/wav.js';
import {pitchTrack} from '../lib/analysis.js';

for (const path of process.argv.slice(2)) {
    const {samples, sampleRate} = readWav(path);
    const track = pitchTrack(samples, sampleRate, {'minF0': 60, 'maxF0': 400});
    const voiced = track.filter((point) => point.f0 > 0 && point.strength > 0.6);
    const semitones = voiced.map((point) => 12 * Math.log2(point.f0));
    const mean = semitones.reduce((sum, value) => sum + value, 0) / semitones.length;
    const std = Math.sqrt(semitones.reduce((sum, value) => sum + (value - mean) * (value - mean), 0) / semitones.length);
    const sorted = [...voiced.map((point) => point.f0)].sort((a, b) => a - b);
    const percentile = (fraction) => sorted[Math.floor(fraction * (sorted.length - 1))];
    // Mean absolute pitch velocity (semitones per second) between consecutive voiced frames
    let velocity = 0;
    let count = 0;
    for (let ix = 1; ix < track.length; ix++) {
        if (track[ix].f0 > 0 && track[ix - 1].f0 > 0) {
            velocity += Math.abs(12 * Math.log2(track[ix].f0 / track[ix - 1].f0)) / 0.01;
            count++;
        }
    }
    console.log(`${path.padEnd(30)} median ${percentile(0.5).toFixed(0)} Hz, 5-95% ${percentile(0.05).toFixed(0)}-${percentile(0.95).toFixed(0)} Hz, ` +
        `std ${std.toFixed(2)} st, velocity ${(velocity / count).toFixed(1)} st/s, voiced ${(voiced.length / 100).toFixed(1)} s of ${(samples.length / sampleRate).toFixed(1)} s`);
}
