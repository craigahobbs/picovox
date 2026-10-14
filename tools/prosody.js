// Prosody statistics of WAV files - speaking rate, pauses, pitch range and declination, pitch-accent rate, and
// rhythm (syllable-nucleus intervals, phrase-final lengthening)
//
// node tools/prosody.js [-t "text"] [-v] a.wav [b.wav ...]   (the text gives the syllable count for the speaking rate)

import {readWav} from '../lib/wav.js';
import {decimate, pitchTrack} from '../lib/analysis.js';
import {textToPhrases} from '../lib/lexicon.js';


const hopMs = 10;


// The syllable count of text
function syllableCount(text) {
    return textToPhrases(text).reduce((sum, phrase) => sum + phrase.words.reduce((wordSum, word) => wordSum + word.syllables, 0), 0);
}


// A level envelope (dB) in 10 ms hops - of the signal band-passed to 300-3000 Hz (first-order sections)
function levelEnvelope(samples, sampleRate) {
    const hop = Math.round(hopMs * sampleRate / 1000);
    const window = 2 * hop;
    const band = new Float64Array(samples.length);
    const aHigh = Math.exp(-2 * Math.PI * 300 / sampleRate);
    const aLow = Math.exp(-2 * Math.PI * 3000 / sampleRate);
    let high = 0;
    let prev = 0;
    let low = 0;
    for (let ix = 0; ix < samples.length; ix++) {
        high = aHigh * (high + samples[ix] - prev);
        prev = samples[ix];
        low = (1 - aLow) * high + aLow * low;
        band[ix] = low;
    }
    const levels = [];
    for (let start = 0; start + window <= samples.length; start += hop) {
        let sum = 0;
        let sumBand = 0;
        for (let ix = start; ix < start + window; ix++) {
            sum += samples[ix] * samples[ix];
            sumBand += band[ix] * band[ix];
        }
        levels.push({'db': 10 * Math.log10(sum / window + 1e-12), 'band': 10 * Math.log10(sumBand / window + 1e-12)});
    }
    return levels;
}


// Mean and standard deviation
function meanStd(values) {
    const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
    const variance = values.reduce((sum, value) => sum + (value - mean) * (value - mean), 0) / Math.max(1, values.length);
    return [mean, Math.sqrt(variance)];
}


// Least-squares slope of y over x
function slope(xs, ys) {
    const [mx] = meanStd(xs);
    const [my] = meanStd(ys);
    let num = 0;
    let den = 0;
    for (let ix = 0; ix < xs.length; ix++) {
        num += (xs[ix] - mx) * (ys[ix] - my);
        den += (xs[ix] - mx) * (xs[ix] - mx);
    }
    return den === 0 ? 0 : num / den;
}


// Peaks of a series with a minimum prominence on both sides - [index]
function peaks(values, prominence) {
    const result = [];
    for (let ix = 1; ix < values.length - 1; ix++) {
        if (values[ix] === null || !(values[ix] >= values[ix - 1] && values[ix] > values[ix + 1])) {
            continue;
        }
        let leftMin = values[ix];
        for (let jx = ix - 1; jx >= 0 && values[jx] !== null && values[jx] <= values[ix]; jx--) {
            leftMin = Math.min(leftMin, values[jx]);
        }
        let rightMin = values[ix];
        for (let jx = ix + 1; jx < values.length && values[jx] !== null && values[jx] <= values[ix]; jx++) {
            rightMin = Math.min(rightMin, values[jx]);
        }
        if (values[ix] - leftMin >= prominence && values[ix] - rightMin >= prominence) {
            result.push(ix);
        }
    }
    return result;
}


// Analyze a WAV file
function analyze(path, syllables, verbose) {
    const wav = readWav(path);
    const factor = wav.sampleRate > 30000 ? 2 : 1;
    const samples = factor === 1 ? wav.samples : decimate(wav.samples, factor);
    const sampleRate = wav.sampleRate / factor;
    const levels = levelEnvelope(samples, sampleRate);
    const maxDb = Math.max(...levels.map((level) => level.db));

    // Speech and pauses - silence below max - 35 dB, pauses of 150 ms or more
    const silent = levels.map((level) => level.db < maxDb - 35);
    const first = silent.indexOf(false);
    const last = silent.lastIndexOf(false);
    const pauses = [];
    for (let ix = first; ix <= last; ix++) {
        if (silent[ix]) {
            let jx = ix;
            while (silent[jx]) {
                jx++;
            }
            if ((jx - ix) * hopMs >= 150) {
                pauses.push([ix, jx]);
            }
            ix = jx;
        }
    }
    const speechMs = (last - first + 1) * hopMs;
    const pauseMs = pauses.reduce((sum, [start, end]) => sum + (end - start) * hopMs, 0);
    const articulationMs = speechMs - pauseMs;

    // Breath groups between pauses
    const groups = [];
    let groupStart = first;
    for (const [start, end] of pauses) {
        groups.push([groupStart, start]);
        groupStart = end;
    }
    groups.push([groupStart, last + 1]);

    // Pitch - semitones re 100 Hz per 10 ms hop (null where unvoiced), octave errors removed against a running median
    const track = pitchTrack(samples, sampleRate, {hopMs, 'minF0': 60, 'maxF0': 400});
    const offset = Math.round((3 * sampleRate / 60) / 2 / (hopMs * sampleRate / 1000));
    const pitch = new Array(levels.length).fill(null);
    track.forEach((point, ix) => {
        const at = ix + offset;
        if (at < pitch.length && point.f0 > 0 && point.strength > 0.6 && !silent[at]) {
            pitch[at] = 12 * Math.log2(point.f0 / 100);
        }
    });
    const voicedValues = pitch.filter((value) => value !== null);
    const sorted = [...voicedValues].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    for (let ix = 0; ix < pitch.length; ix++) {
        if (pitch[ix] !== null && Math.abs(pitch[ix] - median) > 9) {
            pitch[ix] = null;
        }
    }
    const voiced = pitch.filter((value) => value !== null).sort((a, b) => a - b);
    const percentile = (fraction) => voiced[Math.floor(fraction * (voiced.length - 1))];
    const [, pitchStd] = meanStd(voiced);

    // Smoothed pitch (5-hop moving average over voiced, gaps under 120 ms bridged) for peaks and declination
    const smoothPitch = pitch.map((value, ix) => {
        const window = pitch.slice(Math.max(0, ix - 2), ix + 3).filter((other) => other !== null);
        return value === null ? null : window.reduce((sum, other) => sum + other, 0) / window.length;
    });
    const bridged = [...smoothPitch];
    for (let ix = 0; ix < bridged.length; ix++) {
        if (bridged[ix] === null && ix > 0 && bridged[ix - 1] !== null) {
            let jx = ix;
            while (jx < bridged.length && bridged[jx] === null) {
                jx++;
            }
            if (jx < bridged.length && jx - ix <= 12) {
                for (let kx = ix; kx < jx; kx++) {
                    bridged[kx] = bridged[ix - 1] + (bridged[jx] - bridged[ix - 1]) * (kx - ix + 1) / (jx - ix + 1);
                }
            }
            ix = jx;
        }
    }
    const pitchPeaks = peaks(bridged, 1.5);

    // Per-group declination (slope of the voiced pitch over time), start and end levels, and the final movement
    const slopes = [];
    const finals = [];
    const resets = [];
    let prevEnd = null;
    for (const [start, end] of groups) {
        const xs = [];
        const ys = [];
        for (let ix = start; ix < end; ix++) {
            if (smoothPitch[ix] !== null) {
                xs.push(ix * hopMs / 1000);
                ys.push(smoothPitch[ix]);
            }
        }
        if (xs.length > 20) {
            slopes.push(slope(xs, ys));
            const tail = ys.slice(-8);
            finals.push(tail[tail.length - 1] - Math.max(...ys.slice(-40)));
            const head = meanStd(ys.slice(0, 15))[0];
            if (prevEnd !== null) {
                resets.push(head - prevEnd);
            }
            prevEnd = meanStd(ys.slice(-15))[0];
        }
    }

    // Syllable nuclei - peaks of the band level with 4 dB prominence, at most 20 dB under the maximum
    const bandMax = Math.max(...levels.map((level) => level.band));
    const band = levels.map((level) => (level.band < bandMax - 25 ? null : level.band));
    const nuclei = peaks(band, 3).filter((ix) => !silent[ix]);
    const intervals = [];
    const finalIntervals = [];
    const levelSlopes = [];
    for (const [start, end] of groups) {
        const groupNuclei = nuclei.filter((ix) => ix >= start && ix < end);
        if (groupNuclei.length >= 4) {
            levelSlopes.push(slope(groupNuclei.map((ix) => ix * hopMs / 1000), groupNuclei.map((ix) => levels[ix].db)));
        }
        const groupIntervals = groupNuclei.slice(1).map((ix, kx) => (ix - groupNuclei[kx]) * hopMs);
        if (groupIntervals.length >= 2) {
            finalIntervals.push((end - groupNuclei[groupNuclei.length - 1]) * hopMs);
            intervals.push(...groupIntervals);
        }
    }
    let pvi = 0;
    for (let ix = 1; ix < intervals.length; ix++) {
        pvi += Math.abs(intervals[ix] - intervals[ix - 1]) / ((intervals[ix] + intervals[ix - 1]) / 2);
    }

    if (verbose) {
        console.log(`${path}: pauses ${pauses.map(([start, end]) => (end - start) * hopMs).join(' ')} ms; finals ${finals.map((value) => value.toFixed(1)).join(' ')} st; ` +
            `final intervals ${finalIntervals.map((value) => value.toFixed(0)).join(' ')} ms`);
    }
    return {
        'speech s': speechMs / 1000,
        'pauses': pauses.length,
        'pause ms': pauses.length === 0 ? 0 : pauseMs / pauses.length,
        'syl/s': syllables === null ? 0 : syllables / (speechMs / 1000),
        'artic syl/s': syllables === null ? 0 : syllables / (articulationMs / 1000),
        'median Hz': 100 * Math.pow(2, median / 12),
        '5-95 st': percentile(0.95) - percentile(0.05),
        'std st': pitchStd,
        'peaks/s': pitchPeaks.length / (articulationMs / 1000),
        'decl st/s': meanStd(slopes)[0],
        'final st': meanStd(finals)[0],
        'reset st': meanStd(resets)[0],
        'level dB/s': meanStd(levelSlopes)[0],
        'nuclei/s': nuclei.length / (articulationMs / 1000),
        'nPVI': 100 * pvi / Math.max(1, intervals.length - 1),
        'final ms': meanStd(finalIntervals)[0],
        'interval ms': meanStd(intervals)[0]
    };
}


const args = process.argv.slice(2);
let syllables = null;
let verbose = false;
const paths = [];
for (let ix = 0; ix < args.length; ix++) {
    if (args[ix] === '-t') {
        syllables = syllableCount(args[++ix]);
    } else if (args[ix] === '-v') {
        verbose = true;
    } else {
        paths.push(args[ix]);
    }
}
let header = false;
for (const path of paths) {
    const stats = analyze(path, syllables, verbose);
    if (!header) {
        console.log(''.padEnd(26) + Object.keys(stats).map((key) => key.padStart(12)).join(''));
        header = true;
    }
    console.log(path.slice(-26).padEnd(26) + Object.values(stats).map((value) => value.toFixed(value >= 100 ? 0 : 2).padStart(12)).join(''));
}
