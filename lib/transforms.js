// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Voice and style transforms of parameter frames - nn/params.py's tract_scale, pitch_scale, whisper, and effort in
// JavaScript. The neural voice's outputs are interpretable source-filter parameters, so these apply to its predictions
// as they do to any parameter frames. Frames are arrays (or typed arrays) in parameter order; each transform returns
// new frames.
//
//     const whispered = whisper(pitchScale(frames, 1.1), {'intensity': 1});

import {FLOOR_DB, NAMES, TRACT_FREQUENCIES} from './params.js';


const IX = Object.fromEntries(NAMES.map((name, ix) => [name, ix]));


// Scale the vocal tract length - every resonance frequency (scale > 1 is a shorter tract)
export function tractScale(frames, scale) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        for (const name of TRACT_FREQUENCIES) {
            out[IX[name]] *= scale;
        }
        return out;
    });
}


// Scale the formant bandwidths (b1-b5) - below 1 sharpens the resonances, above 1 dulls them
export function bandwidthScale(frames, scale) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        for (const name of ['b1', 'b2', 'b3', 'b4', 'b5']) {
            out[IX[name]] *= scale;
        }
        return out;
    });
}


// Scale the pitch
export function pitchScale(frames, scale) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.f0] *= scale;
        return out;
    });
}


// Whisper - remove the voicing (intensity 0 to 1) and add aspiration (breathDb relative to the frame's voicing level at
// full intensity)
export function whisper(frames, {intensity = 1, breathDb = -28} = {}) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        const av = frame[IX.av];
        out[IX.av] = intensity >= 1 ? FLOOR_DB : av + 20 * Math.log10(1 - intensity);
        const added = av > FLOOR_DB ? av + breathDb + 20 * Math.log10(Math.max(intensity, 1e-9)) : FLOOR_DB;
        out[IX.ah] = 10 * Math.log10(10 ** (frame[IX.ah] / 10) + 10 ** (added / 10));
        out[IX.jitter] *= 1 - intensity;
        out[IX.shimmer] *= 1 - intensity;
        return out;
    });
}


// Vocal effort - louder voicing (db), a brighter source (shelfPerDb dB of shelf per dB), a more pressed pulse (Rd scaled
// by rdPerDb per dB, geometrically)
export function effort(frames, db, {shelfPerDb = 0.5, rdPerDb = -0.02} = {}) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.av] += db;
        out[IX.shelf] += shelfPerDb * db;
        out[IX.rd] = Math.min(2.7, Math.max(0.3, frame[IX.rd] * (1 + rdPerDb) ** db));
        return out;
    });
}


// Apply a list of transforms - [{'pitchScale': 1.2}, {'tractScale': 0.9}, {'whisper': 1}, {'effort': 6},
// {'bandwidthScale': 0.8}]
export function applyTransforms(frames, transforms = []) {
    let out = frames;
    for (const transform of transforms) {
        if ('bandwidthScale' in transform) {
            out = bandwidthScale(out, transform.bandwidthScale);
        }
        if ('pitchScale' in transform) {
            out = pitchScale(out, transform.pitchScale);
        }
        if ('tractScale' in transform) {
            out = tractScale(out, transform.tractScale);
        }
        if ('whisper' in transform) {
            out = whisper(out, {'intensity': transform.whisper});
        }
        if ('effort' in transform) {
            out = effort(out, transform.effort);
        }
    }
    return out;
}
