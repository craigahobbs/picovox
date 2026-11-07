// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Voice and style transforms of parameter frames - pitch, pitch range, whisper, press, level, the source shelf, and
// frication. The neural voice's outputs are interpretable source-filter parameters, so these apply to its predictions as
// they do to any parameter frames. Frames are arrays (or typed arrays) in parameter order; each transform returns new
// frames.
//
//     const whispered = whisper(pitchScale(frames, 1.1), {'intensity': 1});

import {FLOOR_DB, LF_RD_MAX, LF_RD_MIN, NAMES} from './params.js';


const IX = Object.fromEntries(NAMES.map((name, ix) => [name, ix]));


// Scale the pitch
export function pitchScale(frames, scale) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.f0] *= scale;
        return out;
    });
}


// The voiced frames' floor (dB of voicing) - a pitch range's mean is of the frames above it
const voicedDb = -40;


// Widen (or narrow) the pitch range - every frame's pitch moved from the voiced frames' mean pitch (log scale) by scale
// (1.2 is a 20% wider range), over the frames given at once (a sentence's, say)
export function pitchRange(frames, scale) {
    const voiced = frames.filter((frame) => frame[IX.av] > voicedDb);
    const center = voiced.length === 0 ? 0 : voiced.reduce((sum, frame) => sum + Math.log(frame[IX.f0]), 0) / voiced.length;
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        if (voiced.length !== 0) {
            out[IX.f0] = Math.exp(center + scale * (Math.log(frame[IX.f0]) - center));
        }
        return out;
    });
}


// Whisper - remove the voicing (intensity 0 to 1) and add aspiration (WHISPER_BREATH_DB relative to the frame's voicing
// level at full intensity)
const WHISPER_BREATH_DB = -28;

export function whisper(frames, {intensity = 1} = {}) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        const av = frame[IX.av];
        out[IX.av] = intensity >= 1 ? FLOOR_DB : av + 20 * Math.log10(1 - intensity);
        const added = av > FLOOR_DB ? av + WHISPER_BREATH_DB + 20 * Math.log10(Math.max(intensity, 1e-9)) : FLOOR_DB;
        out[IX.ah] = 10 * Math.log10(10 ** (frame[IX.ah] / 10) + 10 ** (added / 10));
        out[IX.jitter] *= 1 - intensity;
        out[IX.shimmer] *= 1 - intensity;
        return out;
    });
}


// Press - vocal effort's voice quality without its loudness: a brighter source (PRESS_SHELF_PER_DB dB of shelf per dB) and
// a more pressed pulse (Rd scaled by PRESS_RD_PER_DB per dB, geometrically) - with level, a louder or quieter voice whose
// consonants keep pace
const PRESS_SHELF_PER_DB = 0.5;
const PRESS_RD_PER_DB = -0.02;

export function press(frames, db) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.shelf] += PRESS_SHELF_PER_DB * db;
        out[IX.rd] = Math.min(LF_RD_MAX, Math.max(LF_RD_MIN, frame[IX.rd] * (1 + PRESS_RD_PER_DB) ** db));
        return out;
    });
}


// The sources' amplitudes - voicing, aspiration, and frication (the frication filter's gains are relative to frication's)
const SOURCE_LEVELS = ['av', 'ah', 'af'];


// Level - every source louder or quieter by db (the levels above the floor - a silent source stays silent), the voice
// unchanged otherwise
export function level(frames, db) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        for (const name of SOURCE_LEVELS) {
            if (frame[IX[name]] > FLOOR_DB) {
                out[IX[name]] = Math.max(FLOOR_DB, frame[IX[name]] + db);
            }
        }
        return out;
    });
}


// Source shelf - the voice source's high shelf (its brightness above the shelf frequency) raised or lowered by db
export function shelf(frames, db) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.shelf] += db;
        return out;
    });
}


// Frication - the frication noise (the fricatives' s, sh, f, ...) louder or quieter by db (a silent frication stays silent) -
// sharper or softer consonants
export function frication(frames, db) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        if (frame[IX.af] > FLOOR_DB) {
            out[IX.af] = Math.max(FLOOR_DB, frame[IX.af] + db);
        }
        return out;
    });
}


// Apply a list of transforms - [{'pitchScale': 1.2}, {'pitchRange': 1.2}, {'whisper': 1}, {'press': 4}, {'level': -3},
// {'frication': 2}, {'shelf': -5}] (a pitch range is of the frames given at once)
export function applyTransforms(frames, transforms = []) {
    let out = frames;
    for (const transform of transforms) {
        if ('pitchScale' in transform) {
            out = pitchScale(out, transform.pitchScale);
        }
        if ('pitchRange' in transform) {
            out = pitchRange(out, transform.pitchRange);
        }
        if ('whisper' in transform) {
            out = whisper(out, {'intensity': transform.whisper});
        }
        if ('press' in transform) {
            out = press(out, transform.press);
        }
        if ('level' in transform) {
            out = level(out, transform.level);
        }
        if ('frication' in transform) {
            out = frication(out, transform.frication);
        }
        if ('shelf' in transform) {
            out = shelf(out, transform.shelf);
        }
    }
    return out;
}
