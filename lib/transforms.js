// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Voice and style transforms of parameter frames - vocal tract length, pitch, pitch range, whisper, vocal effort, press,
// level, pulse softness, the source shelf, and frication. The neural voice's outputs are interpretable source-filter
// parameters, so these apply to its predictions as they do to any parameter frames. Frames are arrays (or typed arrays)
// in parameter order; each transform returns new frames.
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


// Press - vocal effort's voice quality without its loudness: a brighter source (shelfPerDb dB of shelf per dB) and a more
// pressed pulse (Rd scaled by rdPerDb per dB) - with level, a louder or quieter voice whose consonants keep pace
export function press(frames, db, {shelfPerDb = 0.5, rdPerDb = -0.02} = {}) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.shelf] += shelfPerDb * db;
        out[IX.rd] = Math.min(2.7, Math.max(0.3, frame[IX.rd] * (1 + rdPerDb) ** db));
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


// Pulse softness - the glottal pulse's Rd scaled (above 1 a softer pulse, its upper harmonics weaker), within Rd's range
export function rdScale(frames, scale) {
    return frames.map((frame) => {
        const out = Float64Array.from(frame);
        out[IX.rd] = Math.min(2.7, Math.max(0.3, frame[IX.rd] * scale));
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


// Apply a list of transforms - [{'pitchScale': 1.2}, {'pitchRange': 1.2}, {'tractScale': 0.9}, {'whisper': 1},
// {'effort': 6}, {'press': 4}, {'level': -3}, {'frication': 2}, {'rdScale': 1.2}, {'shelf': -5}] (a
// pitch range is of the frames given at once)
export function applyTransforms(frames, transforms = []) {
    let out = frames;
    for (const transform of transforms) {
        if ('pitchScale' in transform) {
            out = pitchScale(out, transform.pitchScale);
        }
        if ('pitchRange' in transform) {
            out = pitchRange(out, transform.pitchRange);
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
        if ('press' in transform) {
            out = press(out, transform.press);
        }
        if ('level' in transform) {
            out = level(out, transform.level);
        }
        if ('frication' in transform) {
            out = frication(out, transform.frication);
        }
        if ('rdScale' in transform) {
            out = rdScale(out, transform.rdScale);
        }
        if ('shelf' in transform) {
            out = shelf(out, transform.shelf);
        }
    }
    return out;
}
