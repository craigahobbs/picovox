// The bridge from the rule voice - lib/speak.js's Klatt frames to the parameter set (nn/params.py), so the rule voice
// can be rendered by nn/js/render.js and nn/dsynth.py: for comparison with lib/klatt.js, and as the network's prior.
//
//     import {speak} from '../../lib/speak.js';
//     const result = speak('Hello.', 'male');
//     const frames = paramsFromSpeak(result);   // [{f0, av, ...}, ...] at 200 frames per second
//
// What the parameter set can't represent exactly: the F1 bandwidth's open-phase increase (ob1 - folded into b1,
// times 0.5 times the open fraction - the weight with the least spectral distance), the voice's higher formants (the
// parameter set's higher poles follow F6 - the male voice's exactly, the female's within ~0.5 kHz above 7 kHz), a
// shelf corner other than 2 kHz, and a noise modulation other than 0.7 (both the voices' defaults).

import {FLOOR_DB, FRAME_RATE} from './params.js';
import {defaultFrame} from '../../lib/klatt.js';
import {openFraction} from './lf.js';


// A linear amplitude in dB, at least a floor
function toDb(amplitude, floor) {
    return amplitude > 0 ? Math.max(floor, 20 * Math.log10(amplitude)) : floor;
}


// Convert one Klatt frame (lib/klatt.js) of a resolved voice (lib/voices.js) to parameter values
export function paramsFromKlattFrame(frameIn, voice, {openB1Weight = 0.5} = {}) {
    const frame = {...defaultFrame, ...frameIn};
    const ob1 = frame.ob1 * openB1Weight * openFraction(frame.rd);
    return {
        'f0': frame.f0, 'av': toDb(frame.av, FLOOR_DB), 'rd': frame.rd, 'shelf': voice.shelfDb,
        'jitter': voice.jitter * frame.jit, 'shimmer': voice.shimmer * frame.shim,
        'ah': toDb(frame.ah, FLOOR_DB), 'af': toDb(frame.af, FLOOR_DB),
        'f1': frame.f1, 'f2': frame.f2, 'f3': frame.f3, 'f4': frame.f4, 'f5': frame.f5, 'f6': frame.f6,
        'b1': frame.b1 + ob1, 'b2': frame.b2, 'b3': frame.b3, 'b4': frame.b4, 'b5': frame.b5, 'b6': frame.b6,
        'fnp': frame.fnp, 'bnp': frame.bnp, 'fnz': frame.fnz, 'bnz': frame.bnz,
        'fp1': frame.fp1, 'bp1': frame.bp1, 'ap1': toDb(frame.ap1, -60), 'fp2': frame.fp2, 'bp2': frame.bp2,
        'ap2': toDb(frame.ap2, -60), 'ab': toDb(frame.ab, -60), 'fhp': frame.fhp
    };
}


// Convert a speak() result's frames (lib/speak.js - 5 ms frames) to parameter frames
export function paramsFromSpeak(result, options = {}) {
    if (result.frames.length > 1 && Math.abs(result.samples.length / result.sampleRate * FRAME_RATE - (result.frames.length - 1)) > 1) {
        throw new Error('speak() frames are not at the parameter frame rate');
    }
    return result.frames.map((frame) => paramsFromKlattFrame(frame, result.voice, options));
}
