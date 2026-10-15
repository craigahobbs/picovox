// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// Voice specs - a voice is data: a set of synthesis parameters, inherited from a base voice and overridden
//
// A spec is an object of metadata ('name', 'label', 'description', and 'base' - the name or spec it inherits from,
// the root voice if none) and parameter overrides. resolveVoice merges the root voice, the spec's base chain, the spec,
// the caller's overrides, and a speaking style into the complete parameter set that speak() synthesizes with.
//
//     const baritone = {'name': 'baritone', 'base': 'male', 'f0': 96, 'tractScale': 0.95, 'rate': 1.08};
//     speak('Call me Ishmael.', baritone);
//
// Parameters are physiological (the vocal tract length and vowel space, the F0 baseline and range, the voice quality),
// habitual (the speaking rate, articulation precision, pauses, intonation style), or acoustic calibration of the
// phoneme rules (burst, frication, and liquid levels). voiceParameters documents each - its meaning, units, typical
// range, and the root voice's value.

import {schwa, vowels} from './phonemes.js';


// The voice parameters - group, root (default) value, units, typical range (for an array, of each number in it), and
// meaning. The root voice is the reference adult male speaker of the phoneme tables.
export const voiceParameters = {
    // Pitch - the F0 baseline and range, and intonation style
    'f0': {
        'group': 'pitch', 'default': 108, 'units': 'Hz', 'range': [60, 280],
        'doc': 'The baseline F0 - the pitch at a phrase start, before accents and declination (adult men ~85-155, women ~165-255)'
    },
    'pitchScale': {
        'group': 'pitch', 'default': 1, 'units': 'ratio', 'range': [0.7, 1.4],
        'doc': 'A multiplier of f0 - speaker variation without editing f0'
    },
    'accent': {
        'group': 'pitch', 'default': 0.36, 'units': 'fraction of F0', 'range': [0.1, 0.7],
        'doc': 'The pitch accent height (randomized +/-25%) - the voice\'s pitch range and liveliness'
    },
    'exclamationAccent': {
        'group': 'pitch', 'default': 1.4, 'units': 'ratio', 'range': [1, 2],
        'doc': 'The accent height multiplier in an exclamation'
    },
    'questionRise': {
        'group': 'pitch', 'default': 1.5, 'units': 'fraction of f0', 'range': [1.2, 1.9],
        'doc': 'The F0 at the end of a yes/no question\'s final rise'
    },
    'endLow': {
        'group': 'pitch', 'default': 0.7, 'units': 'fraction of f0', 'range': [0.5, 0.9],
        'doc': 'The F0 at the end of a statement\'s final fall'
    },
    'pitchBase': {
        'group': 'pitch', 'default': 1, 'units': 'fraction of f0', 'range': [0.8, 1.2],
        'doc': 'The baseline at a phrase start (the declining baseline the accents rise from)'
    },
    'declination': {
        'group': 'pitch', 'default': 0.1, 'units': 'fraction of f0 per second', 'range': [0, 0.3],
        'doc': 'The baseline declination rate through a phrase'
    },
    'declinationMax': {
        'group': 'pitch', 'default': 0.18, 'units': 'fraction of f0', 'range': [0, 0.4],
        'doc': 'The maximum baseline declination'
    },
    'declinationReset': {
        'group': 'pitch', 'default': 0.5, 'units': 'fraction', 'range': [0, 1],
        'doc': 'How much of the declination is reset at an intermediate phrase'
    },
    'downstep': {
        'group': 'pitch', 'default': 0.85, 'units': 'ratio per accent', 'range': [0.6, 1],
        'doc': 'The downstep of each accent after the first in an intermediate phrase'
    },
    'downstepFloor': {
        'group': 'pitch', 'default': 0.6, 'units': 'ratio', 'range': [0.3, 1],
        'doc': 'The lowest downstep'
    },
    'clashAccent': {
        'group': 'pitch', 'default': 0.6, 'units': 'ratio', 'range': [0, 1],
        'doc': 'The strength of an accent weakened by a clash with an adjacent accent'
    },
    'nuclearFloor': {
        'group': 'pitch', 'default': 0.75, 'units': 'fraction of accent', 'range': [0, 1.2],
        'doc': 'The minimum height of a statement\'s nuclear (last) accent'
    },
    'minorHold': {
        'group': 'pitch', 'default': 0.4, 'units': 'fraction of the accent height', 'range': [0, 1],
        'doc': 'The pitch held to an intermediate phrase end (a high phrase accent)'
    },
    'continuationLow': {
        'group': 'pitch', 'default': 0.8, 'units': 'fraction of the baseline', 'range': [0.6, 1],
        'doc': 'The low of a continuation (comma) fall'
    },
    'continuationHigh': {
        'group': 'pitch', 'default': 0.92, 'units': 'fraction of the baseline', 'range': [0.7, 1.2],
        'doc': 'The end of a continuation\'s slight rise'
    },
    'accentBridge': {
        'group': 'pitch', 'default': 0.6, 'units': 'fraction of the lower accent', 'range': [0, 1],
        'doc': 'The bridging of close accents - the valley between them held up (0 - a full dip)'
    },
    'pitchDrift': {
        'group': 'pitch', 'default': 0.15, 'units': 'semitones RMS', 'range': [0, 1],
        'doc': 'The slow, aperiodic F0 wander - low-passed noise (pitchDriftHz)'
    },
    'pitchDriftHz': {
        'group': 'pitch', 'default': 0.8, 'units': 'Hz', 'range': [0.2, 3],
        'doc': 'The corner frequency of the F0 wander'
    },

    // Source - the glottal source and voice quality
    'rd': {
        'group': 'source', 'default': 0.9, 'units': 'Rd', 'range': [0.3, 2.7],
        'doc': 'The LF glottal pulse shape - low is pressed and bright, high is lax and breathy (modal ~0.8-1.4)'
    },
    'rdDrift': {
        'group': 'source', 'default': 0.06, 'units': 'fraction RMS', 'range': [0, 0.15],
        'doc': 'The slow, aperiodic drift of rd - low-passed noise (rdDriftHz)'
    },
    'rdDriftHz': {
        'group': 'source', 'default': 0.6, 'units': 'Hz', 'range': [0.1, 3],
        'doc': 'The corner frequency of the rd drift'
    },
    'flutter': {
        'group': 'source', 'default': 0.1, 'units': 'semitones RMS', 'range': [0, 0.4],
        'doc': 'F0 flutter - the fast, aperiodic F0 perturbation of a natural voice (low-passed noise, flutterHz), between ' +
            'the slow wander and the cycle-to-cycle jitter'
    },
    'flutterHz': {
        'group': 'source', 'default': 8, 'units': 'Hz', 'range': [3, 20],
        'doc': 'The corner frequency of the F0 flutter'
    },
    'breathDb': {
        'group': 'source', 'default': -38, 'units': 'dB re a stressed vowel', 'range': [-55, -15],
        'doc': 'The aspiration noise mixed with voicing - breathiness'
    },
    'noiseModulation': {
        'group': 'source', 'default': 0.7, 'units': 'fraction', 'range': [0, 1],
        'doc': 'How much the aspiration noise pulses with the glottal cycle'
    },
    'jitter': {
        'group': 'source', 'default': 0.011, 'units': 'fraction of the period', 'range': [0, 0.02],
        'doc': 'Random cycle-to-cycle period variation'
    },
    'shimmer': {
        'group': 'source', 'default': 0.04, 'units': 'fraction of the amplitude', 'range': [0, 0.1],
        'doc': 'Random cycle-to-cycle amplitude variation'
    },
    'creak': {
        'group': 'source', 'default': 0.4, 'units': '0-1', 'range': [0, 1],
        'doc': 'Creak (vocal fry) at the end of a statement - irregular, low pulses'
    },
    'glottalAttack': {
        'group': 'source', 'default': true, 'units': 'boolean', 'range': null,
        'doc': 'A vowel after a pause starts with a glottal attack (an abrupt onset with a few irregular pulses)'
    },
    'shelfHz': {
        'group': 'source', 'default': 2000, 'units': 'Hz', 'range': [1000, 4000],
        'doc': 'The voicing\'s high-shelf (spectral tilt correction) corner frequency'
    },
    'shelfDb': {
        'group': 'source', 'default': 14, 'units': 'dB', 'range': [-6, 20],
        'doc': 'The voicing\'s high-shelf gain - brightness (vocal effort)'
    },
    'noiseShelfDb': {
        'group': 'source', 'default': 16, 'units': 'dB', 'range': [-6, 20],
        'doc': 'The breath noise\'s high-shelf gain (at shelfHz) while voicing - the glottal turbulence above the corner, ' +
            'which masks the upper harmonics (voiceless aspiration keeps its spectrum)'
    },
    'openB1': {
        'group': 'source', 'default': 60, 'units': 'Hz', 'range': [0, 150],
        'doc': 'The F1 bandwidth increase while the glottis is open (subglottal coupling)'
    },
    'voicing': {
        'group': 'source', 'default': 1, 'units': 'ratio', 'range': [0, 1],
        'doc': 'A multiplier of every frame\'s voicing amplitude - 0 whispers (with breathDb raised)'
    },
    'vowelDecay': {
        'group': 'source', 'default': 0.2, 'units': 'fraction', 'range': [0, 0.5],
        'doc': 'The loudness decay through a vowel after its early peak'
    },
    'f0Loudness': {
        'group': 'source', 'default': 0.4, 'units': 'dB per semitone', 'range': [0, 1.5],
        'doc': 'The sources\' level rise with the intonation\'s F0 above the declining baseline - accented syllables are louder'
    },
    'levelDeclination': {
        'group': 'source', 'default': 0.08, 'units': 'fraction per second', 'range': [0, 0.3],
        'doc': 'The voicing\'s weakening through a phrase as the subglottal pressure falls'
    },
    'levelDeclinationMax': {
        'group': 'source', 'default': 0.2, 'units': 'fraction', 'range': [0, 0.5],
        'doc': 'The maximum loudness declination'
    },

    // Tract - the vocal tract's resonances
    'vowelBlend': {
        'group': 'tract', 'default': 0, 'units': '0-1', 'range': [0, 1],
        'doc': 'The vowel formant targets - 0 the men\'s averages, 1 the women\'s, between a blend (the vowel space shape)'
    },
    'tractScale': {
        'group': 'tract', 'default': 1, 'units': 'ratio', 'range': [0.85, 1.2],
        'doc': 'Scales every resonance (formants, nasal and lateral poles and zeros, frication) - the inverse of the ' +
            'vocal tract length relative to the voice\'s tables (0.95 is a 5% longer tract)'
    },
    'formantScale': {
        'group': 'tract', 'default': [1, 1, 1], 'units': 'ratio [F1, F2, F3]', 'range': [0.85, 1.35],
        'doc': 'Scales the consonant loci (male reference values) to the voice\'s vowel space'
    },
    'scaleNasalF1': {
        'group': 'tract', 'default': false, 'units': 'boolean', 'range': null,
        'doc': 'Scale a nasal\'s F1 locus by formantScale - false keeps it just below the nasal zero, which cancels it ' +
            '(a higher F1 escapes the zero and the murmur turns vowel-like)'
    },
    'f4': {
        'group': 'tract', 'default': 3450, 'units': 'Hz', 'range': [3000, 4600],
        'doc': 'The fixed fourth formant'
    },
    'f5': {
        'group': 'tract', 'default': 4200, 'units': 'Hz', 'range': [3800, 5500],
        'doc': 'The fixed fifth formant'
    },
    'f6': {
        'group': 'tract', 'default': 5500, 'units': 'Hz', 'range': [4800, 6800],
        'doc': 'The fixed sixth formant'
    },
    'higherFormants': {
        'group': 'tract', 'default': [[6500, 500], [7500, 600], [8500, 700], [9500, 800], [10500, 900], [11500, 1000]],
        'units': '[[Hz, Hz bandwidth], ...]', 'range': [300, 13000],
        'doc': 'The fixed resonances above F6 (a real tract resonates every ~1 kHz) - they keep the high spectrum from falling away'
    },
    'b1': {
        'group': 'tract', 'default': 75, 'units': 'Hz', 'range': [40, 150],
        'doc': 'The F1 bandwidth of vowels'
    },
    'coronalFronting': {
        'group': 'tract', 'default': 0.4, 'units': 'fraction', 'range': [0, 1],
        'doc': 'The fronting of a back rounded vowel (UW; UH and OW less) after a coronal consonant - its F2 raised this ' +
            'fraction toward IY\'s at the onset (American "two", "new", "so")'
    },
    'coronalFrontingEnd': {
        'group': 'tract', 'default': 0.1, 'units': 'fraction', 'range': [0, 1],
        'doc': 'The fraction of that fronting that remains at the vowel\'s end'
    },
    'formantWobble': {
        'group': 'tract', 'default': 0.005, 'units': 'fraction RMS', 'range': [0, 0.02],
        'doc': 'Small, aperiodic articulatory fluctuations of F1-F3 - low-passed noise (formantWobbleHz), independent per formant'
    },
    'formantWobbleHz': {
        'group': 'tract', 'default': 2.5, 'units': 'Hz', 'range': [0.5, 8],
        'doc': 'The corner frequency of the formant fluctuations'
    },
    'nonSibilantScale': {
        'group': 'tract', 'default': 1, 'units': 'ratio', 'range': [0.85, 1.35],
        'doc': 'Scales the non-sibilant (F, V, TH, DH) frication spectra - flat, with no front cavity to resonate'
    },
    'noiseScale': {
        'group': 'tract', 'default': 1, 'units': 'ratio', 'range': [0.85, 1.35],
        'doc': 'Scales the frication and burst noise spectra (male reference values) - the front cavity size'
    },
    'nasalPole': {
        'group': 'tract', 'default': 270, 'units': 'Hz', 'range': [200, 350],
        'doc': 'The nasal murmur\'s pole'
    },
    'nasalZero': {
        'group': 'tract', 'default': 450, 'units': 'Hz', 'range': [350, 600],
        'doc': 'The nasal murmur\'s zero - it cancels F1'
    },
    'nasalizationZero': {
        'group': 'tract', 'default': 360, 'units': 'Hz', 'range': [300, 500],
        'doc': 'The nasal zero of a vowel nasalized next to a nasal'
    },
    'lateralZero': {
        'group': 'tract', 'default': 2200, 'units': 'Hz (0 - none)', 'range': [0, 3000],
        'doc': 'The lateral (L) side branch\'s antiformant, between F2 and F3 (scaled by formantScale F3)'
    },
    'lateralPole': {
        'group': 'tract', 'default': 3300, 'units': 'Hz', 'range': [2500, 4000],
        'doc': 'The lateral side branch\'s extra pole (scaled by formantScale F3)'
    },
    'lateralBw': {
        'group': 'tract', 'default': 450, 'units': 'Hz', 'range': [100, 800],
        'doc': 'The bandwidth of the lateral pole and zero'
    },
    'lateralB2': {
        'group': 'tract', 'default': 100, 'units': 'Hz', 'range': [60, 300],
        'doc': 'The F2 bandwidth of a lateral'
    },

    // Articulation - precision, coarticulation, and the consonant levels
    'reduction': {
        'group': 'articulation', 'default': 1, 'units': 'ratio', 'range': [0, 2],
        'doc': 'Unstressed vowel reduction toward schwa - 0 none (careful speech), above 1 more (casual)'
    },
    'transitionScale': {
        'group': 'articulation', 'default': 1, 'units': 'ratio', 'range': [0.5, 2],
        'doc': 'Scales the consonant-vowel formant transition durations - above 1 slower, more coarticulated'
    },
    'burstDb': {
        'group': 'articulation', 'default': 18, 'units': 'dB', 'range': [6, 30],
        'doc': 'The stop release burst level, relative to the place\'s noise spectrum level'
    },
    'aspirationDb': {
        'group': 'articulation', 'default': -20, 'units': 'dB re a stressed vowel', 'range': [-35, -10],
        'doc': 'The aspiration level of a voiceless stop\'s voice onset time'
    },
    'hhDb': {
        'group': 'articulation', 'default': -24, 'units': 'dB re a stressed vowel', 'range': [-35, -12],
        'doc': 'The aspiration level of HH'
    },
    'fricVoicing': {
        'group': 'articulation', 'default': 0.3, 'units': 'amplitude', 'range': [0.1, 0.6],
        'doc': 'The voicing amplitude of a voiced fricative'
    },
    'fricVoiceBar': {
        'group': 'articulation', 'default': [300, 500], 'units': '[B2 Hz, B3 Hz]', 'range': [90, 1200],
        'doc': 'The damped F2 and F3 bandwidths of a voiced fricative - its voicing is mostly a low voice bar'
    },
    'fricDevoice': {
        'group': 'articulation', 'default': 0.5, 'units': 'fraction', 'range': [0, 1],
        'doc': 'The devoicing of a voiced fricative through its end before a pause'
    },
    'finalFricDb': {
        'group': 'articulation', 'default': -5, 'units': 'dB', 'range': [-15, 0],
        'doc': 'The level of a fricative before a pause'
    },
    'fricRiseMs': {
        'group': 'articulation', 'default': 40, 'units': 'ms', 'range': [0, 80],
        'doc': 'The gradual rise of a voiceless fricative after a pause (an abrupt rise is the affricate cue)'
    },
    'fricRiseDb': {
        'group': 'articulation', 'default': -15, 'units': 'dB', 'range': [-30, 0],
        'doc': 'The starting level of that rise'
    },
    'liquidAv': {
        'group': 'articulation', 'default': 0.6, 'units': 'amplitude', 'range': [0.3, 1],
        'doc': 'The voicing of a prevocalic liquid or glide (a constriction, quieter than the vowel)'
    },
    'codaLiquidAv': {
        'group': 'articulation', 'default': 0.82, 'units': 'amplitude', 'range': [0.4, 1],
        'doc': 'The voicing of a postvocalic liquid or glide (part of the nucleus)'
    },
    'clusterVot': {
        'group': 'articulation', 'default': 60, 'units': 'ms', 'range': [30, 100],
        'doc': 'The voice onset time of a voiceless stop before a liquid or glide in its word (devoicing it)'
    },
    'clusterLiquidScale': {
        'group': 'articulation', 'default': 0.5, 'units': 'ratio', 'range': [0.2, 1],
        'doc': 'The duration of the liquid or glide in such a cluster'
    },
    'clusterMoveMs': {
        'group': 'articulation', 'default': 20, 'units': 'ms', 'range': [5, 60],
        'doc': 'The formant movement time from the stop locus to the liquid or glide in a cluster'
    },
    'trDb': {
        'group': 'articulation', 'default': -18, 'units': 'dB re a stressed vowel', 'range': [-30, -8],
        'doc': 'The postalveolar frication of an alveolar stop before R (affricated "tr")'
    },

    // Timing - the speaking rate, pauses, and lengthening
    'rate': {
        'group': 'timing', 'default': 1.15, 'units': 'ratio', 'range': [0.7, 1.6],
        'doc': 'The speaking rate - Klatt\'s durations are divided by it (and by the rate option)'
    },
    'commaPause': {
        'group': 'timing', 'default': 300, 'units': 'ms', 'range': [80, 800],
        'doc': 'The pause after a comma between long phrases (commaPauseSyllables in all)'
    },
    'commaPauseMin': {
        'group': 'timing', 'default': 160, 'units': 'ms', 'range': [0, 400],
        'doc': 'The pause after a comma between the shortest phrases - it grows with their syllables to commaPause'
    },
    'commaPauseSyllables': {
        'group': 'timing', 'default': 34, 'units': 'syllables', 'range': [1, 80],
        'doc': 'The syllables of the phrases before and after a comma at which its pause is the full commaPause'
    },
    'majorPause': {
        'group': 'timing', 'default': 450, 'units': 'ms', 'range': [80, 1000],
        'doc': 'The pause after a semicolon or colon'
    },
    'sentencePause': {
        'group': 'timing', 'default': 480, 'units': 'ms', 'range': [150, 1200],
        'doc': 'The pause after a sentence'
    },
    'minorPause': {
        'group': 'timing', 'default': 0, 'units': 'ms', 'range': [0, 400],
        'doc': 'The pause at an intermediate phrase break (0 for none)'
    },
    'leadingPause': {
        'group': 'timing', 'default': 180, 'units': 'ms', 'range': [0, 1000],
        'doc': 'The silence before the speech'
    },
    'finalPause': {
        'group': 'timing', 'default': 420, 'units': 'ms', 'range': [0, 2000],
        'doc': 'The silence after the speech (in the player, with the next sentence\'s trimmed leading silence, the sentence pause)'
    },
    'finalLengthen': {
        'group': 'timing', 'default': 1.5, 'units': 'ratio', 'range': [1, 2.5],
        'doc': 'The phrase-final vowel lengthening'
    },
    'unstressedMinimum': {
        'group': 'timing', 'default': 0.9, 'units': 'ratio', 'range': [0.3, 1],
        'doc': 'The minimum duration of an unstressed vowel, a fraction of the stressed minimum'
    },
    'minorLengthen': {
        'group': 'timing', 'default': 1, 'units': 'ratio', 'range': [0.6, 2],
        'doc': 'The intermediate-phrase-final vowel lengthening (other vowels are 0.6)'
    },
    'minorSyllables': {
        'group': 'timing', 'default': 12, 'units': 'syllables', 'range': [6, 30],
        'doc': 'The phrase length above which an intermediate phrase break is inserted'
    }
};


// The root voice - every parameter at its default (the reference adult male speaker)
export const baseVoice = Object.fromEntries(Object.entries(voiceParameters).map(([key, parameter]) => [key, parameter.default]));


// The spec metadata keys - not synthesis parameters
export const voiceMetadata = ['name', 'label', 'description', 'base'];


// The voices
export const voices = {
    'male': {
        'name': 'male', 'label': 'Male', 'description': 'An adult American man - a baritone with a little phrase-final creak',
        'vowelBlend': 0, 'f0': 108, 'accent': 0.36, 'endLow': 0.7, 'rd': 0.9, 'breathDb': -38, 'jitter': 0.011, 'shimmer': 0.04,
        'creak': 0.4, 'shelfDb': 14, 'noiseShelfDb': 16, 'f0Loudness': 0.4, 'openB1': 60, 'b1': 75, 'f4': 3450, 'f5': 4200, 'f6': 5500,
        'higherFormants': [[6500, 500], [7500, 600], [8500, 700], [9500, 800], [10500, 900], [11500, 1000]],
        'formantScale': [1, 1, 1], 'noiseScale': 1, 'burstDb': 18, 'aspirationDb': -20, 'rate': 1.15
    },
    'female': {
        'name': 'female', 'label': 'Female', 'description': 'An adult American woman - a mezzo with a lighter, breathier voice',
        'vowelBlend': 1, 'f0': 172, 'accent': 0.3, 'endLow': 0.6, 'rd': 1.5, 'breathDb': -30, 'jitter': 0.005, 'shimmer': 0.035,
        'creak': 0, 'shelfDb': 7, 'noiseShelfDb': 6, 'f0Loudness': 0.3, 'openB1': 80, 'b1': 90, 'f4': 4100, 'f5': 4900, 'f6': 6000,
        'higherFormants': [[7200, 600], [8400, 700], [9600, 800], [10800, 900], [12000, 1000]],
        'formantScale': [1.12, 1.16, 1.15], 'noiseScale': 1.2, 'burstDb': 16, 'aspirationDb': -18, 'rate': 1.15
    }
};


// The spec chain of a voice - [root-most base spec, ..., the spec]
function specChain(voice) {
    const chain = [];
    let spec = voice;
    while (spec !== undefined && spec !== null) {
        if (typeof spec === 'string') {
            if (!(spec in voices)) {
                throw new Error(`Unknown voice "${spec}"`);
            }
            spec = voices[spec];
        }
        if (chain.includes(spec)) {
            throw new Error(`Voice base cycle at "${spec.name}"`);
        }
        chain.unshift(spec);
        spec = spec.base;
    }
    return chain;
}


// Map the numbers of a value (a number, or a nested array of numbers) with an operand (a number or a like array)
function mapNumbers(value, operand, fn) {
    if (Array.isArray(value)) {
        return value.map((item, ix) => mapNumbers(item, Array.isArray(operand) ? operand[ix] : operand, fn));
    }
    return fn(value, operand);
}


// Is a value a number or a (nested) array of numbers?
function isNumeric(value) {
    return Array.isArray(value) ? value.every(isNumeric) : typeof value === 'number';
}


// Apply a speaking style (or an array of styles, in order) to a voice's parameters - returns new parameters
//
// A style is {'intensity': 0-1 (default 1), 'scale': {param: factor}, 'add': {param: delta}, 'set': {param: value}}, at
// full intensity multiplying, adding to, or replacing parameters, and blended by intensity:
//   scale - value * factor ** intensity (geometric - ratios such as F0 and rate blend evenly in log)
//   add   - value + delta * intensity (decibels, offsets)
//   set   - value + (target - value) * intensity for numbers; booleans and strings switch at intensity 0.5
// The operand of an array parameter is a number (applied to each element) or a like array. Parameters are
// transformed in the order scale, add, set.
export function applyStyle(params, style) {
    if (Array.isArray(style)) {
        return style.reduce(applyStyle, params);
    }
    const result = {...params};
    const intensity = Math.max(0, Math.min(1, style.intensity === undefined ? 1 : style.intensity));
    const operations = {
        'scale': (value, factor) => value * factor ** intensity,
        'add': (value, delta) => value + delta * intensity,
        'set': (value, target) => value + (target - value) * intensity
    };
    for (const [operation, fn] of Object.entries(operations)) {
        for (const [key, operand] of Object.entries(style[operation] || {})) {
            if (!(key in voiceParameters)) {
                throw new Error(`Unknown style parameter "${key}"`);
            }
            const value = result[key];
            if (isNumeric(value) && isNumeric(operand)) {
                result[key] = mapNumbers(value, operand, fn);
            } else if (operation === 'set' && typeof value === typeof operand) {
                result[key] = intensity >= 0.5 ? operand : value;
            } else {
                throw new Error(`Invalid style ${operation} of "${key}"`);
            }
        }
    }
    return result;
}


// The vowel formant targets of a vowel blend - {ph: [[F1, F2, F3], ...]} and schwa [F1, F2, F3]. The end points are
// the tables themselves.
function blendVowels(blend) {
    const mix = (men, women) => mapNumbers(men, women, (man, woman) => man + (woman - man) * blend);
    if (blend === 0 || blend === 1) {
        const table = blend === 0 ? 'men' : 'women';
        return {
            'vowelTargets': Object.fromEntries(Object.entries(vowels).map(([ph, vowel]) => [ph, vowel[table]])),
            'schwaTargets': schwa[table]
        };
    }
    return {
        'vowelTargets': Object.fromEntries(Object.entries(vowels).map(([ph, vowel]) => [ph, mix(vowel.men, vowel.women)])),
        'schwaTargets': mix(schwa.men, schwa.women)
    };
}


// Resolve a voice (a name or a spec) to its complete synthesis parameters - the root voice, the spec's base chain, the
// spec, the overrides (parameters, unchecked - for experiments), and the style, in that order. The rate option
// multiplies the voice's rate, and pitchScale its f0. The result also has the voice's vowel formant targets
// ('vowelTargets' and 'schwaTargets').
export function resolveVoice(voice, {overrides = {}, style = null, rate = 1} = {}) {
    let params = {...baseVoice};
    for (const spec of specChain(voice)) {
        Object.assign(params, spec);
    }
    Object.assign(params, overrides);
    if (style !== null) {
        params = applyStyle(params, style);
    }
    params.rate *= rate;
    params.f0 *= params.pitchScale;
    return {...params, ...blendVowels(params.vowelBlend)};
}


// Validate a voice spec - returns a list of problems (unknown keys, wrong types, values outside the typical range)
export function validateVoice(spec) {
    const errors = [];
    for (const [key, value] of Object.entries(spec)) {
        const parameter = voiceParameters[key];
        if (voiceMetadata.includes(key)) {
            if (key === 'base' && value !== null && typeof value !== 'string' && typeof value !== 'object') {
                errors.push('Invalid base');
            } else if (key !== 'base' && typeof value !== 'string') {
                errors.push(`Invalid ${key}`);
            }
        } else if (parameter === undefined) {
            errors.push(`Unknown parameter "${key}"`);
        } else if (Array.isArray(parameter.default) !== Array.isArray(value) ||
            (isNumeric(parameter.default) ? !isNumeric(value) : typeof value !== typeof parameter.default)) {
            errors.push(`Invalid type of "${key}"`);
        } else if (parameter.range !== null) {
            const [low, high] = parameter.range;
            const outside = [value].flat(2).filter((number) => number < low || number > high);
            if (outside.length > 0) {
                errors.push(`"${key}" outside its typical range [${low}, ${high}]`);
            }
        }
    }
    return errors;
}
