// Text to speech - phrases to timed segments (Klatt's duration rules), segments to synthesizer parameter frames
// (targets, coarticulation, and smoothing), and an intonation (F0) contour

import {consonants, isVowel, noiseSpectra, schwa, vowels} from './phonemes.js';
import {Bandpass, HighPass, makeRandom, normalize, synthesize} from './klatt.js';
import {textToPhrases} from './lexicon.js';


const frameMs = 5;
const sampleRate = 44100;


// The voices
export const voices = {
    'male': {
        'name': 'male', 'table': 'male', 'f0': 108, 'accent': 0.36, 'rd': 0.9, 'breathDb': -40, 'rate': 1.15,
        'f4': 3300, 'f5': 4200, 'f6': 5500, 'b1': 75, 'formantScale': [1, 1, 1], 'noiseScale': 1,
        'jitter': 0.006, 'shimmer': 0.04, 'endLow': 0.7, 'openB1': 60, 'creak': 0.4, 'burstDb': 18, 'aspirationDb': -20, 'glottalAttack': true, 'vowelDecay': 0.2, 'rdDrift': 0.06, 'formantWobble': 0.008, 'shelfHz': 2000, 'shelfDb': 14,
        'fricVoicing': 0.3, 'fricVoiceBar': [300, 500], 'fricDevoice': 0.5, 'finalFricDb': -5, 'fricRiseMs': 40, 'fricRiseDb': -15,
        'higherFormants': [[6500, 500], [7500, 600], [8500, 700], [9500, 800], [10500, 900], [11500, 1000]]
    },
    'female': {
        'name': 'female', 'table': 'female', 'f0': 172, 'accent': 0.4, 'rd': 1.2, 'breathDb': -33, 'rate': 1.15,
        'f4': 4100, 'f5': 4900, 'f6': 6000, 'b1': 90, 'formantScale': [1.12, 1.16, 1.15], 'noiseScale': 1.2,
        'jitter': 0.005, 'shimmer': 0.035, 'endLow': 0.66, 'openB1': 80, 'creak': 0, 'burstDb': 18, 'aspirationDb': -20, 'glottalAttack': true, 'vowelDecay': 0.2, 'rdDrift': 0.06, 'formantWobble': 0.008, 'shelfHz': 2000, 'shelfDb': 10,
        'fricVoicing': 0.3, 'fricVoiceBar': [300, 500], 'fricDevoice': 0.5, 'finalFricDb': -5, 'fricRiseMs': 40, 'fricRiseDb': -15,
        'higherFormants': [[7200, 600], [8400, 700], [9600, 800], [10800, 900], [12000, 1000]]
    }
};


// Phoneme lookups
function phonemeType(ph) {
    if (ph === '_') {
        return 'pause';
    }
    return isVowel(ph) ? 'vowel' : consonants[ph].type;
}

function isVoiced(ph) {
    return isVowel(ph) || (ph in consonants && consonants[ph].voiced);
}

function isObstruent(ph) {
    return ['stop', 'affricate', 'fricative'].includes(phonemeType(ph));
}


// Stop releases by place - the burst duration (ms, voiceless; voiced bursts are two thirds as long), the voiceless
// voice onset time scale, and the frication after the burst (at the constriction as it opens, into the aspiration) - its
// level (dB re the burst) and decay time constant (ms). Velar releases are long and compact, labial ones brief and weak.
const stopRelease = {
    'labial': {'burst': 6, 'vot': 0.85, 'fricationDb': -6, 'fricationMs': 6},
    'alveolar': {'burst': 9, 'vot': 1, 'fricationDb': -3, 'fricationMs': 25},
    'velar': {'burst': 14, 'vot': 1.15, 'fricationDb': -2, 'fricationMs': 30}
};


// Build the timed segment list for phrases - [{ph, type, stress, start, dur, ...}] (ms)
function buildSegments(phrases, voice) {
    const segments = [{'ph': '_', 'type': 'pause', 'dur': 180}];
    phrases.forEach((phrase, ixPhrase) => {
        const phraseSegments = [];
        phrase.words.forEach((word, ixWord) => {
            word.phones.forEach((phone, ixPhone) => {
                phraseSegments.push({
                    'ph': phone.ph, 'type': phonemeType(phone.ph), 'stress': phone.stress, 'word': ixWord,
                    'wordStart': ixPhone === 0, 'wordEnd': ixPhone === word.phones.length - 1,
                    'isFunction': word.isFunction, 'syllables': word.syllables, 'phraseIndex': ixPhrase,
                    'phraseEnd': phrase.end, 'wordText': word.text
                });
            });
        });

        // Mark the phrase-final and word-final syllables (the last vowel and what follows it)
        let lastVowel = -1;
        phraseSegments.forEach((segment, ix) => {
            if (segment.type === 'vowel') {
                lastVowel = ix;
            }
        });
        phraseSegments.forEach((segment, ix) => {
            segment.phraseFinal = ix >= lastVowel;
            const wordSegments = phraseSegments.filter((other) => other.word === segment.word);
            const wordLastVowel = phraseSegments.indexOf(wordSegments.filter((other) => other.type === 'vowel').pop());
            segment.wordFinalSyllable = wordLastVowel === -1 || ix >= wordLastVowel;
        });

        // The syllable stress of each consonant - its nearest vowel's (following within the word, else preceding)
        phraseSegments.forEach((segment, ix) => {
            if (segment.type !== 'vowel') {
                let vowel = phraseSegments.slice(ix + 1).find((other) => other.type === 'vowel' && other.word === segment.word);
                if (vowel === undefined) {
                    vowel = phraseSegments.slice(0, ix).reverse().find((other) => other.type === 'vowel' && other.word === segment.word);
                }
                segment.syllableStress = vowel === undefined ? 0 : vowel.stress;
            }
        });
        segments.push(...phraseSegments);
        const isLast = ixPhrase === phrases.length - 1;
        segments.push({'ph': '_', 'type': 'pause', 'dur': isLast ? 250 : (phrase.end === 'comma' ? 200 : 420)});
    });

    // Durations (Klatt's rules, simplified)
    segments.forEach((segment, ix) => {
        if (segment.type === 'pause') {
            return;
        }
        const prev = segments[ix - 1];
        const next = segments[ix + 1];
        const def = segment.type === 'vowel' ? vowels[segment.ph] : consonants[segment.ph];
        let [inherent, minimum] = def.dur;
        let percent = 1;
        if (segment.type === 'vowel') {
            percent *= segment.phraseFinal ? 1.4 : 0.6;
            if (!segment.wordFinalSyllable) {
                percent *= 0.85;
            }
            if (segment.syllables > 1) {
                percent *= 0.8;
            }
            if (segment.stress === 0) {
                percent *= 0.5;
                minimum *= 0.5;
                if (segment.isFunction) {
                    percent *= 0.8;
                }
            }
            // Postvocalic context - stronger in the phrase-final syllable
            const strength = segment.phraseFinal ? 1 : 0.5;
            let context = 1;
            if (next.type === 'stop' || next.type === 'affricate') {
                context = isVoiced(next.ph) ? 1.2 : 0.7;
            } else if (next.type === 'fricative') {
                context = isVoiced(next.ph) ? 1.5 : 0.85;
            } else if (next.type === 'nasal') {
                context = 0.85;
            } else if (next.type === 'vowel') {
                context = 1.2;
            }
            percent *= 1 + (context - 1) * strength;
        } else {
            if (segment.phraseFinal) {
                percent *= 1.4;
            }
            if (!segment.wordStart) {
                percent *= 0.85;
            }
            if (segment.syllableStress === 0) {
                percent *= 0.7;
            }
            const prevConsonant = prev.type !== 'vowel' && prev.type !== 'pause' && prev.word === segment.word;
            const nextConsonant = next.type !== 'vowel' && next.type !== 'pause' && next.word === segment.word;
            if (prevConsonant || nextConsonant) {
                percent *= 0.75;
            }
        }
        segment.dur = ((inherent - minimum) * percent + minimum) / voice.rate;
    });

    // Stop phases - closure, release burst, and aspiration (voice onset time)
    segments.forEach((segment, ix) => {
        const prev = segments[ix - 1];
        const next = segments[ix + 1];
        if (segment.type === 'stop') {
            // Flapping - an alveolar stop between a vowel (or R) and an unstressed vowel
            if (segment.place !== 'x' && consonants[segment.ph].place === 'alveolar' && (prev.type === 'vowel' || prev.ph === 'R') &&
                next.type === 'vowel' && next.stress === 0) {
                segment.type = 'flap';
                segment.dur = 28 / voice.rate;
                return;
            }
            const released = next.type !== 'stop' && next.type !== 'affricate' && next.type !== 'nasal';
            const release = stopRelease[consonants[segment.ph].place];
            segment.burst = released ? release.burst * (isVoiced(segment.ph) ? 0.67 : 1) : 0;
            let aspiration = 0;
            if (!isVoiced(segment.ph) && released) {
                if (prev.ph === 'S' && prev.word === segment.word) {
                    aspiration = 8;
                } else if (next.type === 'vowel') {
                    aspiration = next.stress === 1 || next.stress === 2 ? 55 : 30;
                } else if (next.type === 'liquid' || next.type === 'glide') {
                    aspiration = 30;
                } else {
                    aspiration = 20;
                }
            } else if (isVoiced(segment.ph) && released && next.type !== 'pause') {
                aspiration = 15;
            }
            segment.aspiration = aspiration * (isVoiced(segment.ph) ? 1 : release.vot) / voice.rate;
            segment.dur += segment.burst + segment.aspiration;
        }
    });

    // Times
    let time = 0;
    for (const segment of segments) {
        segment.start = time;
        time += segment.dur;
    }
    return segments;
}


// Mix two formant triples
function mixFormants(first, second, fraction) {
    return first.map((value, ix) => value + (second[ix] - value) * fraction);
}


// The formant targets of a vowel segment at a fraction through it (diphthongs move start to end)
function vowelTargets(segment, fraction, voice) {
    const def = vowels[segment.ph][voice.table];
    let targets = def[0];
    if (def.length === 2) {
        const glide = Math.max(0, Math.min(1, (fraction - 0.2) / 0.65));
        targets = mixFormants(def[0], def[1], glide * glide * (3 - 2 * glide));
    }
    // Unstressed vowels reduce toward schwa - except a word-final IY, which stays tense ("happy")
    if (segment.stress === 0 && segment.ph !== 'ER') {
        const isHappy = segment.ph === 'IY' && segment.wordEnd && !segment.isFunction;
        targets = mixFormants(targets, schwa[voice.table], segment.ph === 'AH' ? 1 : (segment.isFunction ? 0.5 : (isHappy ? 0.1 : 0.35)));
    }
    return targets;
}


// The nearest vowel context of a consonant - the following vowel if adjacent, else the preceding, else searching out
function contextVowel(segments, ix) {
    if (segments[ix + 1] && segments[ix + 1].type === 'vowel') {
        return segments[ix + 1];
    }
    if (segments[ix - 1] && segments[ix - 1].type === 'vowel') {
        return segments[ix - 1];
    }
    for (let distance = 1; distance < 4; distance++) {
        for (const other of [segments[ix + distance], segments[ix - distance]]) {
            if (other && other.type === 'vowel') {
                return other;
            }
            if (other && other.type === 'pause') {
                break;
            }
        }
    }
    return null;
}


// The formant targets of a consonant in its vowel context
function consonantTargets(segments, ix, voice, context = contextVowel(segments, ix)) {
    const segment = segments[ix];
    const def = consonants[segment.ph];
    const contextFormants = context === null ? schwa[voice.table] : vowelTargets(context, segment.start < context.start ? 0 : 1, voice);
    const loci = def.f.map((value, ixF) => value * voice.formantScale[ixF]);
    if (segment.ph === 'HH') {
        return contextFormants;
    }
    // Velars - two-branched loci: before front vowels F2 and F3 pinch together high; before back vowels F2 sits well
    // above the vowel's, with F3 low
    if (def.place === 'velar') {
        const scale = voice.formantScale;
        const isFront = contextFormants[1] > 1600 * scale[1];
        if (isFront) {
            const f2 = contextFormants[1] * 1.04 + 150 * scale[1];
            return [loci[0], f2, Math.max(f2 + 250 * scale[2], contextFormants[2] - 250 * scale[2])];
        }
        return [loci[0], contextFormants[1] + 450 * scale[1], 2150 * scale[2]];
    }
    const k3 = def.k3 === undefined ? def.k * 0.6 : def.k3;
    return [loci[0], loci[1] + def.k * (contextFormants[1] - loci[1]), loci[2] + k3 * (contextFormants[2] - loci[2])];
}


// Consonant-vowel transition durations (ms) - [F1, F2 and F3] - by consonant type (and liquid or glide phoneme)
const transitionMs = {
    'stop': [30, 50], 'affricate': [30, 45], 'fricative': [30, 45], 'nasal': [20, 40], 'flap': [25, 30],
    'L': [35, 40], 'R': [45, 60], 'W': [50, 70], 'Y': [45, 60]
};


// A consonant's transition durations, or null if it has none (pauses, vowels, HH)
function consonantTransition(segment) {
    if (segment === undefined || segment.type === 'vowel' || segment.type === 'pause' || segment.type === 'aspirate') {
        return null;
    }
    return transitionMs[segment.ph] || transitionMs[segment.type];
}


// The release time of a segment (ms) - a stop's release (end of burst) for transitions through aspiration
function releaseTime(segment) {
    if (segment.type === 'stop' && segment.aspiration > 0) {
        return segment.start + segment.dur - segment.aspiration;
    }
    return segment.start + segment.dur;
}


// Apply the transitions from and to the neighboring consonants' loci to a vowel's formant targets at a time
function applyTransitions(segments, ix, time, formants, voice) {
    const segment = segments[ix];
    const result = [...formants];
    const blend = (locus, elapsed, durations) => {
        for (let kx = 0; kx < 3; kx++) {
            const duration = durations[kx === 0 ? 0 : 1];
            if (elapsed < duration) {
                const weight = Math.pow(1 - Math.max(0, elapsed) / duration, 2);
                result[kx] += (locus[kx] - result[kx]) * weight;
            }
        }
    };
    const prev = segments[ix - 1];
    const prevTransition = consonantTransition(prev);
    if (prevTransition !== null) {
        blend(consonantTargets(segments, ix - 1, voice), time - releaseTime(prev), prevTransition);
    }
    const next = segments[ix + 1];
    const nextTransition = consonantTransition(next);
    if (nextTransition !== null) {
        // The closing transition - toward the consonant's locus in this vowel's context (not the following vowel's)
        blend(consonantTargets(segments, ix + 1, voice, segment), segment.start + segment.dur - time, nextTransition);
    }
    return result;
}


// Frication level calibration - the RMS of a noise spectrum at unit amplitude, and of a vowel and aspiration
const rmsCache = new Map();
function noiseRms(spectrum) {
    const key = JSON.stringify(spectrum);
    if (!rmsCache.has(key)) {
        const random = makeRandom(3);
        const filter1 = new Bandpass();
        const filter1b = new Bandpass();
        const filter2 = new Bandpass();
        const filter2b = new Bandpass();
        const highPass1 = new HighPass();
        const highPass2 = new HighPass();
        filter1.set(spectrum.fp1, spectrum.bp1, sampleRate);
        filter1b.set(spectrum.fp1, spectrum.bp1, sampleRate);
        filter2.set(spectrum.fp2, spectrum.bp2, sampleRate);
        filter2b.set(spectrum.fp2, spectrum.bp2, sampleRate);
        highPass1.set(spectrum.fhp, sampleRate);
        highPass2.set(spectrum.fhp, sampleRate);
        let sum = 0;
        const count = 8000;
        for (let ix = 0; ix < count; ix++) {
            const noise = random.gauss();
            const value = highPass2.step(highPass1.step(spectrum.ab * noise + spectrum.ap1 * filter1b.step(filter1.step(noise)) +
                spectrum.ap2 * filter2b.step(filter2.step(noise))));
            sum += value * value;
        }
        rmsCache.set(key, Math.sqrt(sum / count));
    }
    return rmsCache.get(key);
}

function synthRms(frame, voice) {
    const key = JSON.stringify([frame, voice.name]);
    if (!rmsCache.has(key)) {
        const frames = [];
        for (let ix = 0; ix < 60; ix++) {
            frames.push({...frame});
        }
        const samples = synthesize(frames, {sampleRate, 'seed': 5, 'jitter': 0, 'shimmer': 0, 'sourceShelfHz': voice.shelfHz, 'sourceShelfDb': voice.shelfDb,
            'higherFormants': voice.higherFormants});
        let sum = 0;
        for (let ix = samples.length / 3; ix < samples.length; ix++) {
            sum += samples[Math.floor(ix)] * samples[Math.floor(ix)];
        }
        rmsCache.set(key, Math.sqrt(sum / (samples.length * 2 / 3)));
    }
    return rmsCache.get(key);
}


// The calibration levels for a voice - the stressed vowel RMS and the aspiration RMS at unit amplitude
function calibration(voice) {
    const [f1, f2, f3] = vowels.AA[voice.table][0];
    const base = {'f0': voice.f0, 'rd': voice.rd, f1, f2, f3, 'f4': voice.f4, 'f5': voice.f5, 'f6': voice.f6, 'b1': voice.b1};
    return {
        'vowel': synthRms({...base, 'av': 1}, voice),
        'aspiration': synthRms({...base, 'ah': 1, 'b1': 250}, voice)
    };
}


// A noise spectrum for a voice, at a level relative to a stressed vowel - returns frame parameters
function noiseParams(spectrum, voice, levels, adjustDb = 0) {
    const scaled = {...spectrum, 'fp1': spectrum.fp1 * voice.noiseScale, 'fp2': spectrum.fp2 * voice.noiseScale, 'fhp': spectrum.fhp * voice.noiseScale};
    const af = Math.pow(10, (spectrum.db + adjustDb) / 20) * levels.vowel / noiseRms(scaled);
    return {af, 'fp1': scaled.fp1, 'bp1': scaled.bp1, 'ap1': scaled.ap1, 'fp2': scaled.fp2, 'bp2': scaled.bp2, 'ap2': scaled.ap2, 'ab': scaled.ab, 'fhp': scaled.fhp};
}


// The aspiration amplitude for a level relative to a stressed vowel
function aspirationLevel(db, levels) {
    return Math.pow(10, db / 20) * levels.vowel / levels.aspiration;
}


// The raw (unsmoothed) parameter targets of a frame at a time within a segment
function segmentFrame(segments, ix, time, voice, levels) {
    const segment = segments[ix];
    const local = time - segment.start;
    const fraction = local / segment.dur;
    const frame = {
        'av': 0, 'ah': 0, 'af': 0, 'rd': voice.rd, 'b1': voice.b1, 'b2': 90, 'b3': 150, 'b4': 200, 'b5': 300, 'b6': 400,
        'fnp': 270, 'bnp': 100, 'fnz': 270, 'bnz': 100, 'ap1': 0, 'ap2': 0, 'ab': 0, 'tauF': 15, 'tauA': 5
    };
    let formants;
    const def = consonants[segment.ph];
    switch (segment.type) {
    case 'pause': {
        // Hold the neighboring formants through silence
        const neighbor = local < segment.dur / 2 && ix > 0 ? ix - 1 : ix + 1;
        const other = segments[neighbor];
        if (other === undefined || other.type === 'pause') {
            formants = schwa[voice.table];
        } else if (other.type === 'vowel') {
            formants = vowelTargets(other, neighbor < ix ? 1 : 0, voice);
        } else {
            formants = consonantTargets(segments, neighbor, voice);
        }
        frame.tauF = 5;
        break;
    }
    case 'vowel': {
        formants = applyTransitions(segments, ix, time, vowelTargets(segment, fraction, voice), voice);
        // Loudness peaks early in the vowel and decays ~3 dB by its end
        const envelope = 1 - voice.vowelDecay * Math.max(0, (fraction - 0.25) / 0.75);
        frame.av = (segment.stress === 1 ? 1 : (segment.stress === 2 ? 0.9 : 0.72)) * envelope;
        frame.rd = voice.rd * (segment.stress === 1 ? 0.88 : (segment.stress === 2 ? 0.95 : 1.15));
        frame.ah = aspirationLevel(voice.breathDb, levels) * frame.av;
        frame.tauF = vowels[segment.ph][voice.table].length === 2 ? 12 : 6;

        // A vowel after a pause starts with a glottal attack - an abrupt onset with a few irregular pulses
        if (voice.glottalAttack && segments[ix - 1].type === 'pause' && local < 30) {
            frame.tauA = 1;
            frame.jit = 6;
            frame.shim = 4;
            frame.rd = voice.rd * 0.7;
        }
        break;
    }
    case 'nasal':
        formants = consonantTargets(segments, ix, voice);
        frame.av = 0.8;
        frame.ah = aspirationLevel(voice.breathDb, levels) * 0.5;
        frame.fnz = 450;
        frame.b1 = 100;
        frame.b2 = 300;
        frame.b3 = 300;
        frame.tauF = 10;
        break;
    case 'liquid':
    case 'glide':
        formants = consonantTargets(segments, ix, voice);
        frame.av = 0.82;
        frame.ah = aspirationLevel(voice.breathDb, levels) * 0.6;
        frame.b2 = segment.ph === 'L' ? 150 : 110;
        frame.tauF = segment.ph === 'L' ? 8 : 14;
        break;
    case 'flap':
        formants = consonantTargets(segments, ix, voice);
        formants = [formants[0] * 1.2, formants[1], formants[2]];
        frame.av = 0.55;
        frame.tauF = 8;
        frame.tauA = 3;
        // A /t/ flap is partly devoiced, with a faint release burst (incompletely neutralized with a /d/ flap)
        if (!def.voiced) {
            frame.av = 0.15;
            if (local > segment.dur * 0.5) {
                Object.assign(frame, noiseParams(noiseSpectra.alveolar, voice, levels, voice.burstDb - 6));
            }
            frame.tauA = 1;
        }
        break;
    case 'aspirate':
        formants = consonantTargets(segments, ix, voice);
        frame.ah = aspirationLevel(-24, levels);
        frame.av = segments[ix - 1].type === 'vowel' && segments[ix + 1].type === 'vowel' ? 0.25 : 0;
        frame.b1 = 300;
        frame.tauF = 10;
        break;
    case 'fricative': {
        formants = consonantTargets(segments, ix, voice);
        // A voiceless fricative after a pause rises gradually as the constriction forms (an abrupt rise is the
        // affricate cue). A fricative before a pause is weaker (falling subglottal pressure) and fades out over its end.
        const isPrepausal = segments[ix + 1].type === 'pause';
        const riseMs = Math.min(voice.fricRiseMs, 0.4 * segment.dur);
        const riseDb = segments[ix - 1].type === 'pause' && !def.voiced ? voice.fricRiseDb * Math.max(0, 1 - local / riseMs) : 0;
        const fadeDb = isPrepausal ? voice.finalFricDb - 12 * Math.max(0, (fraction - 0.6) / 0.4) : 0;
        Object.assign(frame, noiseParams(noiseSpectra[segment.ph], voice, levels, riseDb + fadeDb));
        frame.av = def.voiced ? voice.fricVoicing : 0;
        frame.b1 = def.voiced ? 100 : 250;
        if (def.voiced) {
            // The constriction damps the upper formants - the voicing is mostly a low voice bar
            frame.b2 = voice.fricVoiceBar[0];
            frame.b3 = voice.fricVoiceBar[1];
            // Voiced fricatives before a pause devoice toward their end
            if (isPrepausal) {
                frame.av *= 1 - voice.fricDevoice * fraction;
            }
        }
        frame.tauF = 10;
        frame.tauA = 4;
        break;
    }
    case 'affricate': {
        formants = consonantTargets(segments, ix, voice);
        const closure = segment.dur * 0.45;
        if (local >= closure) {
            Object.assign(frame, noiseParams(noiseSpectra[segment.ph], voice, levels));
            frame.av = def.voiced ? 0.35 : 0;
        } else {
            frame.av = def.voiced ? 0.15 : 0;
        }
        frame.b1 = 250;
        frame.tauF = 8;
        frame.tauA = 2;
        break;
    }
    case 'stop': {
        formants = consonantTargets(segments, ix, voice);
        const closureEnd = segment.dur - segment.burst - segment.aspiration;
        const next = segments[ix + 1];
        const release = stopRelease[def.place];
        // The release noise spectrum by place - velar noise is compact, at the locus F2
        const burstSpectrum = def.place === 'velar' ? {...noiseSpectra.velar, 'fp1': formants[1] / voice.noiseScale} : noiseSpectra[def.place];
        if (local < closureEnd) {
            // Closure - silence, or a voice bar for voiced stops (radiated through the walls, ~-15 dB re the vowel)
            if (def.voiced) {
                frame.av = 0.3;
                formants = [180, formants[1], formants[2]];
                frame.b2 = 600;
                frame.b3 = 900;
                frame.b4 = 1200;
                frame.b5 = 1500;
            }
        } else if (local < closureEnd + segment.burst) {
            // The release burst
            Object.assign(frame, noiseParams(burstSpectrum, voice, levels, voice.burstDb + (def.voiced ? -6 : 0)));
            frame.av = def.voiced ? 0.25 : 0;
            frame.tauA = 1;
        } else {
            // Frication at the opening constriction, decaying into the aspiration
            const since = local - closureEnd - segment.burst;
            Object.assign(frame, noiseParams(burstSpectrum, voice, levels, voice.burstDb + release.fricationDb + (def.voiced ? -6 : 0)));
            frame.af *= Math.exp(-since / release.fricationMs);

            // Aspiration - the vocal tract moving from the locus to the next segment's targets
            if (next.type === 'vowel') {
                formants = applyTransitions(segments, ix + 1, time, vowelTargets(next, 0, voice), voice);
            } else if (next.type !== 'pause') {
                const locus = formants;
                const nextFormants = consonantTargets(segments, ix + 1, voice);
                const weight = Math.min(1, (local - closureEnd - segment.burst) / segment.aspiration);
                formants = locus.map((value, kx) => value + (nextFormants[kx] - value) * weight);
            }
            frame.ah = aspirationLevel(voice.aspirationDb + (def.voiced ? -5 : 0), levels);
            // Voiced stops - voicing at once, with the glottis closed (narrow F1); voiceless - the glottis open (wide F1)
            frame.av = def.voiced ? 0.6 : 0;
            frame.b1 = def.voiced ? 120 : 250;
            frame.tauA = def.voiced ? 1 : 3;
        }
        frame.tauF = 8;
        break;
    }
    default:
        throw new Error(`Unknown segment type ${segment.type}`);
    }

    // Nasalization spreading into vowels next to nasals
    if (segment.type === 'vowel') {
        const prev = segments[ix - 1];
        const next = segments[ix + 1];
        if ((next && next.type === 'nasal' && fraction > 0.6) || (prev && prev.type === 'nasal' && fraction < 0.25)) {
            frame.fnz = 360;
        }
    }
    [frame.f1, frame.f2, frame.f3] = formants;
    return frame;
}


// Zero-phase smoothing (forward and backward one-pole) of a frame parameter with per-frame time constants (ms)
function smooth(frames, key, tauKey) {
    for (const direction of [1, -1]) {
        const start = direction === 1 ? 0 : frames.length - 1;
        let value = frames[start][key];
        for (let ix = start; ix >= 0 && ix < frames.length; ix += direction) {
            const tau = typeof tauKey === 'number' ? tauKey : frames[ix][tauKey];
            const alpha = 1 - Math.exp(-frameMs / Math.max(0.5, tau));
            value += alpha * (frames[ix][key] - value);
            frames[ix][key] = value;
        }
    }
}


// The intonation (F0) contour of the segments - an F0 for each frame time
function intonation(segments, voice, frameCount, random) {
    const points = [];
    const phrases = [];
    let current = null;
    for (const segment of segments) {
        if (segment.type === 'pause') {
            current = null;
        } else {
            if (current === null) {
                current = {'segments': [], 'end': segment.phraseEnd};
                phrases.push(current);
            }
            current.segments.push(segment);
        }
    }
    for (const phrase of phrases) {
        const start = phrase.segments[0].start;
        const last = phrase.segments[phrase.segments.length - 1];
        const end = last.start + last.dur;
        const base = (time) => voice.f0 * (1.04 - Math.min(0.14, 0.07 * (time - start) / 1000));
        const accents = phrase.segments.filter((segment) => segment.type === 'vowel' && segment.stress === 1 && !segment.isFunction);
        if (accents.length === 0) {
            const vowel = phrase.segments.find((segment) => segment.type === 'vowel');
            if (vowel !== undefined) {
                accents.push(vowel);
            }
        }
        const isExclamation = phrase.end === 'exclamation';
        points.push({'time': start, 'f0': base(start) * (phrase.end === 'question' ? 1.0 : 1.02)});
        accents.forEach((vowel, ixAccent) => {
            const isNuclear = ixAccent === accents.length - 1;
            const height = voice.accent * (isExclamation ? 1.4 : 1) * Math.pow(0.78, ixAccent) * (ixAccent === 0 ? 1.15 : 1) *
                (0.75 + 0.5 * random.uniform());
            if (ixAccent > 0) {
                // The valley between accents
                const prevVowel = accents[ixAccent - 1];
                const valleyTime = (prevVowel.start + prevVowel.dur + vowel.start) / 2;
                points.push({'time': valleyTime, 'f0': base(valleyTime) * 1.0});
            }
            if (isNuclear && phrase.end === 'question') {
                // Yes/no question - a low nuclear accent, then the rise
                const lowTime = vowel.start + 0.3 * vowel.dur;
                points.push({'time': lowTime, 'f0': base(lowTime) * 0.95});
                points.push({'time': end, 'f0': voice.f0 * 1.5});
            } else if (isNuclear) {
                const peakTime = vowel.start + (vowel.phraseFinal ? 0.3 : 0.55) * vowel.dur;
                points.push({'time': vowel.start, 'f0': base(vowel.start) * (1 + height * 0.4)});
                points.push({'time': peakTime, 'f0': base(peakTime) * (1 + height)});
                if (phrase.end === 'comma') {
                    // Continuation - fall, then a slight rise
                    const lowTime = Math.max(peakTime + 60, end - 100);
                    points.push({'time': lowTime, 'f0': base(lowTime) * 0.94});
                    points.push({'time': end, 'f0': base(end) * 1.06});
                } else {
                    const fallTime = Math.min(end - 20, vowel.start + vowel.dur);
                    points.push({'time': fallTime, 'f0': base(fallTime) * (0.5 * (1 + height) + 0.5 * voice.endLow)});
                    points.push({'time': end, 'f0': voice.f0 * voice.endLow});
                }
            } else {
                const peakTime = vowel.start + 0.6 * vowel.dur;
                points.push({'time': vowel.start, 'f0': base(vowel.start) * (1 + height * 0.5)});
                points.push({'time': peakTime, 'f0': base(peakTime) * (1 + height)});
            }
        });
    }
    points.sort((first, second) => first.time - second.time);

    // Interpolate the points (cosine), hold the ends
    const f0 = new Float64Array(frameCount);
    let ixPoint = 0;
    for (let ix = 0; ix < frameCount; ix++) {
        const time = ix * frameMs;
        while (ixPoint < points.length - 1 && points[ixPoint + 1].time <= time) {
            ixPoint++;
        }
        const pointA = points[ixPoint];
        const pointB = points[Math.min(points.length - 1, ixPoint + 1)];
        if (time <= pointA.time || pointA === pointB) {
            f0[ix] = pointA.f0;
        } else {
            const fraction = (time - pointA.time) / (pointB.time - pointA.time);
            const eased = 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, fraction));
            f0[ix] = Math.exp(Math.log(pointA.f0) + (Math.log(pointB.f0) - Math.log(pointA.f0)) * eased);
        }
    }
    return f0;
}


// A soft limiter - linear below 0.8, approaching 1 above
function softLimit(value) {
    const magnitude = Math.abs(value);
    if (magnitude <= 0.8) {
        return value;
    }
    return Math.sign(value) * (0.8 + 0.2 * Math.tanh((magnitude - 0.8) / 0.2));
}


// Synthesize text with a voice - returns {samples (Float32Array), sampleRate, segments}
// For streaming, onBlock(samples) receives each block as it is synthesized, at a fixed calibrated gain with a soft
// limiter (a whole utterance can't be normalized before it's finished).
export function speak(text, voiceName = 'male', {seed = 1, rate = 1, overrides = {}, onBlock = null} = {}) {
    const voice = {...voices[voiceName], ...overrides};
    voice.rate *= rate;
    const random = makeRandom(seed);
    const levels = calibration(voice);
    const segments = buildSegments(textToPhrases(text), voice);
    const last = segments[segments.length - 1];
    const totalMs = last.start + last.dur;
    const frameCount = Math.ceil(totalMs / frameMs) + 1;

    // Raw targets
    const frames = [];
    let ixSegment = 0;
    for (let ix = 0; ix < frameCount; ix++) {
        const time = ix * frameMs;
        while (ixSegment < segments.length - 1 && segments[ixSegment + 1].start <= time) {
            ixSegment++;
        }
        const frame = segmentFrame(segments, ixSegment, time, voice, levels);
        frame.segment = ixSegment;
        frames.push(frame);
    }

    // Phrase-final voice quality - the voice softens and becomes breathier at the ends of phrases
    for (let ix = 0; ix < frameCount; ix++) {
        const segment = segments[frames[ix].segment];
        if (segment.phraseFinal) {
            let phraseEndTime = segment.start + segment.dur;
            for (let jx = frames[ix].segment + 1; jx < segments.length && segments[jx].type !== 'pause'; jx++) {
                phraseEndTime = segments[jx].start + segments[jx].dur;
            }
            const closeness = Math.max(0, 1 - (phraseEndTime - ix * frameMs) / 250);
            const isStatement = segment.phraseEnd === 'period' || segment.phraseEnd === 'wh' || segment.phraseEnd === 'exclamation';
            const creak = isStatement ? voice.creak * Math.max(0, 1 - (phraseEndTime - ix * frameMs) / 160) : 0;
            frames[ix].creak = creak;
            frames[ix].rd += 0.6 * closeness * (1 - creak) - 0.3 * creak;
            frames[ix].av *= 1 - 0.35 * closeness;
            frames[ix].ah *= 1 + 1.5 * closeness * (1 - creak);
            frames[ix].jit = 1 + 8 * creak;
            frames[ix].shim = 1 + 5 * creak;
        }
    }

    // Smoothing - formants by their segment's time constant, amplitudes quickly
    for (const key of ['f1', 'f2', 'f3']) {
        smooth(frames, key, 'tauF');
    }
    for (const key of ['b1', 'b2', 'b3', 'b4', 'b5', 'fnz']) {
        smooth(frames, key, 6);
    }
    for (const key of ['av', 'ah', 'af']) {
        smooth(frames, key, 'tauA');
    }
    smooth(frames, 'rd', 20);

    // Intonation, with microprosody (F0 raised after voiceless obstruents, lowered in voiced ones) and slow drift
    const f0 = intonation(segments, voice, frameCount, random);
    const drift = [random.uniform() * 6.28, random.uniform() * 6.28, random.uniform() * 6.28];
    for (let ix = 0; ix < frameCount; ix++) {
        const time = ix * frameMs;
        const segment = segments[frames[ix].segment];
        const prev = segments[frames[ix].segment - 1];
        let micro = 1;
        if (segment.type === 'vowel' && prev && isObstruent(prev.ph)) {
            const since = time - segment.start;
            micro = 1 + (isVoiced(prev.ph) ? -0.03 : 0.06) * Math.exp(-since / 35);
        } else if (isObstruent(segment.ph) && isVoiced(segment.ph)) {
            micro = 0.95;
        }
        const t = time / 1000;
        const wobble = 1 + 0.008 * Math.sin(2 * Math.PI * 0.7 * t + drift[0]) + 0.006 * Math.sin(2 * Math.PI * 1.9 * t + drift[1]) +
            0.004 * Math.sin(2 * Math.PI * 4.3 * t + drift[2]);
        // Creak - the pitch drops toward vocal fry
        const creakDrop = 1 - 0.35 * (frames[ix].creak || 0);
        frames[ix].f0 = f0[ix] * micro * wobble * creakDrop;
    }
    smooth(frames, 'f0', 12);

    // Micro-dynamics - slow drifts in voice quality and small articulatory fluctuations of the formants
    const microPhases = Array.from({'length': 8}, () => random.uniform() * 2 * Math.PI);
    for (let ix = 0; ix < frameCount; ix++) {
        const t = ix * frameMs / 1000;
        frames[ix].rd *= 1 + voice.rdDrift * Math.sin(2 * Math.PI * 0.45 * t + microPhases[0]) + 0.05 * Math.sin(2 * Math.PI * 1.3 * t + microPhases[1]);
        for (let kx = 1; kx <= 3; kx++) {
            const wobble = voice.formantWobble * Math.sin(2 * Math.PI * (2.1 + 0.9 * kx) * t + microPhases[1 + kx]) +
                voice.formantWobble * 0.66 * Math.sin(2 * Math.PI * (5.3 + 1.7 * kx) * t + microPhases[4 + kx]);
            frames[ix][`f${kx}`] *= 1 + wobble;
        }
    }

    // The voice's fixed higher formants and open-phase F1 damping
    for (const frame of frames) {
        frame.ob1 = voice.openB1;
        frame.f4 = voice.f4;
        frame.f5 = voice.f5;
        frame.f6 = voice.f6;
    }
    const streamGain = 0.89 / (3.2 * levels.vowel);
    const streamBlock = onBlock === null ? null : (block) => {
        const scaled = new Float32Array(block.length);
        for (let ix = 0; ix < block.length; ix++) {
            scaled[ix] = softLimit(block[ix] * streamGain);
        }
        onBlock(scaled);
    };
    const samples = synthesize(frames, {frameMs, sampleRate, seed, 'jitter': voice.jitter, 'shimmer': voice.shimmer,
        'sourceShelfHz': voice.shelfHz, 'sourceShelfDb': voice.shelfDb, 'higherFormants': voice.higherFormants, 'onBlock': streamBlock});

    let rawPeak = 0;
    for (const sample of samples) {
        rawPeak = Math.max(rawPeak, Math.abs(sample));
    }
    return {'samples': normalize(samples, 0.89), sampleRate, segments, frames, rawPeak, 'vowelLevel': levels.vowel};
}
