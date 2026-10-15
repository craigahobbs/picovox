// Text to speech - phrases to timed segments (Klatt's duration rules), segments to synthesizer parameter frames
// (targets, coarticulation, and smoothing), and an intonation (F0) contour

import {Bandpass, HighPass, makeRandom, normalize, synthesize} from './klatt.js';
import {consonants, isVowel, noiseSpectra, vowels} from './phonemes.js';
import {resolveVoice, voices} from './voices.js';
import {textToPhrases} from './lexicon.js';


// The voices (lib/voices.js), re-exported for convenience
export {voices};


const frameMs = 5;
const sampleRate = 44100;


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


// Words that can begin an intermediate phrase - conjunctions and relatives (preferred), and prepositions
const breakWords = {
    'and': 3, 'but': 3, 'or': 2, 'because': 3, 'while': 3, 'where': 3, 'when': 3, 'which': 3, 'who': 3, 'whose': 3,
    'that': 1, 'until': 2, 'after': 2, 'before': 2, 'since': 2, 'if': 2, 'though': 3, 'although': 3, 'so': 2,
    'with': 1, 'toward': 1, 'towards': 1, 'from': 1, 'into': 1, 'in': 1, 'on': 1, 'at': 1, 'for': 1, 'through': 1,
    'across': 1, 'under': 1, 'without': 1, 'beyond': 1, 'between': 1, 'during': 1, 'against': 1
};


// Words read without a pitch accent though the lexicon counts them as content words - reporting verbs, relatives and
// subordinators, longer prepositions, and reflexive and demonstrative pronouns
const unaccentedWords = new Set([
    'said', 'says', 'replied', 'asked', 'answered', 'who', 'whom', 'whose', 'which', 'when', 'where', 'while', 'because',
    'though', 'although', 'whether', 'yet', 'nor', 'after', 'before', 'under', 'through', 'without', 'within', 'between',
    'against', 'during', 'toward', 'towards', 'until', 'since', 'upon', 'across', 'among', 'around', 'himself', 'herself',
    'itself', 'themselves', 'myself', 'yourself', 'ourselves', 'these', 'those', 'being', 'having'
]);


// The stem of a word for givenness - without a plural or possessive ending
function accentStem(word) {
    return word.length > 3 ? word.replace(/(?:'s|s)$/u, '') : word;
}


// Intermediate phrase breaks of a long phrase - the word indices that begin a new intermediate phrase. A phrase of
// more than minorSyllables splits before the break word (following a content word) nearest its middle, recursively.
function minorBreaks(words, voice, first = 0, last = words.length) {
    const syllables = words.slice(first, last).reduce((sum, word) => sum + word.syllables, 0);
    if (syllables <= voice.minorSyllables) {
        return [];
    }
    let best = null;
    let bestScore = Infinity;
    let before = words[first].syllables;
    for (let ix = first + 1; ix < last; ix++) {
        const word = words[ix];
        const after = syllables - before;
        if (word.text in breakWords && !words[ix - 1].isFunction && before >= 5 && after >= 5) {
            const score = Math.abs(before - after) - 2 * breakWords[word.text];
            if (score < bestScore) {
                best = ix;
                bestScore = score;
            }
        }
        before += word.syllables;
    }
    if (best === null) {
        return [];
    }
    return [...minorBreaks(words, voice, first, best), best, ...minorBreaks(words, voice, best, last)];
}


// The syllable count of a phrase
function phraseSyllables(phrase) {
    return phrase.words.reduce((sum, word) => sum + word.syllables, 0);
}


// The pause after a phrase (ms) - the utterance's trailing silence after the last; after a sentence, the sentence pause;
// after a semicolon or colon, the major-break pause; and after a comma, a pause that grows with the syllables of the
// phrases on either side (audiobook readers rarely pause between short phrases, and usually between long ones)
function phrasePause(phrases, ixPhrase, voice) {
    const phrase = phrases[ixPhrase];
    if (ixPhrase === phrases.length - 1) {
        return voice.finalPause;
    }
    if (phrase.end !== 'comma') {
        return voice.sentencePause;
    }
    if (phrase.major) {
        return voice.majorPause;
    }
    const length = Math.min(1, (phraseSyllables(phrase) + phraseSyllables(phrases[ixPhrase + 1])) / voice.commaPauseSyllables);
    return voice.commaPauseMin + (voice.commaPause - voice.commaPauseMin) * length;
}


// Build the timed segment list for phrases - [{ph, type, stress, start, dur, ...}] (ms)
function buildSegments(phrases, voice) {
    const segments = [{'ph': '_', 'type': 'pause', 'dur': voice.leadingPause}];
    phrases.forEach((phrase, ixPhrase) => {
        const phraseSegments = [];
        const breaks = minorBreaks(phrase.words, voice);
        phrase.words.forEach((word, ixWord) => {
            word.phones.forEach((phone, ixPhone) => {
                phraseSegments.push({
                    'ph': phone.ph, 'type': phonemeType(phone.ph), 'stress': phone.stress, 'word': ixWord,
                    'wordStart': ixPhone === 0, 'wordEnd': ixPhone === word.phones.length - 1,
                    'isFunction': word.isFunction, 'syllables': word.syllables, 'phraseIndex': ixPhrase,
                    'phraseEnd': phrase.end, 'wordText': word.text, 'minor': breaks.filter((ixBreak) => ixBreak <= ixWord).length
                });
            });
        });

        // Mark the phrase-final, intermediate-phrase-final, and word-final syllables (the last vowel and what follows it)
        let lastVowel = -1;
        const minorLastVowel = [];
        phraseSegments.forEach((segment, ix) => {
            if (segment.type === 'vowel') {
                lastVowel = ix;
                minorLastVowel[segment.minor] = ix;
            }
        });
        phraseSegments.forEach((segment, ix) => {
            segment.phraseInitial = ix === 0;
            segment.phraseFinal = ix >= lastVowel;
            segment.minorFinal = !segment.phraseFinal && segment.minor < breaks.length && ix >= minorLastVowel[segment.minor];
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
        // The segments, with a brief pause at each intermediate phrase break (the phrase's intonation continues through it)
        phraseSegments.forEach((segment, ix) => {
            if (ix > 0 && segment.minor !== phraseSegments[ix - 1].minor && voice.minorPause > 0) {
                segments.push({'ph': '_', 'type': 'pause', 'dur': voice.minorPause, 'minorPause': true});
            }
            segments.push(segment);
        });
        segments.push({'ph': '_', 'type': 'pause', 'dur': phrasePause(phrases, ixPhrase, voice)});
    });

    // Durations (Klatt's rules, simplified)
    segments.forEach((segment, ix) => {
        if (segment.type === 'pause') {
            return;
        }
        const prev = segments[ix - 1];
        const next = segments[ix + 1];
        const def = segment.type === 'vowel' ? vowels[segment.ph] : consonants[segment.ph];
        const [inherent] = def.dur;
        let [, minimum] = def.dur;
        let percent = 1;
        if (segment.type === 'vowel') {
            percent *= segment.phraseFinal ? voice.finalLengthen : (segment.minorFinal ? voice.minorLengthen : 0.6);
            if (!segment.wordFinalSyllable) {
                percent *= 0.85;
            }
            if (segment.syllables > 1) {
                percent *= 0.8;
            }
            if (segment.stress === 0) {
                percent *= 0.5;
                minimum *= voice.unstressedMinimum;
                if (segment.isFunction) {
                    percent *= 0.8;
                }
            }
            // Postvocalic context - stronger in the phrase-final syllable
            const strength = segment.phraseFinal ? 1 : (segment.minorFinal ? 0.75 : 0.5);
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
            } else if (segment.minorFinal) {
                percent *= 1.2;
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
                } else if ((next.type === 'liquid' || next.type === 'glide') && next.word === segment.word &&
                    next.phraseIndex === segment.phraseIndex) {
                    // A cluster - the liquid or glide is devoiced through a long voice onset time, overlapping its start
                    aspiration = voice.clusterVot;
                    next.dur *= voice.clusterLiquidScale;
                    segment.cluster = true;
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

    // Vowels after a coronal consonant (fronted back vowels)
    segments.forEach((segment, ix) => {
        if (segment.type === 'vowel' && isCoronal(segments[ix - 1])) {
            segment.coronalOnset = true;
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


// The default glide of a vowel with two targets - the fractions through the vowel where it starts and ends
const defaultGlide = [0.2, 0.85];


// Is a consonant coronal (tongue tip or blade, or the palatal glide) - excluding the liquids, whose tongue body is back?
function isCoronal(segment) {
    return segment !== undefined && segment.type !== 'vowel' && segment.type !== 'pause' && segment.type !== 'liquid' &&
        ['alveolar', 'dental', 'postalveolar', 'palatal'].includes(consonants[segment.ph].place);
}


// Mix two formant triples
function mixFormants(first, second, fraction) {
    return first.map((value, ix) => value + (second[ix] - value) * fraction);
}


// The formant targets of a vowel segment at a fraction through it (diphthongs move start to end over their glide - the
// vowel table's, or defaultGlide). Unstressed vowels are reduced toward schwa (by the voice's reduction), except
// phrase-initially - with no preceding context to undershoot from, an initial "a" is a full "uh" (a bare schwa after
// silence is heard as "the")
function vowelTargets(segment, fraction, voice) {
    const def = voice.vowelTargets[segment.ph];
    const [glideStart, glideEnd] = vowels[segment.ph].glide || defaultGlide;
    const glide = Math.max(0, Math.min(1, (fraction - glideStart) / (glideEnd - glideStart)));
    const eased = glide * glide * (3 - 2 * glide);
    let [targets] = def;
    if (def.length === 2) {
        targets = mixFormants(def[0], def[1], eased);
    }
    // A back rounded vowel after a coronal (or Y) is fronted - the tongue body, forward for the consonant, moves back
    // late in the vowel (American "two", "do", "new", "so"): F2 raised toward IY's, mostly released by the vowel's end
    if (segment.coronalOnset && vowels[segment.ph].fronting) {
        const fronting = voice.coronalFronting * vowels[segment.ph].fronting *
            (voice.coronalFrontingEnd + (1 - voice.coronalFrontingEnd) * (1 - eased));
        targets = [targets[0], targets[1] + fronting * (voice.vowelTargets.IY[0][1] - targets[1]), targets[2]];
    }
    // Unstressed vowels reduce toward schwa - except a word-final IY, which stays tense ("happy"), and a phrase-initial
    // vowel, which starts full after silence (a sentence-initial "a")
    if (segment.stress === 0 && segment.ph !== 'ER' && !segment.phraseInitial) {
        const isHappy = segment.ph === 'IY' && segment.wordEnd && !segment.isFunction;
        const reduction = segment.ph === 'AH' ? 1 : (segment.isFunction ? 0.5 : (isHappy ? 0.1 : 0.35));
        targets = mixFormants(targets, voice.schwaTargets, Math.min(1, reduction * voice.reduction));
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
    const contextFormants = context === null ? voice.schwaTargets : vowelTargets(context, segment.start < context.start ? 0 : 1, voice);
    // A nasal's F1 sits just above the nasal zero, which cancels it and leaves the low nasal pole - it isn't scaled
    // for the voice (scaleNasalF1), as a higher (female) F1 escapes the zero and the murmur turns vowel-like
    const isFixedNasalF1 = def.type === 'nasal' && !voice.scaleNasalF1;
    const loci = def.f.map((value, ixF) => (isFixedNasalF1 && ixF === 0 ? value : value * voice.formantScale[ixF]));
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


// Apply the transitions from and to the neighboring consonants' loci to a vowel's formant targets at a time (their
// durations scaled by the voice's transitionScale)
function applyTransitions(segments, ix, time, formants, voice) {
    const segment = segments[ix];
    const result = [...formants];
    const blend = (locus, elapsed, durations) => {
        for (let kx = 0; kx < 3; kx++) {
            const duration = durations[kx === 0 ? 0 : 1] * voice.transitionScale;
            if (elapsed < duration) {
                const weight = (1 - Math.max(0, elapsed) / duration) ** 2;
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
    const key = JSON.stringify([frame, voice.shelfHz, voice.shelfDb, voice.higherFormants]);
    if (!rmsCache.has(key)) {
        const frames = [];
        for (let ix = 0; ix < 60; ix++) {
            frames.push({...frame});
        }
        const samples = synthesize(frames, {
            sampleRate, 'seed': 5, 'jitter': 0, 'shimmer': 0, 'sourceShelfHz': voice.shelfHz, 'sourceShelfDb': voice.shelfDb,
            'higherFormants': voice.higherFormants
        });
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
    const [[f1, f2, f3]] = voice.vowelTargets.AA;
    const base = {'f0': voice.f0, 'rd': voice.rd, f1, f2, f3, 'f4': voice.f4, 'f5': voice.f5, 'f6': voice.f6, 'b1': voice.b1};
    return {
        'vowel': synthRms({...base, 'av': 1}, voice),
        'aspiration': synthRms({...base, 'ah': 1, 'b1': 250}, voice)
    };
}


// A noise spectrum for a voice, at a level relative to a stressed vowel - returns frame parameters
function noiseParams(spectrum, voice, levels, adjustDb = 0) {
    const scale = spectrum.sibilant === false ? voice.nonSibilantScale : voice.noiseScale;
    const scaled = {...spectrum, 'fp1': spectrum.fp1 * scale, 'fp2': spectrum.fp2 * scale, 'fhp': spectrum.fhp * scale};
    const af = 10 ** ((spectrum.db + adjustDb) / 20) * levels.vowel / noiseRms(scaled);
    return {
        af, 'fp1': scaled.fp1, 'bp1': scaled.bp1, 'ap1': scaled.ap1, 'fp2': scaled.fp2, 'bp2': scaled.bp2, 'ap2': scaled.ap2,
        'ab': scaled.ab, 'fhp': scaled.fhp
    };
}


// The aspiration amplitude for a level relative to a stressed vowel
function aspirationLevel(db, levels) {
    return 10 ** (db / 20) * levels.vowel / levels.aspiration;
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
            formants = voice.schwaTargets;
        } else if (other.type === 'vowel') {
            formants = vowelTargets(other, neighbor < ix ? 1 : 0, voice);
        } else if (other.type === 'nasal') {
            // A nasal's murmur resonances too (its nasal pole and zero, and bandwidths) - otherwise the smoothed onset
            // or offset of the murmur passes through an oral, vowel-like spectrum (an utterance-initial "n" is heard as
            // "an")
            const nasal = segmentFrame(segments, neighbor, neighbor < ix ? other.start + other.dur - 1 : other.start, voice, levels);
            formants = [nasal.f1, nasal.f2, nasal.f3];
            for (const key of ['b1', 'b2', 'b3', 'fnp', 'bnp', 'fnz', 'bnz']) {
                frame[key] = nasal[key];
            }
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
        frame.tauF = voice.vowelTargets[segment.ph].length === 2 || (segment.coronalOnset && vowels[segment.ph].fronting) ? 12 : 6;

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
        // The murmur - the nasal pole, with the zero cancelling F1 (the nasal F1 locus and the pole and zero are not
        // scaled by formantScale, so the zero still cancels F1 in the female voice)
        frame.fnp = voice.nasalPole;
        frame.fnz = voice.nasalZero;
        frame.b1 = 100;
        frame.b2 = 300;
        frame.b3 = 300;
        frame.tauF = 10;
        break;
    case 'liquid':
    case 'glide':
        formants = consonantTargets(segments, ix, voice);
        // A prevocalic liquid or glide is a constriction - quieter than the vowel; after the vowel it joins the nucleus
        frame.av = segments[ix + 1].type === 'vowel' ? voice.liquidAv : voice.codaLiquidAv;
        frame.ah = aspirationLevel(voice.breathDb, levels) * 0.6;
        frame.b2 = segment.ph === 'L' ? voice.lateralB2 : 110;
        frame.tauF = segment.ph === 'L' ? 8 : 14;
        // The lateral's side branch - an antiformant between F2 and F3, and an extra pole above it (on the nasal pair)
        if (segment.ph === 'L' && voice.lateralZero > 0) {
            frame.fnz = voice.lateralZero * voice.formantScale[2];
            frame.fnp = voice.lateralPole * voice.formantScale[2];
            frame.bnz = voice.lateralBw;
            frame.bnp = voice.lateralBw;
        }
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
        // Aspiration through the following vowel's formants - between vowels, breathy voicing (an abducted, still
        // vibrating glottis): a strong F1 region with weak upper formants
        frame.ah = aspirationLevel(voice.hhDb, levels);
        if (segments[ix - 1].type === 'vowel' && segments[ix + 1].type === 'vowel') {
            frame.av = 0.6;
            frame.rd = 2.2;
        }
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
            [frame.b2, frame.b3] = voice.fricVoiceBar;
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
    // A stop - the remaining segment type
    case 'stop':
    default: {
        formants = consonantTargets(segments, ix, voice);
        const closureEnd = segment.dur - segment.burst - segment.aspiration;
        const next = segments[ix + 1];
        const release = stopRelease[def.place];
        // The release noise spectrum by place - velar noise is compact, at the locus F2
        const burstSpectrum = def.place === 'velar'
            ? {...noiseSpectra.velar, 'fp1': formants[1] / voice.noiseScale} : noiseSpectra[def.place];
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
                const isLiquid = next.type === 'liquid' || next.type === 'glide';
                const moveMs = isLiquid ? Math.min(segment.aspiration, voice.clusterMoveMs) : segment.aspiration;
                const weight = Math.min(1, (local - closureEnd - segment.burst) / moveMs);
                formants = locus.map((value, kx) => value + (nextFormants[kx] - value) * weight);
            }
            frame.ah = aspirationLevel(voice.aspirationDb + (def.voiced ? -5 : 0), levels);
            // An alveolar stop before R is affricated - postalveolar frication fading through the devoiced R
            if (segment.cluster && next.ph === 'R' && def.place === 'alveolar') {
                const fade = 1 - (local - closureEnd - segment.burst) / segment.aspiration;
                const db = voice.trDb - noiseSpectra.SH.db + 10 * Math.log10(Math.max(0.05, fade));
                Object.assign(frame, noiseParams(noiseSpectra.SH, voice, levels, db));
            }
            // Voiced stops - voicing at once, with the glottis closed (narrow F1); voiceless - the glottis open (wide F1)
            frame.av = def.voiced ? 0.6 : 0;
            frame.b1 = def.voiced ? 120 : 250;
            frame.tauA = def.voiced ? 1 : 3;
        }
        frame.tauF = 8;
        break;
    }
    }

    // Nasalization spreading into vowels next to nasals
    if (segment.type === 'vowel') {
        const prev = segments[ix - 1];
        const next = segments[ix + 1];
        if ((next && next.type === 'nasal' && fraction > 0.6) || (prev && prev.type === 'nasal' && fraction < 0.25)) {
            frame.fnz = voice.nasalizationZero;
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


// The intonation (F0) contour of the segments - {f0, baseline}, an F0 and the phrase's declining baseline (the contour
// without its accents; the F0 itself outside phrases) for each frame time
function intonation(segments, voice, frameCount, random) {
    const points = [];
    const phrases = [];
    const baselines = [];
    let current = null;
    for (const segment of segments) {
        if (segment.type === 'pause') {
            if (!segment.minorPause) {
                current = null;
            }
        } else {
            if (current === null) {
                current = {'segments': [], 'end': segment.phraseEnd};
                phrases.push(current);
            }
            current.segments.push(segment);
        }
    }
    // The recently mentioned content words - a repeated (given) word is deaccented - by their content-word count
    const mentions = new Map();
    let mentionCount = 0;

    // The declination carried into a phrase from the phrases before it in its sentence (ms of the declination clock)
    let prevPhrase = null;
    for (const phrase of phrases) {
        const [{start}] = phrase.segments;
        const last = phrase.segments[phrase.segments.length - 1];
        const end = last.start + last.dur;
        const isContinued = prevPhrase !== null && prevPhrase.end === 'comma';
        const carry = isContinued ? (1 - voice.commaReset) * (prevPhrase.carry + prevPhrase.duration) : 0;
        prevPhrase = {'end': phrase.end, 'duration': end - start, carry};

        // The intermediate phrases' start and end times
        const minorStarts = [];
        const minorEnds = [];
        for (const segment of phrase.segments) {
            if (minorStarts[segment.minor] === undefined) {
                minorStarts[segment.minor] = segment.start;
            }
            minorEnds[segment.minor] = segment.start + segment.dur;
        }

        // The declining baseline, partly reset at each intermediate phrase, and continuing from a comma phrase before it
        const base = (time) => {
            let minor = 0;
            while (minor + 1 < minorStarts.length && minorStarts[minor + 1] <= time) {
                minor++;
            }
            const elapsed = carry + time - start - voice.declinationReset * (minorStarts[minor] - start);
            return voice.f0 * (voice.pitchBase - Math.min(voice.declinationMax, voice.declination * elapsed / 1000));
        };
        baselines.push({start, end, base});

        // The accented syllables - the primary-stressed vowels of content words, weaker on a given (recently mentioned) word
        const givenWords = new Set();
        for (const segment of phrase.segments) {
            if (segment.wordStart && !segment.isFunction) {
                const stem = accentStem(segment.wordText);
                const mention = mentions.get(stem);
                if (mention !== undefined && mentionCount - mention < voice.givenWords) {
                    givenWords.add(segment.word);
                }
                mentions.set(stem, ++mentionCount);
            }
        }
        const accents = [];
        for (const segment of phrase.segments) {
            if (segment.type === 'vowel' && segment.stress === 1 && !segment.isFunction && !unaccentedWords.has(segment.wordText)) {
                const isGiven = givenWords.has(segment.word);
                accents.push({'vowel': segment, 'strength': isGiven ? voice.givenAccent : 1, isGiven});
            }
        }
        if (accents.length === 0) {
            const vowel = phrase.segments.find((segment) => segment.type === 'vowel');
            if (vowel !== undefined) {
                accents.push({vowel, 'strength': 1, 'isGiven': false});
            }
        }

        // The nucleus falls on the last new word - given words after it are left unaccented
        const lastNew = accents.findLastIndex((accent) => !accent.isGiven);
        if (lastNew !== -1) {
            accents.splice(lastNew + 1);
        }

        // Accent clash - of two accents on adjacent words with no syllable between, the one that is neither the first nor
        // the last of its intermediate phrase is weakened
        const isFirst = (ix) => ix === 0 || accents[ix - 1].vowel.minor !== accents[ix].vowel.minor;
        const isLast = (ix) => ix === accents.length - 1 || accents[ix + 1].vowel.minor !== accents[ix].vowel.minor;
        const vowelIndex = (vowel) => phrase.segments.slice(0, phrase.segments.indexOf(vowel))
            .filter((segment) => segment.type === 'vowel').length;
        for (let ix = 1; ix < accents.length; ix++) {
            const prev = accents[ix - 1].vowel;
            const {vowel} = accents[ix];
            const between = phrase.segments.slice(phrase.segments.indexOf(prev) + 1, phrase.segments.indexOf(vowel));
            if (vowel.minor === prev.minor && vowel.word === prev.word + 1 && !between.some((segment) => segment.type === 'vowel') &&
                accents[ix - 1].strength === 1) {
                if (!isLast(ix)) {
                    accents[ix].strength = voice.clashAccent;
                } else if (!isFirst(ix - 1)) {
                    accents[ix - 1].strength = voice.clashAccent;
                }
            }
        }

        // Dense accent runs - an accent within accentSpacing syllables of the last full accent, neither the first nor the
        // last of its intermediate phrase, is weakened (readers alternate strong and weak accents)
        let lastFull = null;
        accents.forEach((accent, ix) => {
            const syllable = vowelIndex(accent.vowel);
            if (isFirst(ix)) {
                lastFull = null;
            }
            if (accent.strength === 1 && lastFull !== null && !isLast(ix) && syllable - lastFull < voice.accentSpacing) {
                accent.strength = voice.clashAccent;
            }
            if (accent.strength === 1) {
                lastFull = syllable;
            }
        });

        const isExclamation = phrase.end === 'exclamation';
        // The phrase onset - high at a sentence start, partly reset after a comma
        const onset = isContinued ? 1 + (voice.phraseOnset - 1) * voice.commaReset : voice.phraseOnset;
        points.push({'time': start, 'f0': base(start) * (phrase.end === 'question' ? 1.0 : onset)});
        let downstep = 0;
        let prevHeight = 0;
        accents.forEach(({vowel, strength}, ixAccent) => {
            const isNuclear = ixAccent === accents.length - 1;
            const isMinorFinal = !isNuclear && isLast(ixAccent);

            // Downstep within each intermediate phrase, to a floor - the phrase's first accent is the highest
            if (isFirst(ixAccent)) {
                downstep = ixAccent === 0 ? 1.15 : 1;
            } else if (strength === 1) {
                downstep = Math.max(voice.downstepFloor, downstep * voice.downstep);
            }
            const exclamation = isExclamation ? voice.exclamationAccent : 1;
            let height = voice.accent * exclamation * downstep * strength * (0.75 + 0.5 * random.uniform());
            if (isNuclear && phrase.end !== 'question') {
                height = Math.max(height, voice.accent * voice.nuclearFloor);
            }
            if (ixAccent > 0 && !isFirst(ixAccent)) {
                // The valley between accents - bridged (a shallow dip) when the accents are close
                const prevVowel = accents[ixAccent - 1].vowel;
                const gap = vowel.start - (prevVowel.start + prevVowel.dur);
                const valleyTime = (prevVowel.start + prevVowel.dur + vowel.start) / 2;
                const bridge = voice.accentBridge * Math.max(0, 1 - gap / 300) * Math.min(prevHeight, height);
                points.push({'time': valleyTime, 'f0': base(valleyTime) * (1 + bridge)});
            }
            prevHeight = height;
            if (isNuclear && phrase.end === 'question') {
                // Yes/no question - a low nuclear accent, then the rise
                const lowTime = vowel.start + 0.3 * vowel.dur;
                points.push({'time': lowTime, 'f0': base(lowTime) * 0.95});
                points.push({'time': end, 'f0': voice.f0 * voice.questionRise});
            } else if (isNuclear) {
                const peakTime = vowel.start + (vowel.phraseFinal ? 0.3 : 0.55) * vowel.dur;
                points.push({'time': vowel.start, 'f0': base(vowel.start) * (1 + height * 0.4)});
                points.push({'time': peakTime, 'f0': base(peakTime) * (1 + height)});
                if (phrase.end === 'comma') {
                    // Continuation - fall, then a slight rise
                    const lowTime = Math.max(peakTime + 60, end - 100);
                    points.push({'time': lowTime, 'f0': base(lowTime) * voice.continuationLow});
                    points.push({'time': end, 'f0': base(end) * voice.continuationHigh});
                } else {
                    const fallTime = Math.min(end - 20, vowel.start + vowel.dur);
                    points.push({'time': fallTime, 'f0': base(fallTime) * (0.5 * (1 + height) + 0.5 * voice.endLow)});
                    points.push({'time': end, 'f0': voice.f0 * voice.endLow});
                }
            } else {
                const peakTime = vowel.start + 0.6 * vowel.dur;
                points.push({'time': vowel.start, 'f0': base(vowel.start) * (1 + height * 0.5)});
                points.push({'time': peakTime, 'f0': base(peakTime) * (1 + height)});
                if (isMinorFinal) {
                    // An intermediate phrase end - the pitch held up (a high phrase accent) to the boundary
                    const minorEnd = minorEnds[vowel.minor];
                    points.push({'time': Math.max(peakTime + 40, minorEnd - 10), 'f0': base(peakTime) * (1 + height * voice.minorHold)});
                }
            }
        });
    }
    points.sort((first, second) => first.time - second.time);

    // The slopes at the points (log F0 per ms) - monotone cubic (Fritsch-Butland), level at the turning points and the
    // ends, scaled by contourFlow (0 - the contour comes to rest at every point)
    const logF0 = points.map((point) => Math.log(point.f0));
    const deltas = points.slice(1).map((point, ix) => (logF0[ix + 1] - logF0[ix]) / Math.max(frameMs, point.time - points[ix].time));
    const slopes = points.map((point, ix) => {
        if (ix === 0 || ix === points.length - 1 || deltas[ix - 1] * deltas[ix] <= 0) {
            return 0;
        }
        const spanBefore = point.time - points[ix - 1].time;
        const spanAfter = points[ix + 1].time - point.time;
        const weightBefore = 2 * spanAfter + spanBefore;
        const weightAfter = spanAfter + 2 * spanBefore;
        return voice.contourFlow * (weightBefore + weightAfter) / (weightBefore / deltas[ix - 1] + weightAfter / deltas[ix]);
    });

    // Interpolate the points (cubic Hermite in log F0), hold the ends
    const f0 = new Float64Array(frameCount);
    let ixPoint = 0;
    for (let ix = 0; ix < frameCount; ix++) {
        const time = ix * frameMs;
        while (ixPoint < points.length - 1 && points[ixPoint + 1].time <= time) {
            ixPoint++;
        }
        const pointA = points[ixPoint];
        const ixB = Math.min(points.length - 1, ixPoint + 1);
        const pointB = points[ixB];
        if (time <= pointA.time || pointA === pointB) {
            f0[ix] = pointA.f0;
        } else {
            const span = pointB.time - pointA.time;
            const fraction = (time - pointA.time) / span;
            const square = fraction * fraction;
            const cube = square * fraction;
            f0[ix] = Math.exp((2 * cube - 3 * square + 1) * logF0[ixPoint] + (cube - 2 * square + fraction) * span * slopes[ixPoint] +
                (3 * square - 2 * cube) * logF0[ixB] + (cube - square) * span * slopes[ixB]);
        }
    }
    const baseline = Float64Array.from(f0);
    for (const {start, end, base} of baselines) {
        for (let ix = Math.ceil(start / frameMs); ix * frameMs <= end; ix++) {
            baseline[ix] = base(ix * frameMs);
        }
    }
    return {f0, baseline};
}


// A smooth random process at unit RMS, one value per frame - Gaussian noise through two one-pole low-pass stages at a
// corner frequency (Hz), started in its steady state. Natural drift is aperiodic; a sum of sinusoids is heard as a vibrato
// or warble.
const smoothNoiseRms = new Map();
function smoothNoise(random, count, cornerHz) {
    const alpha = 1 - Math.exp(-2 * Math.PI * cornerHz * frameMs / 1000);
    if (!smoothNoiseRms.has(alpha)) {
        // The RMS of the two-stage filter's impulse response
        let stage1 = alpha;
        let stage2 = 0;
        let sum = 0;
        for (let ix = 0; ix < 40 / alpha; ix++) {
            stage2 += alpha * (stage1 - stage2);
            sum += stage2 * stage2;
            stage1 *= 1 - alpha;
        }
        smoothNoiseRms.set(alpha, Math.sqrt(sum));
    }
    const rms = smoothNoiseRms.get(alpha);
    const values = new Float64Array(count);
    let stage1 = 0;
    let stage2 = 0;
    for (let ix = -Math.ceil(6 / alpha); ix < count; ix++) {
        stage1 += alpha * (random.gauss() - stage1);
        stage2 += alpha * (stage1 - stage2);
        if (ix >= 0) {
            values[ix] = stage2 / rms;
        }
    }
    return values;
}


// A soft limiter - linear below 0.8, approaching 1 above
function softLimit(value) {
    const magnitude = Math.abs(value);
    if (magnitude <= 0.8) {
        return value;
    }
    return Math.sign(value) * (0.8 + 0.2 * Math.tanh((magnitude - 0.8) / 0.2));
}


// Synthesize text with a voice (a name or a spec - lib/voices.js) - returns {samples (Float32Array), sampleRate,
// segments, frames, rawPeak, vowelLevel, voice (the resolved parameters)}. The options are the random seed, the rate (a
// multiplier of the voice's), parameter overrides, and a speaking style (applyStyle). For streaming, onBlock(samples)
// receives each block as it is synthesized, at a fixed calibrated gain with a soft limiter (a whole utterance can't be
// normalized before it's finished).
export function speak(text, voiceSpec = 'male', {seed = 1, rate = 1, overrides = {}, style = null, onBlock = null} = {}) {
    const voice = resolveVoice(voiceSpec, {overrides, style, rate});
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

    // Loudness declination - the voicing weakens slowly through each phrase as the subglottal pressure falls
    let phraseStart = 0;
    for (let ix = 0; ix < frameCount; ix++) {
        const segment = segments[frames[ix].segment];
        if (segment.type === 'pause') {
            phraseStart = segment.start + segment.dur;
        } else {
            const elapsed = (ix * frameMs - phraseStart) / 1000;
            frames[ix].av *= 1 - Math.min(voice.levelDeclinationMax, voice.levelDeclination * elapsed);
        }
    }

    // Smoothing - formants by their segment's time constant, amplitudes quickly
    for (const key of ['f1', 'f2', 'f3']) {
        smooth(frames, key, 'tauF');
    }
    for (const key of ['b1', 'b2', 'b3', 'b4', 'b5', 'fnz', 'fnp', 'bnz', 'bnp']) {
        smooth(frames, key, 6);
    }
    for (const key of ['av', 'ah', 'af']) {
        smooth(frames, key, 'tauA');
    }
    smooth(frames, 'rd', 20);

    // Intonation, with microprosody (F0 raised after voiceless obstruents, lowered in voiced ones), and the voice's
    // aperiodic micro-variation - a slow wander and a faster flutter of F0, never a periodic modulation (a vibrato)
    const {f0, baseline} = intonation(segments, voice, frameCount, random);
    const pitchWander = smoothNoise(random, frameCount, voice.pitchDriftHz);
    const pitchFlutter = smoothNoise(random, frameCount, voice.flutterHz);
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
        const wander = 2 ** ((voice.pitchDrift * pitchWander[ix] + voice.flutter * pitchFlutter[ix]) / 12);
        // Creak - the pitch drops toward vocal fry
        const creakDrop = 1 - 0.35 * (frames[ix].creak || 0);
        frames[ix].f0 = f0[ix] * micro * wander * creakDrop;
        // Accents are louder - the sources' levels rise with the pitch above the declining baseline (a higher subglottal
        // pressure raises both)
        const loudness = 10 ** (voice.f0Loudness * Math.max(0, 12 * Math.log2(f0[ix] / baseline[ix])) / 20);
        frames[ix].av *= loudness;
        frames[ix].ah *= loudness;
        frames[ix].af *= loudness;
    }
    smooth(frames, 'f0', 12);

    // Micro-dynamics - slow, aperiodic drifts of the voice quality and small articulatory fluctuations of the formants
    const rdWander = smoothNoise(random, frameCount, voice.rdDriftHz);
    const formantWander = [1, 2, 3].map(() => smoothNoise(random, frameCount, voice.formantWobbleHz));
    for (let ix = 0; ix < frameCount; ix++) {
        frames[ix].rd *= 1 + voice.rdDrift * rdWander[ix];
        for (let kx = 1; kx <= 3; kx++) {
            frames[ix][`f${kx}`] *= 1 + voice.formantWobble * formantWander[kx - 1][ix];
        }
    }

    // The voice's fixed higher formants, open-phase F1 damping, and voicing (0 whispers)
    for (const frame of frames) {
        frame.av *= voice.voicing;
        frame.ob1 = voice.openB1;
        frame.f4 = voice.f4;
        frame.f5 = voice.f5;
        frame.f6 = voice.f6;
    }
    // The vocal tract length - every resonance scaled
    const {tractScale} = voice;
    if (tractScale !== 1) {
        for (const frame of frames) {
            for (const key of ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'fnp', 'fnz', 'fp1', 'fp2', 'fhp']) {
                if (frame[key] !== undefined) {
                    frame[key] *= tractScale;
                }
            }
        }
    }
    const higherFormants = voice.higherFormants.map(([frequency, bandwidth]) => [frequency * tractScale, bandwidth]);
    // The streaming gain - headroom of 4 times the calibrated vowel level for the louder accents (f0Loudness on the full
    // accent range), so the soft limiter touches only the rare peak
    const streamGain = 0.89 / (4 * levels.vowel);
    const streamBlock = onBlock === null ? null : (block) => {
        const scaled = new Float32Array(block.length);
        for (let ix = 0; ix < block.length; ix++) {
            scaled[ix] = softLimit(block[ix] * streamGain);
        }
        onBlock(scaled);
    };
    const samples = synthesize(frames, {frameMs, sampleRate, seed, 'jitter': voice.jitter, 'shimmer': voice.shimmer,
        'noiseModulation': voice.noiseModulation, 'sourceShelfHz': voice.shelfHz, 'sourceShelfDb': voice.shelfDb,
        'noiseShelfDb': voice.noiseShelfDb, higherFormants,
        'onBlock': streamBlock});

    let rawPeak = 0;
    for (const sample of samples) {
        rawPeak = Math.max(rawPeak, Math.abs(sample));
    }
    return {'samples': normalize(samples, 0.89), sampleRate, segments, frames, rawPeak, 'vowelLevel': levels.vowel, voice};
}
