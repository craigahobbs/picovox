// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {baseVoice, resolveVoice, voices as voiceSpecs} from '../lib/voices.js';
import {speak, voices} from '../lib/speak.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


const frameMs = 5;


// The resolved voices
const maleVoice = resolveVoice('male');
const femaleVoice = resolveVoice('female');


// The non-pause segments of a word
function wordSegments(result, wordText, occurrence = 0) {
    const starts = result.segments.filter((segment) => segment.wordText === wordText && segment.wordStart);
    const start = starts[occurrence];
    const ixStart = result.segments.indexOf(start);
    const segments = [];
    for (let ix = ixStart; ix < result.segments.length; ix++) {
        const segment = result.segments[ix];
        if (segment.type === 'pause' || (ix > ixStart && segment.wordStart)) {
            break;
        }
        segments.push(segment);
    }
    return segments;
}


// A word's segment of a phoneme
function wordSegment(result, wordText, ph, occurrence = 0) {
    return wordSegments(result, wordText, occurrence).find((segment) => segment.ph === ph);
}


// The frames within a segment
function segmentFrames(result, segment) {
    const ixSegment = result.segments.indexOf(segment);
    return result.frames.filter((frame) => frame.segment === ixSegment);
}


// The frame at the middle of a segment
function middleFrame(result, segment) {
    return result.frames[Math.round((segment.start + segment.dur / 2) / frameMs)];
}


// A frame parameter at a time (ms)
function frameValue(result, key, time) {
    return result.frames[Math.round(time / frameMs)][key];
}


// The F0 at a time (ms)
function f0At(result, time) {
    return frameValue(result, 'f0', time);
}


// The vowels of a result
function vowelSegments(result) {
    return result.segments.filter((segment) => segment.type === 'vowel');
}


// The mean F0 of the voiced frames
function meanF0(result) {
    const voiced = result.frames.filter((frame) => frame.av > 0.3);
    return voiced.reduce((sum, frame) => sum + frame.f0, 0) / voiced.length;
}


// The peak absolute sample
function peak(samples) {
    let max = 0;
    for (const sample of samples) {
        max = Math.max(max, Math.abs(sample));
    }
    return max;
}


test('speak.voices', () => {
    // The voice specs, re-exported
    assert.equal(voices, voiceSpecs);
});


test('speak, result', () => {
    const result = speak('The birch canoe slid on the smooth planks.');
    assert.equal(result.sampleRate, 44100);
    assert(result.samples instanceof Float32Array);

    // Normalized to 0.89 peak
    assert(Math.abs(peak(result.samples) - 0.89) < 1e-6);
    assert(result.rawPeak > 0);
    assert(result.vowelLevel > 0);

    // The segments are contiguous, starting and ending with a pause
    const {segments} = result;
    assert.equal(segments[0].type, 'pause');
    assert.equal(segments[0].start, 0);
    assert.equal(segments[0].dur, baseVoice.leadingPause);
    assert.equal(segments[segments.length - 1].type, 'pause');
    assert.equal(segments[segments.length - 1].dur, baseVoice.finalPause);
    for (let ix = 1; ix < segments.length; ix++) {
        assert(Math.abs(segments[ix].start - (segments[ix - 1].start + segments[ix - 1].dur)) < 1e-9);
        assert(segments[ix].dur > 0);
    }
    assert.deepEqual(
        segments.filter((segment) => segment.type !== 'pause').map((segment) => segment.ph).join(' '),
        'DH AH B ER CH K AH N UW S L IH D AA N DH AH S M UW DH P L AE NG K S'
    );

    // The samples and frames span the segments
    const last = segments[segments.length - 1];
    const totalMs = last.start + last.dur;
    assert.equal(result.frames.length, Math.ceil(totalMs / frameMs) + 1);
    assert(Math.abs(result.samples.length / result.sampleRate * 1000 - (result.frames.length - 1) * frameMs) < 1);

    // Every frame is finite
    for (const frame of result.frames) {
        for (const key of ['f0', 'f1', 'f2', 'f3', 'av', 'ah', 'af', 'rd']) {
            assert(Number.isFinite(frame[key]), key);
        }
        assert(frame.f1 < frame.f2 && frame.f2 < frame.f3);
    }
});


test('speak, determinism by seed', () => {
    const text = 'Did you remember the coffee?';
    const first = speak(text, 'male', {'seed': 3});
    const second = speak(text, 'male', {'seed': 3});
    const third = speak(text, 'male', {'seed': 4});
    assert.deepEqual(first.samples, second.samples);
    assert.equal(third.samples.length, first.samples.length);
    assert.notDeepEqual(third.samples, first.samples);
    assert.notDeepEqual(third.frames.map((frame) => frame.f0), first.frames.map((frame) => frame.f0));

    // The segments do not depend on the seed
    assert.deepEqual(third.segments, first.segments);
});


test('speak, female and male', () => {
    const text = 'We were away a year ago.';
    const male = speak(text, 'male');
    const female = speak(text, 'female');
    const default_ = speak(text);
    assert.deepEqual(default_.samples, male.samples);

    // The same timing
    assert.deepEqual(female.segments.map((segment) => segment.dur), male.segments.map((segment) => segment.dur));

    // Higher pitch and higher formants
    assert(meanF0(male) > 80 && meanF0(male) < 140);
    assert(meanF0(female) > 1.4 * meanF0(male));
    const maleVowel = middleFrame(male, wordSegment(male, 'away', 'EY'));
    const femaleVowel = middleFrame(female, wordSegment(female, 'away', 'EY'));
    assert(femaleVowel.f2 > maleVowel.f2);
    assert(femaleVowel.f3 > maleVowel.f3);
    assert.equal(male.frames[0].f4, maleVoice.f4);
    assert.equal(female.frames[0].f4, femaleVoice.f4);

    // The male voice creaks at the end of a statement, the female voice does not
    assert(Math.max(...male.frames.map((frame) => frame.creak || 0)) > 0.3);
    assert.equal(Math.max(...female.frames.map((frame) => frame.creak || 0)), 0);
});


test('speak, statement intonation', () => {
    const result = speak('We were away a year ago.');
    const vowels = vowelSegments(result);
    const [first] = vowels;
    const last = vowels[vowels.length - 1];

    // A statement falls - it ends well below its start and below the voice's base pitch
    const startF0 = f0At(result, first.start + 10);
    const endF0 = f0At(result, last.start + last.dur - 10);
    assert(endF0 < 0.85 * startF0);
    assert(endF0 < maleVoice.f0);

    // The accent peaks rise above the start
    const maxF0 = Math.max(...result.frames.map((frame) => frame.f0));
    assert(maxF0 > startF0 * 1.1);

    // The phrase ends with creak
    const lastFrame = result.frames[Math.round((last.start + last.dur - 10) / frameMs)];
    assert(lastFrame.creak > 0);
});


test('speak, yes/no question intonation', () => {
    const result = speak('Were you away a year ago?');
    const vowels = vowelSegments(result);
    const [first] = vowels;
    const last = vowels[vowels.length - 1];
    assert.equal(last.phraseEnd, 'question');

    // A yes/no question rises - it ends well above its start
    const startF0 = f0At(result, first.start + 10);
    const endF0 = f0At(result, last.start + last.dur - 10);
    assert(endF0 > 1.2 * startF0);
    assert(endF0 > maleVoice.f0 * 1.2);

    // The low nuclear accent - the rise starts low
    assert(f0At(result, last.start + 0.3 * last.dur) < endF0 * 0.85);

    // No creak in a question
    assert.equal(Math.max(...result.frames.map((frame) => frame.creak || 0)), 0);
});


test('speak, wh-question intonation', () => {
    const result = speak('Where were you a year ago?');
    const vowels = vowelSegments(result);
    const [first] = vowels;
    const last = vowels[vowels.length - 1];
    assert.equal(last.phraseEnd, 'wh');

    // A wh-question falls like a statement, with creak
    const startF0 = f0At(result, first.start + 10);
    const endF0 = f0At(result, last.start + last.dur - 10);
    assert(endF0 < 0.85 * startF0);
    assert(Math.max(...result.frames.map((frame) => frame.creak || 0)) > 0);
});


test('speak, exclamation intonation', () => {
    const statement = speak('We were away a year ago.');
    const exclamation = speak('We were away a year ago!');
    assert.equal(vowelSegments(exclamation)[0].phraseEnd, 'exclamation');

    // Higher accents, the same timing
    assert.deepEqual(exclamation.segments.map((segment) => segment.dur), statement.segments.map((segment) => segment.dur));
    const statementMax = Math.max(...statement.frames.map((frame) => frame.f0));
    const exclamationMax = Math.max(...exclamation.frames.map((frame) => frame.f0));
    assert(exclamationMax > statementMax * 1.05);
});


test('speak, continuation intonation', () => {
    const result = speak('When we were away, we ran.');
    const vowels = vowelSegments(result);
    const commaVowels = vowels.filter((vowel) => vowel.phraseEnd === 'comma');
    const commaLast = commaVowels[commaVowels.length - 1];
    const commaEnd = commaLast.start + commaLast.dur;

    // The comma pause - between short phrases (seven syllables), shorter than the full comma pause
    const ixPause = result.segments.indexOf(commaLast) + 1;
    assert.equal(result.segments[ixPause].type, 'pause');
    const commaPause = baseVoice.commaPauseMin + (baseVoice.commaPause - baseVoice.commaPauseMin) * 7 / baseVoice.commaPauseSyllables;
    assert(Math.abs(result.segments[ixPause].dur - commaPause) < 1e-9);

    // The continuation - a fall, then a slight rise, ending above the statement's end
    const lowF0 = Math.min(...segmentFrames(result, commaLast).map((frame) => frame.f0));
    const commaEndF0 = f0At(result, commaEnd - 5);
    assert(commaEndF0 > lowF0);
    const lastSegment = result.segments[result.segments.length - 2];
    assert(commaEndF0 > f0At(result, lastSegment.start + lastSegment.dur - 5) * 1.1);

    // No creak at a comma
    for (const frame of segmentFrames(result, commaLast)) {
        assert(!(frame.creak > 0));
    }
});


test('speak, comma pauses', () => {
    // The pause after a comma grows with the syllables of the phrases on either side, up to the full comma pause
    const commaPause = (result) => result.segments.filter((segment) => segment.type === 'pause')[1].dur;
    const short = speak('Yes, we ran.');
    const longText = 'The old professor walked slowly through the garden, and talked about the history of the ancient city.';
    const long = speak(longText);
    assert(commaPause(short) < commaPause(long));
    assert(commaPause(short) > baseVoice.commaPauseMin);
    assert.equal(commaPause(speak(longText, 'male', {'overrides': {'commaPauseSyllables': 20}})), baseVoice.commaPause);

    // A semicolon or colon is a major break
    assert.equal(commaPause(speak('Yes; we ran.')), baseVoice.majorPause);
    assert.equal(commaPause(speak('Yes: we ran.')), baseVoice.majorPause);

    // Overridden
    assert.equal(commaPause(speak('Yes, we ran.', 'male', {'overrides': {'commaPause': 300, 'commaPauseMin': 300}})), 300);
    assert.equal(commaPause(speak('Yes; we ran.', 'male', {'overrides': {'majorPause': 700}})), 700);
});


test('speak, leading and final pauses', () => {
    const result = speak('We ran.', 'male', {'overrides': {'leadingPause': 100, 'finalPause': 600}});
    const pauses = result.segments.filter((segment) => segment.type === 'pause');
    assert.deepEqual(pauses.map((pause) => pause.dur), [100, 600]);
});


test('speak, sentence pause', () => {
    const result = speak('We ran. We ran.');
    const pauses = result.segments.filter((segment) => segment.type === 'pause');
    assert.deepEqual(pauses.map((pause) => pause.dur), [baseVoice.leadingPause, baseVoice.sentencePause, baseVoice.finalPause]);

    // Overridden
    const overridden = speak('We ran. We ran.', 'male', {'overrides': {'sentencePause': 500}});
    assert.deepEqual(
        overridden.segments.filter((segment) => segment.type === 'pause').map((pause) => pause.dur),
        [baseVoice.leadingPause, 500, baseVoice.finalPause]
    );
});


test('speak, rate', () => {
    const normal = speak('We were away a year ago.');
    const fast = speak('We were away a year ago.', 'male', {'rate': 2});
    const normalVowel = wordSegment(normal, 'away', 'EY');
    const fastVowel = wordSegment(fast, 'away', 'EY');
    assert(Math.abs(fastVowel.dur - normalVowel.dur / 2) < 1e-9);
    assert(fast.samples.length < normal.samples.length * 0.75);
});


test('speak, phrase-final lengthening', () => {
    const result = speak('Bob saw Bob.');
    const firstVowel = wordSegment(result, 'bob', 'AA', 0);
    const lastVowel = wordSegment(result, 'bob', 'AA', 1);
    assert(!firstVowel.phraseFinal);
    assert(lastVowel.phraseFinal);
    assert(lastVowel.dur > 1.6 * firstVowel.dur);

    // The final consonant's closure lengthens too
    const closure = (segment) => segment.dur - segment.burst - segment.aspiration;
    const finalB = wordSegments(result, 'bob', 1).pop();
    assert(finalB.phraseFinal);
    assert(closure(finalB) > closure(wordSegments(result, 'bob', 0).pop()));
});


test('speak, vowel durations by context', () => {
    // Before a voiced stop, a vowel is longer than before a voiceless one
    const voicedStop = speak('bead.');
    const voicelessStop = speak('beat.');
    assert(wordSegment(voicedStop, 'bead', 'IY').dur > 1.4 * wordSegment(voicelessStop, 'beat', 'IY').dur);

    // Likewise before fricatives
    const voicedFricative = speak('peas.');
    const voicelessFricative = speak('peace.');
    assert(wordSegment(voicedFricative, 'peas', 'IY').dur > 1.5 * wordSegment(voicelessFricative, 'peace', 'IY').dur);

    // Before a nasal, a vowel is shorter than in an open syllable
    const nasal = speak('bean.');
    const open = speak('bee.');
    assert(wordSegment(nasal, 'bean', 'IY').dur < wordSegment(open, 'bee', 'IY').dur);

    // Before a vowel, a vowel is longer
    const hiatus = speak('the idea of it');
    assert(wordSegment(hiatus, 'idea', 'IY').dur > 0);

    // Unstressed vowels are shorter, function word vowels shorter still
    const result = speak('a banana to the cabana');
    const stressed = wordSegments(result, 'banana').filter((segment) => segment.type === 'vowel');
    assert(stressed[1].stress === 1 && stressed[0].stress === 0);
    assert(stressed[0].dur < stressed[1].dur / 2);
    const functionWord = speak('to two men');
    assert(wordSegment(functionWord, 'to', 'UW').dur < wordSegment(functionWord, 'two', 'UW').dur * 0.7);
});


test('speak, consonant durations', () => {
    // Consonants in a cluster are shorter
    const single = speak('sip it');
    const cluster = speak('slip it');
    assert(wordSegment(cluster, 'slip', 'S').dur < wordSegment(single, 'sip', 'S').dur);

    // Word-medial consonants in unstressed syllables are shorter
    const result = speak('the mama of mine');
    const [, medialM] = wordSegments(result, 'mama').filter((segment) => segment.ph === 'M');
    assert(medialM.syllableStress === 0 && !medialM.wordStart);
    assert(medialM.dur < wordSegment(result, 'mine', 'M').dur);
});


test('speak, stop phases', () => {
    const result = speak('The pie spy apt to bag a cat.');
    const {rate} = maleVoice;

    // An aspirated voiceless stop before a stressed vowel - a burst and a long voice onset time
    const pieP = wordSegment(result, 'pie', 'P');
    assert.equal(pieP.burst, 6);
    assert(Math.abs(pieP.aspiration - 55 * 0.85 / rate) < 1e-9);

    // After S, unaspirated
    const spyP = wordSegment(result, 'spy', 'P');
    assert(Math.abs(spyP.aspiration - 8 * 0.85 / rate) < 1e-9);

    // Before a stop, unreleased
    const aptP = wordSegment(result, 'apt', 'P');
    assert.equal(aptP.burst, 0);
    assert.equal(aptP.aspiration, 0);

    // A voiced stop - a shorter burst, a brief voice onset after it before a vowel
    const bagB = wordSegment(result, 'bag', 'B');
    assert(Math.abs(bagB.burst - 6 * 0.67) < 1e-9);
    assert(Math.abs(bagB.aspiration - 15 / rate) < 1e-9);

    // Velar bursts are the longest
    const catK = wordSegment(result, 'cat', 'K');
    assert.equal(catK.burst, 14);

    // A voiceless stop before a pause - released with a short aspiration
    const catT = wordSegment(result, 'cat', 'T');
    assert.equal(catT.burst, 9);
    assert(Math.abs(catT.aspiration - 20 / rate) < 1e-9);

    // The closure is silent, the burst is noise, the aspiration is breathy
    const pieFrames = segmentFrames(result, pieP);
    const closureEnd = pieP.start + pieP.dur - pieP.burst - pieP.aspiration;
    const closureFrames = pieFrames.filter((frame) => result.frames.indexOf(frame) * frameMs < closureEnd - 10);
    assert(closureFrames.slice(2).every((frame) => frame.av < 0.05));
    assert(pieFrames.some((frame) => frame.af > 0));
    assert(pieFrames.some((frame) => frame.ah > 0));

    // A voiced stop closure has a voice bar
    const bagFrames = segmentFrames(result, bagB);
    assert(bagFrames.some((frame) => frame.av > 0.1));
});


test('speak, stop releases before consonants', () => {
    const result = speak('The cats bad apple, it lay at last, bad.');
    const {rate} = maleVoice;

    // Before a fricative
    const catsT = wordSegment(result, 'cats', 'T');
    assert(Math.abs(catsT.aspiration - 20 / rate) < 1e-9);

    // Before a liquid in another word
    const itT = wordSegment(result, 'it', 'T');
    assert(Math.abs(itT.aspiration - 30 / rate) < 1e-9);

    // A voiced stop before a vowel, and before a pause
    const badD = wordSegment(result, 'bad', 'D', 0);
    assert(Math.abs(badD.aspiration - 15 / rate) < 1e-9);
    const finalD = wordSegment(result, 'bad', 'D', 1);
    assert.equal(finalD.aspiration, 0);
    assert(finalD.burst > 0);
});


test('speak, flapping', () => {
    const result = speak('The butter and the ladder, a lot of water.');
    const {rate} = maleVoice;

    // An alveolar stop between a vowel and an unstressed vowel is a flap
    for (const [word, ph] of [['butter', 'T'], ['ladder', 'D'], ['lot', 'T'], ['water', 'T']]) {
        const segment = wordSegment(result, word, ph);
        assert.equal(segment.type, 'flap', word);
        assert(Math.abs(segment.dur - 28 / rate) < 1e-9, word);
    }

    // A voiceless flap is partly devoiced, with a faint burst
    const butterFrames = segmentFrames(result, wordSegment(result, 'butter', 'T'));
    const ladderFrames = segmentFrames(result, wordSegment(result, 'ladder', 'D'));
    assert(Math.min(...butterFrames.map((frame) => frame.av)) < Math.min(...ladderFrames.map((frame) => frame.av)));
    assert(Math.max(...butterFrames.map((frame) => frame.af)) > Math.max(...ladderFrames.map((frame) => frame.af)));

    // Not before a stressed vowel
    const attack = speak('attack');
    assert.equal(wordSegment(attack, 'attack', 'T').type, 'stop');
});


test('speak, clusters', () => {
    const plain = speak('the lay');
    const result = speak('the play, a tree.');
    const {rate} = maleVoice;

    // A voiceless stop before a liquid in its word - a long voice onset time devoicing the liquid, which is shortened
    const playP = wordSegment(result, 'play', 'P');
    assert(playP.cluster);
    assert(Math.abs(playP.aspiration - maleVoice.clusterVot * 0.85 / rate) < 1e-9);
    const playL = wordSegment(result, 'play', 'L');
    const layL = wordSegment(plain, 'lay', 'L');
    assert(playL.dur < layL.dur * 0.5);

    // TR is affricated - postalveolar frication through the aspiration
    const treeT = wordSegment(result, 'tree', 'T');
    assert(treeT.cluster);
    const releaseStart = treeT.start + treeT.dur - treeT.aspiration;
    const releaseFrames = segmentFrames(result, treeT).filter((frame) => result.frames.indexOf(frame) * frameMs > releaseStart);
    assert(releaseFrames.some((frame) => frame.af > 0));

    // The cluster move - shorter with a longer move time override
    const slow = speak('the play, a tree.', 'male', {'overrides': {'clusterMoveMs': 60}});
    assert.notDeepEqual(slow.frames.map((frame) => frame.f2), result.frames.map((frame) => frame.f2));
});


test('speak, nasals', () => {
    const result = speak('No man ran.');

    // The nasal murmur - the nasal zero
    const manM = wordSegment(result, 'man', 'M');
    assert(Math.abs(middleFrame(result, manM).fnz - 450) < 30);

    // Nasalized vowel next to a nasal
    const manAE = wordSegment(result, 'man', 'AE');
    const vowelFrames = segmentFrames(result, manAE);
    assert(vowelFrames[0].fnz > 340);
    assert(vowelFrames[vowelFrames.length - 1].fnz > 340);
    assert(middleFrame(result, manAE).fnz < 300);

    // The nasal F1 isn't scaled for the female voice
    const female = speak('No man ran.', 'female');
    const femaleM = wordSegment(female, 'man', 'M');
    assert(Math.abs(middleFrame(female, femaleM).f1 - middleFrame(result, manM).f1) < 0.2 * middleFrame(result, manM).f1);

    // The pause before an initial nasal and after a final one hold the murmur's resonances
    assert(Math.abs(result.frames[30].fnz - 450) < 10);
    const finalPause = result.segments[result.segments.length - 1];
    assert(Math.abs(frameValue(result, 'fnz', finalPause.start + 40) - 450) < 10);
});


test('speak, laterals', () => {
    const result = speak('a lay');
    const layL = wordSegment(result, 'lay', 'L');
    const frame = middleFrame(result, layL);
    assert(Math.abs(frame.fnz - maleVoice.lateralZero) < 300);
    assert(Math.abs(frame.fnp - maleVoice.lateralPole) < 300);

    // A female lateral is scaled
    const female = speak('a lay', 'female');
    assert(middleFrame(female, wordSegment(female, 'lay', 'L')).fnz > frame.fnz);

    // Without the lateral branch
    const noLateral = speak('a lay', 'male', {'overrides': {'lateralZero': 0}});
    assert(middleFrame(noLateral, wordSegment(noLateral, 'lay', 'L')).fnz < 500);

    // A coda liquid joins the nucleus - louder than a prevocalic one
    const coda = speak('the ball game');
    const codaL = wordSegment(coda, 'ball', 'L');
    assert(middleFrame(coda, codaL).av > frame.av);
});


test('speak, glides and aspirates', () => {
    const result = speak('The way ahead, you go home.');

    // HH between vowels - breathy voicing
    const aheadHH = wordSegment(result, 'ahead', 'HH');
    assert(middleFrame(result, aheadHH).av > 0.2);
    assert(middleFrame(result, aheadHH).rd > maleVoice.rd * 1.5);

    // HH after a pause - voiceless
    const home = speak('home');
    assert(middleFrame(home, wordSegment(home, 'home', 'HH')).av < 0.1);

    // Y and W glides have their loci
    assert(middleFrame(result, wordSegment(result, 'you', 'Y')).f2 > 1800);
    assert(middleFrame(result, wordSegment(result, 'way', 'W')).f2 < 1200);
});


test('speak, fricatives and affricates', () => {
    const result = speak('See the zoo, judge the church, buzz.');

    // A voiceless fricative is noise only
    const seeS = wordSegment(result, 'see', 'S');
    assert(middleFrame(result, seeS).af > 0);
    assert(middleFrame(result, seeS).av < 0.05);

    // A voiceless fricative after a pause rises gradually
    const seeFrames = segmentFrames(result, seeS);
    assert(seeFrames[1].af < middleFrame(result, seeS).af);

    // A voiced fricative is voiced and noise
    const zooZ = wordSegment(result, 'zoo', 'Z');
    assert(middleFrame(result, zooZ).av > 0.1);
    assert(middleFrame(result, zooZ).af > 0);

    // A voiced fricative before a pause devoices and fades
    const buzzZ = wordSegment(result, 'buzz', 'Z');
    const buzzFrames = segmentFrames(result, buzzZ);
    assert(buzzFrames[buzzFrames.length - 1].af < middleFrame(result, buzzZ).af);

    // Affricates - closure, then frication
    for (const [word, ph] of [['judge', 'JH'], ['church', 'CH']]) {
        const segment = wordSegment(result, word, ph);
        const frames = segmentFrames(result, segment);
        assert(frames[frames.length - 2].af > frames[1].af, word);
    }
});


test('speak, velars', () => {
    // Before a front vowel, a velar's F2 is high; before a back vowel, lower
    const result = speak('a key, a coo, a go.');
    const keyK = wordSegment(result, 'key', 'K');
    const cooK = wordSegment(result, 'coo', 'K');
    const releaseF2 = (segment) => frameValue(result, 'f2', segment.start + segment.dur - segment.aspiration - 2);
    assert(releaseF2(keyK) > releaseF2(cooK) + 300);
    assert(wordSegment(result, 'go', 'G').burst > 0);
});


test('speak, coronal fronting', () => {
    // A back rounded vowel after a coronal starts fronted (F2 high) and glides back; after a labial it doesn't
    const text = 'a boot, a toot.';
    const result = speak(text);
    const bootUW = wordSegment(result, 'boot', 'UW');
    const tootUW = wordSegment(result, 'toot', 'UW');
    assert(!bootUW.coronalOnset && tootUW.coronalOnset);
    const f2At = (res, segment, fraction) => frameValue(res, 'f2', segment.start + fraction * segment.dur);
    assert(f2At(result, tootUW, 0.3) > f2At(result, bootUW, 0.3) + 300);
    assert(f2At(result, tootUW, 0.3) > f2At(result, tootUW, 0.75) + 200);

    // Without fronting, the vowel's F2 holds near its target
    const unfronted = speak(text, 'male', {'overrides': {'coronalFronting': 0}});
    const unfrontedUW = wordSegment(unfronted, 'toot', 'UW');
    assert(f2At(unfronted, unfrontedUW, 0.3) < f2At(result, tootUW, 0.3) - 300);

    // Liquids don't front a following vowel
    assert(!wordSegment(speak('a loot.'), 'loot', 'UW').coronalOnset);
});


test('speak, diphthong glide timing', () => {
    // A wide diphthong holds its nucleus before gliding late; a narrow one glides early
    const result = speak('a bide, a bayed.');
    const progress = (segment, fraction) => {
        const [[startF1], [endF1]] = maleVoice.vowelTargets[segment.ph];
        return (startF1 - frameValue(result, 'f1', segment.start + fraction * segment.dur)) / (startF1 - endF1);
    };
    assert(progress(wordSegment(result, 'bide', 'AY'), 0.45) < progress(wordSegment(result, 'bayed', 'EY'), 0.45));
});


test('speak, non-sibilant frication', () => {
    // The non-sibilants' flat spectra scale by nonSibilantScale, the sibilants' by noiseScale
    const text = 'a fuss.';
    const result = speak(text, 'female');
    const fFrame = middleFrame(result, wordSegment(result, 'fuss', 'F'));
    const sFrame = middleFrame(result, wordSegment(result, 'fuss', 'S'));
    const scaled = speak(text, 'female', {'overrides': {'nonSibilantScale': 1.1}});
    assert(Math.abs(middleFrame(scaled, wordSegment(scaled, 'fuss', 'F')).fp1 - fFrame.fp1 * 1.1) < 1);
    assert.equal(middleFrame(scaled, wordSegment(scaled, 'fuss', 'S')).fp1, sFrame.fp1);
    assert(femaleVoice.noiseScale !== femaleVoice.nonSibilantScale);
});


test('speak, vowel reduction', () => {
    // An unstressed vowel moves toward schwa - a word-final IY much less so ("happy")
    const result = speak('the happy city of it');
    const happyIY = wordSegment(result, 'happy', 'IY');
    assert(happyIY.stress === 0 && happyIY.wordEnd);
    assert(middleFrame(result, happyIY).f2 > 1900);

    // A phrase-initial unstressed vowel is full
    const initial = speak('Above it.');
    assert(vowelSegments(initial)[0].phraseInitial);

    // Secondary stress
    const understand = speak('understand.');
    assert.equal(wordSegment(understand, 'understand', 'AH').stress, 2);
});


test('speak, glottal attack', () => {
    const result = speak('Apple.');
    const [vowel] = vowelSegments(result);
    const frame = result.frames[Math.ceil(vowel.start / frameMs) + 1];
    assert.equal(frame.jit, 6);
    assert.equal(frame.shim, 4);

    // Disabled
    const noAttack = speak('Apple.', 'male', {'overrides': {'glottalAttack': false}});
    assert.equal(noAttack.frames[Math.ceil(vowel.start / frameMs) + 1].jit, undefined);
});


test('speak, intermediate phrases', () => {
    const text = 'The old professor walked slowly through the garden and talked about the history of the ancient city ' +
        'while the students listened carefully to every word.';
    const result = speak(text);
    const minors = new Set(result.segments.filter((segment) => segment.type !== 'pause').map((segment) => segment.minor));
    assert(minors.size >= 3);

    // Each intermediate phrase begins with a break word
    const breakWords = new Set();
    for (let ix = 1; ix < result.segments.length; ix++) {
        const segment = result.segments[ix];
        const prev = result.segments[ix - 1];
        if (segment.type !== 'pause' && prev.type !== 'pause' && segment.minor !== prev.minor) {
            assert(segment.wordStart);
            breakWords.add(segment.wordText);
        }
    }
    assert([...breakWords].some((word) => ['and', 'while'].includes(word)));

    // The intermediate-phrase-final syllables are marked, and only one phrase-final syllable
    assert(result.segments.some((segment) => segment.minorFinal));
    assert(!result.segments.some((segment) => segment.minorFinal && segment.phraseFinal));

    // Lengthened with an override
    const lengthened = speak(text, 'male', {'overrides': {'minorLengthen': 1.6}});
    const minorFinalVowel = result.segments.find((segment) => segment.minorFinal && segment.type === 'vowel');
    const ixVowel = result.segments.indexOf(minorFinalVowel);
    assert(lengthened.segments[ixVowel].dur > minorFinalVowel.dur * 1.15);

    // No pause at an intermediate phrase break by default; with minorPause, a brief one, through which the phrase's
    // intonation continues (no phrase-final fall before it)
    assert.equal(result.segments.filter((segment) => segment.type === 'pause').length, 2);
    const paused = speak(text, 'male', {'overrides': {'minorPause': 150}});
    const minorPauses = paused.segments.filter((segment) => segment.minorPause);
    assert(minorPauses.length >= 2);
    assert(minorPauses.every((pause) => pause.type === 'pause' && pause.dur === 150));
    const beforePause = paused.segments[paused.segments.indexOf(minorPauses[0]) - 1];
    assert(!beforePause.phraseFinal);
    assert(f0At(paused, beforePause.start + beforePause.dur - 5) > paused.voice.f0 * paused.voice.endLow * 1.1);

    // A long phrase without break words isn't split
    const unbroken = speak('Seventeen elephants happily devoured magnificent watermelons yesterday.');
    assert(unbroken.segments.every((segment) => segment.type === 'pause' || segment.minor === 0));

    // A short phrase isn't split
    const short = speak('The cat sat and the dog ran.');
    assert(short.segments.every((segment) => segment.type === 'pause' || segment.minor === 0));
});


test('speak, accents', () => {
    // The peak F0 of a word's vowel
    const wordPeak = (result, word, ph) => Math.max(...segmentFrames(result, wordSegment(result, word, ph)).map((frame) => frame.f0));

    // Accent clash - of adjacent accented words, the later is weakened when it isn't the last
    const clashText = 'Big black dogs bark loudly.';
    const clash = speak(clashText);
    const noClash = speak(clashText, 'male', {'overrides': {'clashAccent': 1}});
    assert(wordPeak(clash, 'black', 'AE') < wordPeak(noClash, 'black', 'AE'));
    assert(Math.abs(wordPeak(clash, 'big', 'IH') - wordPeak(noClash, 'big', 'IH')) < 1);

    // Accent clash at the phrase end - the earlier is weakened
    const clashEndText = 'Beautiful green trees.';
    const clashEnd = speak(clashEndText);
    const noClashEnd = speak(clashEndText, 'male', {'overrides': {'clashAccent': 1}});
    assert(wordPeak(clashEnd, 'green', 'IY') < wordPeak(noClashEnd, 'green', 'IY'));

    // Dense accent runs - an interior accent close after a full accent is weakened
    const denseText = 'Tall men eat red apples daily.';
    const dense = speak(denseText);
    const sparse = speak(denseText, 'male', {'overrides': {'accentSpacing': 1}});
    assert(wordPeak(dense, 'eat', 'IY') < wordPeak(sparse, 'eat', 'IY'));

    // A given (repeated) word is deaccented
    const givenText = 'The red bird sang, and the red bird flew.';
    const given = speak(givenText);
    const notGiven = speak(givenText, 'male', {'overrides': {'givenWords': 0}});
    const secondRed = (result) => Math.max(...segmentFrames(result, wordSegment(result, 'red', 'EH', 1)).map((frame) => frame.f0));
    assert(secondRed(given) < secondRed(notGiven) * 0.97);
    assert(Math.abs(wordPeak(given, 'red', 'EH') - wordPeak(notGiven, 'red', 'EH')) < 1);

    // A given word after the phrase's last new word is unaccented - the nucleus moves to the new word
    const postText = 'I saw the cat, and then I fed the cat.';
    const post = speak(postText);
    const notPost = speak(postText, 'male', {'overrides': {'givenWords': 0}});
    const secondCat = (result) => Math.max(...segmentFrames(result, wordSegment(result, 'cat', 'AE', 1)).map((frame) => frame.f0));
    assert(secondCat(post) < secondCat(notPost) * 0.95);

    // A phrase of given words only keeps its accents
    const allGiven = speak('The cat. The cat.');
    const allGivenCat = Math.max(...segmentFrames(allGiven, wordSegment(allGiven, 'cat', 'AE', 1)).map((frame) => frame.f0));
    assert(allGivenCat > f0At(allGiven, wordSegment(allGiven, 'cat', 'K', 1).start));

    // A reporting verb is unaccented
    const reported = speak('Come home, said Jonathan.');
    assert(wordPeak(reported, 'said', 'EH') < wordPeak(reported, 'jonathan', 'AA'));

    // A phrase of function words only - the first vowel is accented
    const functionOnly = speak('It is.');
    assert(Math.max(...functionOnly.frames.map((frame) => frame.f0)) > maleVoice.f0);

    // A phrase with no vowel
    const noVowel = speak('Shh.');
    const shh = noVowel.segments.filter((segment) => segment.type !== 'pause');
    assert.deepEqual(shh.map((segment) => segment.ph), ['SH']);
    assert(shh[0].phraseFinal);
    assert(noVowel.frames.every((frame) => Number.isFinite(frame.f0)));
});


test('speak, microprosody', () => {
    // F0 is raised after a voiceless obstruent and lowered after a voiced one
    const voiceless = speak('a pie', 'male', {'seed': 2});
    const voiced = speak('a buy', 'male', {'seed': 2});
    const pa = wordSegment(voiceless, 'pie', 'AY');
    const ba = wordSegment(voiced, 'buy', 'AY');
    const paF0 = f0At(voiceless, pa.start + 5);
    const baF0 = f0At(voiced, ba.start + 5);
    assert(paF0 > baF0);
});


test('speak, pitchScale and tractScale', () => {
    const text = 'We were away a year ago.';
    const base = speak(text);

    // The pitch scale scales F0
    const higher = speak(text, 'male', {'overrides': {'pitchScale': 1.2}});
    for (let ix = 0; ix < base.frames.length; ix += 10) {
        assert(Math.abs(higher.frames[ix].f0 / base.frames[ix].f0 - 1.2) < 1e-6);
    }

    // The tract scale scales every resonance
    const longer = speak(text, 'male', {'overrides': {'tractScale': 0.9}});
    for (let ix = 0; ix < base.frames.length; ix += 10) {
        for (const key of ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'fnz', 'fnp']) {
            assert(Math.abs(longer.frames[ix][key] / base.frames[ix][key] - 0.9) < 1e-9, key);
        }
        assert.equal(longer.frames[ix].f0, base.frames[ix].f0);
    }
    const fricative = speak('a sea', 'male', {'overrides': {'tractScale': 0.9}});
    const fricativeBase = speak('a sea');
    const fricativeFrame = middleFrame(fricative, wordSegment(fricative, 'sea', 'S'));
    const fricativeBaseFrame = middleFrame(fricativeBase, wordSegment(fricativeBase, 'sea', 'S'));
    assert(Math.abs(fricativeFrame.fp1 / fricativeBaseFrame.fp1 - 0.9) < 1e-9);
    assert.notDeepEqual(fricative.samples, fricativeBase.samples);
});


test('speak, streaming', () => {
    const text = 'Did you remember the coffee?';
    const blocks = [];
    const result = speak(text, 'male', {'onBlock': (block) => blocks.push(block)});
    assert(blocks.length > 1);

    // The streamed samples are the same synthesis - the normalized samples at a fixed gain
    const streamed = new Float32Array(blocks.reduce((sum, block) => sum + block.length, 0));
    let offset = 0;
    for (const block of blocks) {
        streamed.set(block, offset);
        offset += block.length;
    }
    assert.equal(streamed.length, result.samples.length);
    const streamGain = 0.89 / (4 * result.vowelLevel);
    const scale = result.rawPeak / 0.89 * streamGain;
    for (let ix = 0; ix < streamed.length; ix++) {
        const expected = result.samples[ix] * scale;
        assert(Math.abs(expected) < 0.8 - 1e-4, 'unexpected soft limiting');
        assert(Math.abs(streamed[ix] - expected) < 1e-5);
    }

    // The stream doesn't change the result
    assert.deepEqual(speak(text).samples, result.samples);
});


test('speak, streaming soft limit', () => {
    // Loud bursts are soft-limited below full scale
    const blocks = [];
    const result = speak('Pat cut a kit.', 'male', {'overrides': {'burstDb': 50}, 'onBlock': (block) => blocks.push(block)});
    const streamPeak = Math.max(...blocks.map(peak));
    assert(streamPeak > 0.8);
    assert(streamPeak <= 1);
    assert(result.rawPeak * 0.89 / (3.2 * result.vowelLevel) > 1);
});


test('speak, voice specs', () => {
    const text = 'We were away a year ago.';

    // A spec is the same as its name
    const female = speak(text, 'female');
    assert.deepEqual(speak(text, voices.female).samples, female.samples);
    assert.equal(female.voice.name, 'female');
    assert.equal(female.voice.rate, voices.female.rate);

    // A new voice - a few overrides of a base
    const higher = speak(text, {'name': 'higher', 'base': 'female', 'f0': 200});
    assert.equal(higher.voice.f0, 200);
    assert.equal(higher.voice.vowelBlend, 1);
    assert(meanF0(higher) > meanF0(female) * 1.1);

    // The rate option multiplies the voice's rate
    assert.equal(speak(text, 'male', {'rate': 2}).voice.rate, voices.male.rate * 2);

    // An unknown voice
    assert.throws(() => speak(text, 'nobody'), /Unknown voice "nobody"/);
});


test('speak, style', () => {
    const text = 'We were away a year ago.';
    const base = speak(text);

    // A style scales the voice's parameters
    const higher = speak(text, 'male', {'style': {'scale': {'f0': 1.2}}});
    for (let ix = 0; ix < base.frames.length; ix += 10) {
        assert(Math.abs(higher.frames[ix].f0 / base.frames[ix].f0 - 1.2) < 1e-6);
    }

    // At zero intensity, nothing changes
    assert.deepEqual(speak(text, 'male', {'style': {'intensity': 0, 'scale': {'f0': 1.2}, 'add': {'breathDb': 10}}}).samples, base.samples);

    // Styles in order
    const both = speak(text, 'male', {'style': [{'scale': {'rate': 2}}, {'add': {'commaPause': 100}}]});
    assert.equal(both.voice.rate, voices.male.rate * 2);
    assert.equal(both.voice.commaPause, baseVoice.commaPause + 100);
});


test('speak, voicing', () => {
    // No voicing - a whisper of the aspiration alone
    const result = speak('We were away a year ago.', 'male', {'overrides': {'voicing': 0, 'breathDb': -10}});
    assert(result.frames.every((frame) => frame.av === 0));
    assert(result.frames.some((frame) => frame.ah > 0));
    assert(result.rawPeak > 0);
});


test('speak, articulation precision', () => {
    // Vowel reduction - a word-final unstressed IY is reduced a little toward schwa (its F2 lowered)
    const text = 'the happy city of it';
    const happyF2 = (reduction) => {
        const result = speak(text, 'male', {'overrides': {reduction}});
        return middleFrame(result, wordSegment(result, 'happy', 'IY')).f2;
    };
    assert(happyF2(0) > happyF2(1) + 20);
    assert(happyF2(1) > happyF2(2) + 20);

    // Slower transitions - the same timing, different formant tracks
    const base = speak('a bad day');
    const slow = speak('a bad day', 'male', {'overrides': {'transitionScale': 2}});
    assert.deepEqual(slow.segments, base.segments);
    const badAE = wordSegment(base, 'bad', 'AE');
    const atOnset = (result) => frameValue(result, 'f2', badAE.start + 30);
    assert(atOnset(slow) < atOnset(base));
});


test('speak, intonation parameters', () => {
    // The question rise
    const question = speak('Were you away a year ago?');
    const higherQuestion = speak('Were you away a year ago?', 'male', {'overrides': {'questionRise': 1.8}});
    const questionEnd = (result) => {
        const last = vowelSegments(result).pop();
        return f0At(result, last.start + last.dur - 10);
    };
    assert(questionEnd(higherQuestion) > questionEnd(question) * 1.1);

    // Without the exclamation accent boost, an exclamation's pitch is a statement's
    const statement = speak('We were away a year ago.');
    const exclamation = speak('We were away a year ago!', 'male', {'overrides': {'exclamationAccent': 1}});
    assert.deepEqual(exclamation.frames.map((frame) => frame.f0), statement.frames.map((frame) => frame.f0));

    // A lower phrase onset
    const onsetStart = (result) => f0At(result, vowelSegments(result)[0].start + 5);
    const lowerOnset = speak('We were away a year ago.', 'male', {'overrides': {'phraseOnset': 1}});
    assert(onsetStart(lowerOnset) < onsetStart(statement) * 0.95);

    // A contour that comes to rest at each target is different
    const resting = speak('We were away a year ago.', 'male', {'overrides': {'contourFlow': 0}});
    assert.notDeepEqual(resting.frames.map((frame) => frame.f0), statement.frames.map((frame) => frame.f0));

    // The declination continues across a comma without a reset
    const commaText = 'When we were away, we ran far.';
    const commaReset = speak(commaText, 'male', {'overrides': {'commaReset': 1}});
    const commaCarry = speak(commaText, 'male', {'overrides': {'commaReset': 0}});
    const secondPhraseStart = (result) => f0At(result, wordSegment(result, 'we', 'W', 1).start);
    assert(secondPhraseStart(commaCarry) < secondPhraseStart(commaReset) * 0.98);

    // Without pitch drift, and without flutter - the F0 wander is aperiodic low-passed noise
    const statementF0 = statement.frames.map((frame) => frame.f0);
    const steady = speak('We were away a year ago.', 'male', {'overrides': {'pitchDrift': 0}});
    assert.notDeepEqual(steady.frames.map((frame) => frame.f0), statementF0);
    const unfluttered = speak('We were away a year ago.', 'male', {'overrides': {'flutter': 0}});
    assert.notDeepEqual(unfluttered.frames.map((frame) => frame.f0), statementF0);
    const still = speak('We were away a year ago.', 'male', {'overrides': {'pitchDrift': 0, 'flutter': 0}});
    const wander = statementF0.map((value, ix) => 12 * Math.log2(value / still.frames[ix].f0));
    const rms = Math.sqrt(wander.reduce((sum, value) => sum + value * value, 0) / wander.length);
    assert(rms > 0.03 && rms < 0.6);
});


test('speak, nasal parameters', () => {
    const text = 'No man ran.';
    const result = speak(text, 'male', {'overrides': {'nasalPole': 300, 'nasalZero': 500, 'nasalizationZero': 400}});
    const manM = wordSegment(result, 'man', 'M');
    assert(Math.abs(middleFrame(result, manM).fnz - 500) < 30);
    assert(Math.abs(middleFrame(result, manM).fnp - 300) < 30);
    const manAE = segmentFrames(result, wordSegment(result, 'man', 'AE'));
    assert(manAE[manAE.length - 1].fnz > 390);

    // A scaled nasal F1 locus - higher in the female voice
    const female = speak(text, 'female');
    const scaled = speak(text, 'female', {'overrides': {'scaleNasalF1': true}});
    const nasalF1 = (voiceResult) => middleFrame(voiceResult, wordSegment(voiceResult, 'man', 'M')).f1;
    assert(nasalF1(scaled) > nasalF1(female) * 1.05);
});


test('speak, source parameters', () => {
    // A stronger HH
    const home = speak('home');
    const strongHome = speak('home', 'male', {'overrides': {'hhDb': -14}});
    const hhAh = (result) => middleFrame(result, wordSegment(result, 'home', 'HH')).ah;
    assert(hhAh(strongHome) > hhAh(home) * 2);

    // The aspiration's glottal modulation
    const unmodulated = speak('home', 'male', {'overrides': {'noiseModulation': 0}});
    assert.notDeepEqual(unmodulated.samples, home.samples);

    // Louder accents - the sources rise with the pitch above the baseline (and are untouched at or below it)
    const flat = speak('We were away a year ago.', 'male', {'overrides': {'f0Loudness': 0}});
    const loud = speak('We were away a year ago.', 'male', {'overrides': {'f0Loudness': 1.2}});
    const gains = loud.frames.map((frame, ix) => (flat.frames[ix].av > 0.01 ? frame.av / flat.frames[ix].av : 1));
    assert(Math.max(...gains) > 1.3);
    assert(Math.min(...gains) > 0.999);
    assert(gains.some((gain) => Math.abs(gain - 1) < 1e-9));
    assert.deepEqual(loud.segments, flat.segments);

    // Micro-dynamics - the rd drift and the formant fluctuations are small, aperiodic, and removable
    const text = 'We were away a year ago.';
    const base = speak(text);
    const still = speak(text, 'male', {'overrides': {'rdDrift': 0, 'formantWobble': 0}});
    const ratios = (key) => base.frames.map((frame, ix) => frame[key] / still.frames[ix][key] - 1);
    for (const [key, limit] of [['rd', 0.3], ['f2', 0.03]]) {
        const values = ratios(key);
        assert(values.some((value) => Math.abs(value) > 1e-4));
        assert(values.every((value) => Math.abs(value) < limit));
    }
});


test('speak, vowel blend', () => {
    // Between the men's and the women's vowel spaces
    const text = 'Bob saw Bob.';
    const vowelF1 = (vowelBlend) => {
        const result = speak(text, 'male', {'overrides': {vowelBlend}});
        return middleFrame(result, wordSegment(result, 'saw', 'AO')).f1;
    };
    const [men, half, women] = [vowelF1(0), vowelF1(0.5), vowelF1(1)];
    assert(men < half && half < women);
    assert(Math.abs(half - (men + women) / 2) < 0.1 * (women - men));
});
