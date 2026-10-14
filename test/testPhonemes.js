// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {consonants, isVowel, noiseSpectra, schwa, vowels} from '../lib/phonemes.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// The ARPAbet phonemes of the CMU Pronouncing Dictionary
const arpabetVowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'];
const arpabetConsonants = [
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
];


test('phonemes.isVowel', () => {
    for (const ph of arpabetVowels) {
        assert(isVowel(ph), ph);
    }
    for (const ph of arpabetConsonants) {
        assert(!isVowel(ph), ph);
    }
    assert(!isVowel('_'));
});


test('phonemes, the ARPAbet', () => {
    assert.deepEqual(Object.keys(vowels).sort(), arpabetVowels);
    assert.deepEqual(Object.keys(consonants).sort(), arpabetConsonants);
});


test('phonemes.vowels', () => {
    for (const [ph, vowel] of Object.entries(vowels)) {
        // Inherent duration exceeds the minimum
        const [inherent, minimum] = vowel.dur;
        assert(inherent > minimum && minimum > 0, ph);

        // One target (monophthong) or two (diphthong or lax glide), the same for each voice
        assert(vowel.male.length === 1 || vowel.male.length === 2, ph);
        assert.equal(vowel.female.length, vowel.male.length, ph);
        for (const [ixTarget, maleTarget] of vowel.male.entries()) {
            const femaleTarget = vowel.female[ixTarget];
            for (const target of [maleTarget, femaleTarget]) {
                // Formants ascend - F1 < F2 < F3
                assert.equal(target.length, 3, ph);
                assert(target[0] < target[1] && target[1] < target[2], ph);
            }

            // The female vocal tract is shorter - every formant is higher
            for (let ixFormant = 0; ixFormant < 3; ixFormant++) {
                assert(femaleTarget[ixFormant] > maleTarget[ixFormant], ph);
            }
        }
    }

    // Diphthongs move
    for (const ph of ['AY', 'AW', 'OY', 'EY', 'OW']) {
        assert.equal(vowels[ph].male.length, 2, ph);
    }

    // Tense IY is longer than lax IH
    assert(vowels.IY.dur[0] > vowels.IH.dur[0]);

    // IY is the front high vowel - the lowest F1 and the highest F2
    const [[iyF1, iyF2]] = vowels.IY.male;
    for (const [ph, vowel] of Object.entries(vowels)) {
        if (ph !== 'IY') {
            assert(vowel.male[0][0] >= iyF1, ph);
            assert(vowel.male[0][1] < iyF2, ph);
        }
    }
});


test('phonemes.schwa', () => {
    for (const voice of ['male', 'female']) {
        const [f1, f2, f3] = schwa[voice];
        assert(f1 < f2 && f2 < f3);
    }
    assert(schwa.female[0] > schwa.male[0]);
});


test('phonemes.consonants', () => {
    const types = new Set(['stop', 'affricate', 'fricative', 'aspirate', 'nasal', 'liquid', 'glide']);
    for (const [ph, consonant] of Object.entries(consonants)) {
        assert(types.has(consonant.type), ph);
        assert.equal(typeof consonant.voiced, 'boolean', ph);
        assert.equal(typeof consonant.place, 'string', ph);
        const [inherent, minimum] = consonant.dur;
        assert(inherent > minimum && minimum > 0, ph);
        assert.equal(consonant.f.length, 3, ph);
        assert(consonant.f[0] < consonant.f[1] && consonant.f[1] < consonant.f[2], ph);
        assert(consonant.k >= 0 && consonant.k <= 1, ph);
        assert(consonant.k3 === undefined || (consonant.k3 >= 0 && consonant.k3 <= 1), ph);
    }

    // Voiced and voiceless pairs share a place and a type
    const pairs = [['P', 'B'], ['T', 'D'], ['K', 'G'], ['CH', 'JH'], ['F', 'V'], ['TH', 'DH'], ['S', 'Z'], ['SH', 'ZH']];
    for (const [voiceless, voiced] of pairs) {
        assert(!consonants[voiceless].voiced, voiceless);
        assert(consonants[voiced].voiced, voiced);
        assert.equal(consonants[voiceless].place, consonants[voiced].place);
        assert.equal(consonants[voiceless].type, consonants[voiced].type);
    }

    // Nasals, liquids, and glides are voiced
    for (const consonant of Object.values(consonants)) {
        if (['nasal', 'liquid', 'glide'].includes(consonant.type)) {
            assert(consonant.voiced);
        }
    }
});


test('phonemes.noiseSpectra', () => {
    // Every fricative and affricate has its own spectrum, every stop place has a burst spectrum
    for (const [ph, consonant] of Object.entries(consonants)) {
        if (consonant.type === 'fricative' || consonant.type === 'affricate') {
            assert(ph in noiseSpectra, ph);
        } else if (consonant.type === 'stop') {
            assert(consonant.place in noiseSpectra, ph);
        }
    }
    for (const [name, spectrum] of Object.entries(noiseSpectra)) {
        assert.deepEqual(Object.keys(spectrum), ['fp1', 'bp1', 'ap1', 'fp2', 'bp2', 'ap2', 'ab', 'fhp', 'db'], name);
        assert(spectrum.db < 0, name);
        assert(spectrum.fhp < spectrum.fp1, name);
    }

    // Voiced fricatives are weaker than their voiceless pairs, sibilants are the loudest fricatives
    for (const [voiceless, voiced] of [['S', 'Z'], ['SH', 'ZH'], ['CH', 'JH'], ['F', 'V'], ['TH', 'DH']]) {
        assert(noiseSpectra[voiced].db < noiseSpectra[voiceless].db, voiced);
    }
    assert(noiseSpectra.S.db > noiseSpectra.F.db);
    assert(noiseSpectra.SH.db > noiseSpectra.TH.db);

    // S is higher than SH
    assert(noiseSpectra.S.fp1 > noiseSpectra.SH.fp1);
});
