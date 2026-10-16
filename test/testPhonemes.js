// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {strict as assert} from 'node:assert';
import {isVowel} from '../lib/phonemes.js';
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
    assert(!isVowel('IY1'));
});
