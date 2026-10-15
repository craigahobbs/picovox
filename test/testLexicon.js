// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

import {numberWords, textToPhrases, whWords} from '../lib/lexicon.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// A fresh instance of the lexicon module, with no dictionary loaded
function freshLexicon(name) {
    return import(`../lib/lexicon.js?${name}`);
}


// The phones of a phrase's words as ARPAbet strings (stress digits appended)
function wordPhones(phrase) {
    return phrase.words.map((word) => word.phones.map(({ph, stress}) => (stress === null ? ph : `${ph}${stress}`)).join(' '));
}


// The texts of a phrase's words
function wordTexts(phrase) {
    return phrase.words.map((word) => word.text);
}


// A small pronouncing dictionary - a comment, a blank line, and an alternate pronunciation
const miniDictionary = `\
# a full-line comment

walk W AO1 K
walk(2) W AA1 K
box B AA1 K S # a trailing comment
quick K W IH1 K
the DH AH0
have HH AE1 V
`;


test('lexicon.setDictionary, no dictionary', async () => {
    const lexicon = await freshLexicon('no-dictionary');
    // As in a browser - no Node built-in modules
    const {process} = globalThis;
    const {getBuiltinModule} = process;
    process.getBuiltinModule = undefined;
    try {
        assert.throws(() => lexicon.textToPhrases('walk'), {'message': 'No pronouncing dictionary - call setDictionary(text) first'});
    } finally {
        process.getBuiltinModule = getBuiltinModule;
    }
});


test('lexicon.setDictionary', async () => {
    const lexicon = await freshLexicon('set-dictionary');
    lexicon.setDictionary(miniDictionary);

    // First pronunciations only, trailing comments stripped
    const [phrase] = lexicon.textToPhrases('walk box quick');
    assert.deepEqual(wordPhones(phrase), ['W AO1 K', 'B AA1 K S', 'K W IH1 K']);

    // Setting the dictionary again replaces it
    lexicon.setDictionary('walk W AA1 K\n');
    assert.deepEqual(wordPhones(lexicon.textToPhrases('walk')[0]), ['W AA1 K']);
});


test('lexicon.textToPhrases, Node dictionary', () => {
    const [phrase] = textToPhrases('Hello world');
    assert.deepEqual(wordTexts(phrase), ['hello', 'world']);
    assert.deepEqual(wordPhones(phrase), ['HH AH0 L OW1', 'W ER1 L D']);
    assert.deepEqual(phrase.words.map((word) => word.syllables), [2, 1]);
    assert.deepEqual(phrase.words.map((word) => word.isFunction), [false, false]);
    assert.equal(phrase.end, 'period');
});


test('lexicon.textToPhrases, unknown words with suffixes', async () => {
    const lexicon = await freshLexicon('suffixes');
    lexicon.setDictionary(miniDictionary);
    const [phrase] = lexicon.textToPhrases('walks boxes walked walking walker quickly');
    assert.deepEqual(wordPhones(phrase), [
        'W AO1 K Z',
        'B AA1 K S IH0 Z',
        'W AO1 K D',
        'W AO1 K IH0 NG',
        'W AO1 K ER0',
        'K W IH1 K L IY0'
    ]);
    assert.deepEqual(phrase.words.map((word) => word.syllables), [1, 2, 1, 2, 2, 2]);
});


test('lexicon.textToPhrases, unknown words spelled by letter', async () => {
    const lexicon = await freshLexicon('letters');
    lexicon.setDictionary(miniDictionary);
    const [phrase] = lexicon.textToPhrases('blorr zug fex jive yoh');
    assert.deepEqual(wordPhones(phrase), [
        // Doubled letters collapse
        'B L AA1 R',
        'Z AH1 G',
        // X is two phones
        'F EH1 K S',
        'JH IH1 V EH1',
        'Y AA1 HH'
    ]);

    // Doubled phones from different letters collapse, apostrophes inside a word are skipped
    assert.deepEqual(wordPhones(lexicon.textToPhrases('kcuq o\'qt')[0]), ['K AH1 K', 'AA1 K T']);

    // A suffix whose stem is unknown falls back to letters
    assert.deepEqual(wordPhones(lexicon.textToPhrases('cats')[0]), ['K AE1 T S']);
});


test('lexicon.textToPhrases, function words', async () => {
    const lexicon = await freshLexicon('function-words');
    lexicon.setDictionary(miniDictionary);
    const [phrase] = lexicon.textToPhrases('have the box');
    assert.deepEqual(phrase.words.map((word) => word.isFunction), [true, true, false]);

    // Function words are unaccented - their vowels' stress is reduced
    assert.deepEqual(wordPhones(phrase), ['HH AE0 V', 'DH AH0', 'B AA1 K S']);
});


test('lexicon.textToPhrases, phrase ends', () => {
    const phrases = textToPhrases('One, two; three: four. Is it? What is it? Who? Stop! Go');
    assert.deepEqual(phrases.map(wordTexts), [
        ['one'], ['two'], ['three'], ['four'], ['is', 'it'], ['what', 'is', 'it'], ['who'], ['stop'], ['go']
    ]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), [
        'comma', 'comma', 'comma', 'period', 'question', 'wh', 'wh', 'exclamation', 'period'
    ]);
    assert.deepEqual(phrases.map((phrase) => phrase.major), [false, true, true, false, false, false, false, false, false]);
});


test('lexicon.textToPhrases, empty phrases', () => {
    assert.deepEqual(textToPhrases(''), []);
    assert.deepEqual(textToPhrases('  ,.;!? -- '), []);
    const phrases = textToPhrases('...Well,, then!!');
    assert.deepEqual(phrases.map(wordTexts), [['well'], ['then']]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), ['comma', 'exclamation']);
});


test('lexicon.textToPhrases, period ends a word', () => {
    // A word with a period ends the phrase - the following period has no words
    const phrases = textToPhrases('The end. The rest');
    assert.deepEqual(phrases.map(wordTexts), [['the', 'end'], ['the', 'rest']]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), ['period', 'period']);
});


test('lexicon.textToPhrases, abbreviations', () => {
    // An abbreviation's period does not end the phrase
    const phrases = textToPhrases('Mr. and Mrs. Smith met Dr. Jones on St. Mark, vs. etc.');
    assert.deepEqual(phrases.map(wordTexts), [
        ['mister', 'and', 'missus', 'smith', 'met', 'doctor', 'jones', 'on', 'saint', 'mark'],
        ['versus', 'etcetera']
    ]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), ['comma', 'period']);

    // Without a period too
    assert.deepEqual(wordTexts(textToPhrases('Dr Smith')[0]), ['doctor', 'smith']);
});


test('lexicon.textToPhrases, apostrophes', () => {
    // Curly apostrophes are straightened, leading and trailing apostrophes stripped
    const [phrase] = textToPhrases('Don’t the dogs\' bones');
    assert.deepEqual(wordTexts(phrase), ['don\'t', 'the', 'dogs', 'bones']);
    assert.deepEqual(wordPhones(phrase)[0], 'D OW1 N T');
});


test('lexicon.textToPhrases, numbers', () => {
    assert.deepEqual(wordTexts(textToPhrases('7 ships')[0]), ['seven', 'ships']);
    assert.deepEqual(wordTexts(textToPhrases('612')[0]), ['six', 'hundred', 'twelve']);
    assert.deepEqual(wordTexts(textToPhrases('3.14')[0]), ['three', 'point', 'one', 'four']);
    assert.deepEqual(wordTexts(textToPhrases('0.5')[0]), ['zero', 'point', 'five']);

    // A trailing period ends the phrase - it is not a decimal point
    const phrases = textToPhrases('It is 42. Yes');
    assert.deepEqual(phrases.map(wordTexts), [['it', 'is', 'forty', 'two'], ['yes']]);
});


test('lexicon.numberWords', () => {
    assert.deepEqual(numberWords(0), ['zero']);
    assert.deepEqual(numberWords(13), ['thirteen']);
    assert.deepEqual(numberWords(19), ['nineteen']);
    assert.deepEqual(numberWords(20), ['twenty']);
    assert.deepEqual(numberWords(40), ['forty']);
    assert.deepEqual(numberWords(99), ['ninety', 'nine']);
    assert.deepEqual(numberWords(100), ['one', 'hundred']);
    assert.deepEqual(numberWords(101), ['one', 'hundred', 'one']);
    assert.deepEqual(numberWords(1000), ['one', 'thousand']);
    assert.deepEqual(numberWords(2024), ['two', 'thousand', 'twenty', 'four']);
    assert.deepEqual(numberWords(1234567), [
        'one', 'million', 'two', 'hundred', 'thirty', 'four', 'thousand', 'five', 'hundred', 'sixty', 'seven'
    ]);
    assert.deepEqual(numberWords(3e9), ['three', 'billion']);
    assert.deepEqual(numberWords(12e9 + 5), ['twelve', 'billion', 'five']);

    // Not a number
    assert.deepEqual(numberWords(NaN), []);
});


test('lexicon.whWords', () => {
    assert(whWords.has('what'));
    assert(whWords.has('how'));
    assert(!whWords.has('is'));
});
