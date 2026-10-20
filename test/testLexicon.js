// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {numberWords, setPronouncer, textToPhrases, whWords} from '../lib/lexicon.js';
import {strict as assert} from 'node:assert';
import {g2pModel} from '../lib/models/g2pModel.js';
import {loadG2p} from '../lib/g2p.js';
import test from 'node:test';


// A fresh instance of the lexicon module, with no pronouncer set
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


// A small pronouncer - a few words' phones
const miniPhones = {
    'walk': ['W', 'AO1', 'K'], 'box': ['B', 'AA1', 'K', 'S'], 'quick': ['K', 'W', 'IH1', 'K'], 'the': ['DH', 'AH0'],
    'have': ['HH', 'AE1', 'V']
};
function miniPronouncer(word) {
    return miniPhones[word];
}


// The module's pronouncer - the G2P network, as the package sets it
setPronouncer(await loadG2p(g2pModel));


test('lexicon.textToPhrases, no pronouncer', async () => {
    const lexicon = await freshLexicon('no-pronouncer');
    assert.throws(() => lexicon.textToPhrases('walk'), {'message': 'No pronouncer - call setPronouncer(pronounce) first'});

    // Unset again
    lexicon.setPronouncer(miniPronouncer);
    assert.deepEqual(wordPhones(lexicon.textToPhrases('walk')[0]), ['W AO1 K']);
    lexicon.setPronouncer(null);
    assert.throws(() => lexicon.textToPhrases('walk'), {'message': 'No pronouncer - call setPronouncer(pronounce) first'});
});


test('lexicon.setPronouncer', async () => {
    const lexicon = await freshLexicon('set-pronouncer');

    // The pronouncer pronounces every word - function words lose their stress
    const words = [];
    lexicon.setPronouncer((word) => {
        words.push(word);
        return word === 'the' ? ['DH', 'IY1'] : ['B', 'AA1', 'K', 'S'];
    });
    assert.deepEqual(wordPhones(lexicon.textToPhrases('the walk')[0]), ['DH IY0', 'B AA1 K S']);
    assert.deepEqual(words, ['the', 'walk']);

    // Setting it again replaces it
    lexicon.setPronouncer(miniPronouncer);
    assert.deepEqual(wordPhones(lexicon.textToPhrases('the walk')[0]), ['DH AH0', 'W AO1 K']);
});


test('lexicon.setPronouncer, readings from context', async () => {
    const lexicon = await freshLexicon('set-pronouncer-readings');

    // An ambiguous word takes the reading the pronouncer chooses from the text's items (its words and phrase-end marks)
    const choices = [];
    const pronounce = (word, reading = 0) => {
        if (word === 'read') {
            return reading === 1 ? ['R', 'IY1', 'D'] : ['R', 'EH1', 'D'];
        }
        return ['W', 'ER1', 'D'];
    };
    pronounce.ambiguous = (word) => word === 'read';
    pronounce.chooseReading = (items, index) => {
        choices.push([items, index]);
        return items[index - 1] === 'will' ? 1 : 0;
    };
    lexicon.setPronouncer(pronounce);
    const phrases = lexicon.textToPhrases('I will read; I read it. Read!');
    assert.deepEqual(phrases.map(wordPhones), [['W ER0 D', 'W ER0 D', 'R IY1 D'], ['W ER0 D', 'R EH1 D', 'W ER0 D'], ['R EH1 D']]);
    assert.deepEqual(phrases.map((phrase) => [phrase.end, phrase.major]), [['comma', true], ['period', false], ['exclamation', false]]);
    const items = ['i', 'will', 'read', ',', 'i', 'read', 'it', '.', 'read', '!'];
    assert.deepEqual(choices, [[items, 2], [items, 5], [items, 8]]);
});


test('lexicon.textToPhrases, the G2P network', () => {
    const [phrase] = textToPhrases('Hello world');
    assert.deepEqual(wordTexts(phrase), ['hello', 'world']);
    assert.deepEqual(wordPhones(phrase), ['HH AH0 L OW1', 'W ER1 L D']);
    assert.deepEqual(phrase.words.map((word) => word.syllables), [2, 1]);
    assert.deepEqual(phrase.words.map((word) => word.isFunction), [false, false]);
    assert.equal(phrase.end, 'period');

    // Words no dictionary has, and a heteronym read from its context
    assert.deepEqual(wordPhones(textToPhrases('Queequeg')[0]), ['K W IY1 K W EH2 G']);
    assert.deepEqual(wordPhones(textToPhrases('I will read it.')[0])[2], 'R IY1 D');
    assert.deepEqual(wordPhones(textToPhrases('I read it yesterday.')[0])[1], 'R EH1 D');
});


test('lexicon.textToPhrases, function words', async () => {
    const lexicon = await freshLexicon('function-words');
    lexicon.setPronouncer(miniPronouncer);
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


test('lexicon.textToPhrases, month and day abbreviations', () => {
    // With their period, the months and days are spelled out (without ending the phrase); without it, they're words
    const phrases = textToPhrases('On Mon. Nov. 5 and Fri. Feb. 2, mon ami, Aug. sat on the sun.');
    assert.deepEqual(phrases.map(wordTexts), [
        ['on', 'monday', 'november', 'five', 'and', 'friday', 'february', 'two'],
        ['mon', 'ami'],
        ['august', 'sat', 'on', 'the', 'sun']
    ]);
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


test('lexicon.textToPhrases, object property names', () => {
    // Words that name an object's inherited properties are words, not abbreviations
    const [phrase] = textToPhrases('The constructor. Constructor tostring valueof');
    assert.deepEqual(wordTexts(phrase), ['the', 'constructor']);
    assert.deepEqual(wordTexts(textToPhrases('Constructor tostring valueof')[0]), ['constructor', 'tostring', 'valueof']);
});


test('lexicon.textToPhrases, accents', () => {
    // Accented letters lose their accents, and the ligatures read as English says them (Cæsar, Œdipus)
    const [phrase] = textToPhrases('Señor Amadís, ');
    assert.deepEqual(wordTexts(phrase), ['senor', 'amadis']);
    assert.deepEqual(wordPhones(phrase)[0], 'S IY2 N Y AO1 R');
    assert.deepEqual(wordTexts(textToPhrases('Œdipus Cæsar Straße in Łódź, Søren, ŋa')[0]),
        ['edipus', 'cesar', 'strase', 'in', 'lodz']);
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
