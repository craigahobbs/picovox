// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {
    moneyWords, numberTokenWords, numberWords, setPronouncer, textToPhrases, timeWords, whWords, yearWords
} from '../lib/lexicon.js';
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
    assert.deepEqual(phrases.map((phrase) => phrase.end), ['comma', 'period', 'exclamation']);
    const items = ['i', 'will', 'read', ',', 'i', 'read', 'it', '.', 'read', '!'];
    assert.deepEqual(choices, [[items, 2], [items, 5], [items, 8]]);
});


test('lexicon.setPronouncer, words in capitals', async () => {
    const lexicon = await freshLexicon('set-pronouncer-capitals');

    // A word in capitals the pronouncer finds spelled is its letters' names ("a" as "ay", "i" as "eye") - asked only of
    // words in capitals (two letters or more), and a spelled word isn't a function word or an ambiguous one. The
    // pronouncer: "ai" is spelled, "us" alone among lowercase words
    const asked = new Set();
    const letters = {'ay': ['EY1'], 'eye': ['AY1'], 't': ['T', 'IY1'], 'u': ['Y', 'UW1'], 's': ['EH1', 'S']};
    const pronounce = (word, reading = 0) => letters[word] ?? (reading === 1 ? ['R', 'IY1', 'D'] : ['W', 'ER1', 'D']);
    pronounce.ambiguous = (word) => word === 'us' || word === 'read';
    pronounce.chooseReading = () => 1;
    pronounce.spelled = (word, beside) => {
        asked.add(`${word} ${beside}`);
        return word === 'ai' || (word === 'us' && !beside);
    };
    lexicon.setPronouncer(pronounce);
    const phrases = lexicon.textToPhrases('The AI, US. It is NOT ait; I read. A US');
    assert.deepEqual(phrases.map(wordTexts), [['the', 'ai'], ['us'], ['it', 'is', 'not', 'ait'], ['i', 'read'], ['a', 'us']]);
    assert.deepEqual(phrases.map(wordPhones), [
        ['W ER0 D', 'EY1 AY1'], ['Y UW1 EH1 S'], ['W ER0 D', 'W ER0 D', 'W ER1 D', 'W ER1 D'], ['W ER0 D', 'R IY1 D'],
        ['W ER0 D', 'Y UW1 EH1 S']
    ]);
    assert.deepEqual([...asked].toSorted(), ['ai false', 'not false', 'us false']);

    // Beside words in capitals that are read - its neighbors in its phrase, both if it has two - a word is asked as one
    // beside them ("us" is read); beside a lowercase word or a spelled one, as one alone. A capital letter alone is read
    assert.deepEqual(lexicon.textToPhrases('NOT US, the US AI. US, NOT them; NOT US AI, tell US A lie, NOT US A').map(wordPhones), [
        ['W ER1 D', 'R IY0 D'], ['W ER0 D', 'Y UW1 EH1 S', 'EY1 AY1'], ['Y UW1 EH1 S'], ['W ER1 D', 'W ER0 D'],
        ['W ER1 D', 'Y UW1 EH1 S', 'EY1 AY1'], ['W ER1 D', 'Y UW1 EH1 S', 'W ER0 D', 'W ER1 D'], ['W ER1 D', 'R IY0 D', 'W ER0 D']
    ]);
    assert.deepEqual([...asked].toSorted(), ['ai false', 'ai true', 'not false', 'not true', 'us false', 'us true']);

    // A spelled word's -s is its last letter's ("AIs", "US's" - the pronouncer is asked of the capitals)
    Object.assign(letters, {'eye\'s': ['AY1', 'Z'], 's\'s': ['EH1', 'S', 'IH0', 'Z']});
    asked.clear();
    assert.deepEqual(lexicon.textToPhrases('AIs, the US\'s nots').map(wordPhones),
        [['EY1 AY1 Z'], ['W ER0 D', 'Y UW1 EH1 S IH0 Z', 'W ER1 D']]);
    assert.deepEqual([...asked].toSorted(), ['ai false', 'us false']);

    // A pronouncer without a spelled head reads every word
    lexicon.setPronouncer(miniPronouncer);
    assert.deepEqual(wordPhones(lexicon.textToPhrases('THE BOX')[0]), ['DH AH0', 'B AA1 K S']);
});


test('lexicon.textToPhrases, words in capitals', () => {
    // The G2P network's spelled head - an acronym is spelled, a word in capitals read
    assert.deepEqual(wordPhones(textToPhrases('The AI and the TOC.')[0]), ['DH AH0', 'EY1 AY1', 'AH0 N D', 'DH AH0', 'T IY1 OW1 S IY1']);
    assert.deepEqual(wordPhones(textToPhrases('I did NOT say STOP!')[0]), ['AY0', 'D IH0 D', 'N AA1 T', 'S EY1', 'S T AA1 P']);
    assert.deepEqual(wordPhones(textToPhrases('the ai')[0]), ['DH AH0', 'AY1']);

    // A common word that, alone in capitals, is an acronym - read beside a word in capitals
    assert.deepEqual(wordPhones(textToPhrases('The US and AD.')[0]), ['DH AH0', 'Y UW1 EH1 S', 'AH0 N D', 'EY1 D IY1']);
    assert.deepEqual(wordPhones(textToPhrases('LEAVE US ALONE!')[0]), ['L IY1 V', 'AH0 S', 'AH0 L OW1 N']);
    assert.deepEqual(wordPhones(textToPhrases('The CEOs met IBM\'s board.')[0]),
        ['DH AH0', 'S IY1 IY1 OW1 Z', 'M EH1 T', 'AY1 B IY1 EH1 M Z', 'B AO1 R D']);
});


test('lexicon.textToPhrases, initials', () => {
    // An initial - a capital letter with its period - is its letter's name, and doesn't end the phrase
    const phrases = textToPhrases('Dr. A. B. Jones met I. M. Pei in the U.S. capital. Plan a. Then b');
    assert.deepEqual(phrases.map(wordTexts), [
        ['doctor', 'ay', 'b', 'jones', 'met', 'eye', 'm', 'pei', 'in', 'the', 'u', 's', 'capital'], ['plan', 'a'], ['then', 'b']
    ]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), ['period', 'period', 'period']);
    assert.deepEqual(wordPhones(phrases[0]).slice(1, 3), ['EY1', 'B IY1']);

    // A sentence's last initial ends it
    assert.deepEqual(textToPhrases('It was Dr. X.').map(wordTexts), [['it', 'was', 'doctor', 'x']]);

    // "I." after a lowercase word, and not before an initial, is the pronoun
    assert.deepEqual(textToPhrases('So was I. M. Pei, said I. Asimov').map(wordTexts),
        [['so', 'was', 'eye', 'm', 'pei'], ['said', 'i'], ['asimov']]);
    assert.deepEqual(textToPhrases('So was I. then, said I, he').map(wordTexts), [['so', 'was', 'i'], ['then'], ['said', 'i'], ['he']]);
    assert.deepEqual(textToPhrases('I. Asimov, said I.').map(wordTexts), [['eye', 'asimov'], ['said', 'i']]);
});


test('lexicon.textToPhrases, units', () => {
    // A unit after a number is its words - the plural, but after 1; "sq" and "cu" before one are square and cubic
    assert.deepEqual(textToPhrases('It is 40,075 km around, 5 mi or 1 km wide, 660 ft. It weighs 1.5 kg').map(wordTexts), [
        ['it', 'is', 'forty', 'thousand', 'seventy', 'five', 'kilometers', 'around'], ['five', 'miles', 'or', 'one', 'kilometer', 'wide'],
        ['six', 'hundred', 'sixty', 'feet'], ['it', 'weighs', 'one', 'point', 'five', 'kilograms']
    ]);
    assert.deepEqual(textToPhrases('93,000 sq mi, 530 cu mi, 27 GW, 55 mph, 5 mL and 5 sq of it').map(wordTexts), [
        ['ninety', 'three', 'thousand', 'square', 'miles'], ['five', 'hundred', 'thirty', 'cubic', 'miles'],
        ['twenty', 'seven', 'gigawatts'],
        ['fifty', 'five', 'miles', 'per', 'hour'], ['five', 'milliliters', 'and', 'five', 'sq', 'of', 'it']
    ]);

    // Not after a number, or not as the unit is written, it's a word
    assert.deepEqual(textToPhrases('A km away, 4G and 5 Ma ago').map(wordTexts),
        [['a', 'km', 'away'], ['four', 'g', 'and', 'five', 'ma', 'ago']]);
    assert.deepEqual(textToPhrases('km away at 10 ft/s over 5 sq').map(wordTexts),
        [['km', 'away', 'at', 'ten', 'feet', 'per', 'second', 'over', 'five', 'sq']]);

    // Units with a slash or a power, and degrees
    const text = 'At 90 km/h, 1 m/s, 240,000 km2, 3 m³ and 1 ft2 at 0 °C (32 °F), a 350° view, 1° off';
    assert.deepEqual(textToPhrases(text).map(wordTexts), [
        ['at', 'ninety', 'kilometers', 'per', 'hour'], ['one', 'meter', 'per', 'second'],
        ['two', 'hundred', 'forty', 'thousand', 'square', 'kilometers'], ['three', 'cubic', 'meters', 'and', 'one', 'square', 'foot', 'at',
            'zero', 'degrees', 'celsius'], ['thirty', 'two', 'degrees', 'fahrenheit'],
        ['a', 'three', 'hundred', 'fifty', 'degrees', 'view'], ['one', 'degree', 'off']
    ]);

    // In capitals they aren't units - the letters and the number
    assert.deepEqual(textToPhrases('On the M2, the M² and KM/H').map(wordTexts),
        [['on', 'the', 'm', 'two'], ['the', 'm', 'and', 'km', 'h']]);
});


test('lexicon.textToPhrases, symbols', () => {
    // The symbols read as words - a minus sign before a number (a hyphen only after a space or an opening parenthesis)
    assert.deepEqual(textToPhrases('Tom & Jerry: 2 + 2 = 4, 3 × 5, ±2 ≈ 2, ~500, <10 and >5, a < b').map(wordTexts), [
        ['tom', 'and', 'jerry'], ['two', 'plus', 'two', 'equals', 'four'], ['three', 'times', 'five'],
        ['plus', 'or', 'minus', 'two', 'approximately', 'two'], ['about', 'five', 'hundred'],
        ['less', 'than', 'ten', 'and', 'greater', 'than', 'five'], ['a', 'b']
    ]);
    assert.deepEqual(textToPhrases('−5 °C, -3 (-2) and 1939-45 − x').map(wordTexts), [
        ['minus', 'five', 'degrees', 'celsius'], ['minus', 'three'], ['minus', 'two'],
        ['and', 'nineteen', 'thirty', 'nine', 'forty', 'five', 'x']
    ]);
});


test('lexicon.textToPhrases, dotted abbreviations', () => {
    // A dotted abbreviation is its letters' names - "i.e." and "e.g." their words - and doesn't end the phrase
    const phrases = textToPhrases('He has a Ph.D., i.e. a doctorate, e.g. from the U.S.A. or the U.K, a.k.a. Britain. The end');
    assert.deepEqual(phrases.map(wordTexts), [
        ['he', 'has', 'a', 'p', 'h', 'd'], ['that', 'is', 'a', 'doctorate'],
        ['for', 'example', 'from', 'the', 'u', 's', 'ay', 'or', 'the', 'u', 'k'],
        ['ay', 'k', 'ay', 'britain'], ['the', 'end']
    ]);

    // a.m. and p.m. are their own
    assert.deepEqual(textToPhrases('At 5 p.m. or a.m.').map(wordTexts), [['at', 'five', 'pee', 'em', 'or', 'ay', 'em']]);
});


test('lexicon.textToPhrases, the G2P network', () => {
    const [phrase] = textToPhrases('Hello world');
    assert.deepEqual(wordTexts(phrase), ['hello', 'world']);
    assert.deepEqual(wordPhones(phrase), ['HH AH0 L OW1', 'W ER1 L D']);
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
});


test('lexicon.textToPhrases, parentheses', () => {
    // An aside's parentheses are comma breaks - before its opening and after its closing
    let phrases = textToPhrases('He left the room (quietly, as always) and closed the door.');
    assert.deepEqual(phrases.map(wordTexts), [
        ['he', 'left', 'the', 'room'], ['quietly'], ['as', 'always'], ['and', 'closed', 'the', 'door']
    ]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), ['comma', 'comma', 'comma', 'period']);

    // A punctuation mark right after a parenthesis ends the phrase (the parenthesis adds no break)
    phrases = textToPhrases('She was born (1818), lived (long); and died (old). Was it (really)? (Aside.) The end (truly)');
    assert.deepEqual(phrases.map(wordTexts), [
        ['she', 'was', 'born'], ['eighteen', 'eighteen'], ['lived'], ['long'], ['and', 'died'], ['old'], ['was', 'it'], ['really'],
        ['aside'], ['the', 'end'], ['truly']
    ]);
    assert.deepEqual(phrases.map((phrase) => phrase.end), [
        'comma', 'comma', 'comma', 'comma', 'comma', 'period', 'comma', 'question', 'period', 'comma', 'period'
    ]);
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


test('lexicon.textToPhrases, title abbreviations', () => {
    // With their period, the titles and the rest are spelled out (without ending the phrase); "No." before a number is
    // "number" - otherwise the word; "Gen." and "Ch." are words
    const phrases = textToPhrases('Rev. Smith, Esq., Capt. Hook, and Ms. Ames saw No. 5, Vol. 2, and Fig. 3 on Mt. Ida; I said no. ' +
        'No. Gen. Lee, Ch. 4, Col. Jr. Sr. Prof. Lt.');
    assert.deepEqual(phrases.map(wordTexts), [
        ['reverend', 'smith'], ['esquire'], ['captain', 'hook'], ['and', 'ms'], ['ames', 'saw', 'number', 'five'], ['volume', 'two'],
        ['and', 'figure', 'three', 'on', 'mount', 'ida'], ['i', 'said', 'no'], ['no'], ['gen'], ['lee'], ['ch'], ['four'],
        ['colonel', 'junior', 'senior', 'professor', 'lieutenant']
    ]);
    assert.deepEqual(textToPhrases('I said no.').map(wordTexts), [['i', 'said', 'no']]);
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

    // Commas group thousands; a four-digit number is a year
    assert.deepEqual(wordTexts(textToPhrases('23,000,000 in 1972')[0]),
        ['twenty', 'three', 'million', 'in', 'nineteen', 'seventy', 'two']);
    assert.deepEqual(textToPhrases('1,2,3456').map(wordTexts), [['one'], ['two'], ['thirty', 'four', 'fifty', 'six']]);
    assert.deepEqual(textToPhrases('12,34567').map(wordTexts),
        [['twelve'], ['thirty', 'four', 'thousand', 'five', 'hundred', 'sixty', 'seven']]);

    // Ordinals and plurals - not words (1st is not one saint)
    assert.deepEqual(textToPhrases('The 1st, 4th of the 1970s').map(wordTexts),
        [['the', 'first'], ['fourth', 'of', 'the', 'nineteen', 'seventies']]);
    assert.deepEqual(wordTexts(textToPhrases('4things 5stars')[0]), ['four', 'things', 'five', 'stars']);

    // Percentages, money (with its scale), and times (the colon isn't a phrase break)
    assert.deepEqual(wordTexts(textToPhrases('50% of $5 million')[0]), ['fifty', 'percent', 'of', 'five', 'million', 'dollars']);
    assert.deepEqual(wordTexts(textToPhrases('$5 millions')[0]), ['five', 'dollars', 'millions']);
    assert.deepEqual(wordTexts(textToPhrases('At 10:30 or 6pm')[0]), ['at', 'ten', 'thirty', 'or', 'six', 'pee', 'em']);
    assert.deepEqual(textToPhrases('Psalm 119:105').map(wordTexts), [['psalm', 'one', 'hundred', 'nineteen'], ['one', 'hundred', 'five']]);

    // a.m. and p.m. - not the word am, and not inside a word
    assert.deepEqual(wordTexts(textToPhrases('In the a.m. I am')[0]), ['in', 'the', 'ay', 'em', 'i', 'am']);
    assert.deepEqual(wordTexts(textToPhrases('At 3 amps')[0]), ['at', 'three', 'amps']);
    assert.deepEqual(textToPhrases('The a.mile').map(wordTexts), [['the', 'a'], ['mile']]);
});


test('lexicon.moneyWords', () => {
    // Dollars and cents
    assert.deepEqual(moneyWords('$1,234.56'),
        ['one', 'thousand', 'two', 'hundred', 'thirty', 'four', 'dollars', 'and', 'fifty', 'six', 'cents']);
    assert.deepEqual(moneyWords('$1.01'), ['one', 'dollar', 'and', 'one', 'cent']);
    assert.deepEqual(moneyWords(`$${'1'.repeat(16)}`), [...Array(16).fill('one'), 'dollars']);
    assert.deepEqual(moneyWords(`$${'2'.repeat(16)}.25`), [...Array(16).fill('two'), 'dollars', 'and', 'twenty', 'five', 'cents']);
    assert.deepEqual(moneyWords('$1.00'), ['one', 'dollar']);
    assert.deepEqual(moneyWords('$0.50'), ['fifty', 'cents']);
    assert.deepEqual(moneyWords('$0.00'), ['zero', 'dollars']);

    // Whole amounts, decimals, and scales
    assert.deepEqual(moneyWords('$1'), ['one', 'dollar']);
    assert.deepEqual(moneyWords('$0'), ['zero', 'dollars']);
    assert.deepEqual(moneyWords('$1972'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two', 'dollars']);
    assert.deepEqual(moneyWords('$1.5'), ['one', 'point', 'five', 'dollars']);
    assert.deepEqual(moneyWords('$1 million'), ['one', 'million', 'dollars']);
    assert.deepEqual(moneyWords('$2.75\nbillion'), ['two', 'point', 'seven', 'five', 'billion', 'dollars']);

    // Pounds and euros
    assert.deepEqual(moneyWords('£1.50'), ['one', 'pound', 'and', 'fifty', 'pence']);
    assert.deepEqual(moneyWords('£0.01'), ['one', 'penny']);
    assert.deepEqual(moneyWords('€20'), ['twenty', 'euros']);
});


test('lexicon.timeWords', () => {
    assert.deepEqual(timeWords('10:30'), ['ten', 'thirty']);
    assert.deepEqual(timeWords('10:05'), ['ten', 'oh', 'five']);
    assert.deepEqual(timeWords('12:00'), ['twelve', 'o\'clock']);
    assert.deepEqual(timeWords('14:00'), ['fourteen', 'hundred']);
    assert.deepEqual(timeWords('0:00'), ['zero', 'hundred']);

    // With a.m. or p.m.
    assert.deepEqual(timeWords('9:00 p.m.'), ['nine', 'pee', 'em']);
    assert.deepEqual(timeWords('10:30am'), ['ten', 'thirty', 'ay', 'em']);
    assert.deepEqual(timeWords('6pm'), ['six', 'pee', 'em']);
    assert.deepEqual(timeWords('7 a.m'), ['seven', 'ay', 'em']);
});


test('lexicon.numberTokenWords', () => {
    // Four-digit numbers are years; with commas, or with leading zeros, numbers
    assert.deepEqual(numberTokenWords('1972'), ['nineteen', 'seventy', 'two']);
    assert.deepEqual(numberTokenWords('1,972'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two']);
    assert.deepEqual(numberTokenWords('0972'), ['nine', 'hundred', 'seventy', 'two']);
    assert.deepEqual(numberTokenWords('19720'), ['nineteen', 'thousand', 'seven', 'hundred', 'twenty']);
    assert.deepEqual(numberTokenWords('23,000,000'), ['twenty', 'three', 'million']);

    // Past 15 digits, digit by digit (leading zeros aside)
    assert.deepEqual(numberTokenWords('999,999,999,999,999').slice(0, 5), ['nine', 'hundred', 'ninety', 'nine', 'trillion']);
    assert.deepEqual(numberTokenWords('0001234567890123456'),
        ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'one', 'two', 'three', 'four', 'five', 'six']);
    assert.equal(numberTokenWords('7'.repeat(400)).length, 400);
    assert.deepEqual(numberTokenWords('0'), ['zero']);

    // Decimals
    assert.deepEqual(numberTokenWords('3.14'), ['three', 'point', 'one', 'four']);
    assert.deepEqual(numberTokenWords('1972.5'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two', 'point', 'five']);

    // Ordinals
    assert.deepEqual(numberTokenWords('1st'), ['first']);
    assert.deepEqual(numberTokenWords('12th'), ['twelfth']);
    assert.deepEqual(numberTokenWords('22nd'), ['twenty', 'second']);
    assert.deepEqual(numberTokenWords('40th'), ['fortieth']);
    assert.deepEqual(numberTokenWords('1900th'), ['one', 'thousand', 'nine', 'hundredth']);

    // Plurals
    assert.deepEqual(numberTokenWords('1970s'), ['nineteen', 'seventies']);
    assert.deepEqual(numberTokenWords('1900s'), ['nineteen', 'hundreds']);
    assert.deepEqual(numberTokenWords('80\'s'), ['eighties']);
    assert.deepEqual(numberTokenWords('6s'), ['sixes']);

    // Percentages
    assert.deepEqual(numberTokenWords('50%'), ['fifty', 'percent']);
    assert.deepEqual(numberTokenWords('3.5%'), ['three', 'point', 'five', 'percent']);
    assert.deepEqual(numberTokenWords('1972%'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two', 'percent']);
});


test('lexicon.yearWords', () => {
    assert.deepEqual(yearWords(1972), ['nineteen', 'seventy', 'two']);
    assert.deepEqual(yearWords(1900), ['nineteen', 'hundred']);
    assert.deepEqual(yearWords(1905), ['nineteen', 'oh', 'five']);
    assert.deepEqual(yearWords(1066), ['ten', 'sixty', 'six']);
    assert.deepEqual(yearWords(2024), ['twenty', 'twenty', 'four']);

    // A millennium's first years are numbers
    assert.deepEqual(yearWords(2000), ['two', 'thousand']);
    assert.deepEqual(yearWords(2001), ['two', 'thousand', 'one']);
    assert.deepEqual(yearWords(1009), ['one', 'thousand', 'nine']);
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
    assert.deepEqual(numberWords(4e12 + 2), ['four', 'trillion', 'two']);

    // Not a number
    assert.deepEqual(numberWords(NaN), []);
});


test('lexicon.whWords', () => {
    assert(whWords.has('what'));
    assert(whWords.has('how'));
    assert(!whWords.has('is'));
});
