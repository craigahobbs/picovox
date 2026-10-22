// The text front end - normalization, tokenizing into phrases, and pronunciation. The pronouncer turns a word into
// ARPAbet phones with stress - the G2P network's (setPronouncer with lib/g2p.js's loadG2p): every word is pronounced by
// the network, with no dictionary at runtime (the voice model's training data is pronounced by it too - picovox_train/frontend.py,
// nn/js/tokenizeLines.js).

import {isVowel} from './phonemes.js';


// The pronouncer
let pronouncer = null;


// Set the pronouncer - pronounce(word, reading) returns the word's ARPAbet phones with stress in a reading (0, the
// default); with ambiguous(word) and chooseReading(items, index) (lib/g2p.js loadG2p), the reading of an ambiguous
// word is chosen from its context (null unsets it)
export function setPronouncer(pronounce) {
    pronouncer = pronounce;
}


// Get the pronouncer - the one set
function getPronouncer() {
    if (pronouncer === null) {
        throw new Error('No pronouncer - call setPronouncer(pronounce) first');
    }
    return pronouncer;
}


// Function words - unaccented, with reduced stress
const functionWords = new Set([
    'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'nor', 'is', 'are', 'was', 'were', 'be',
    'been', 'am', 'it', 'its', 'that', 'this', 'with', 'as', 'by', 'from', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'do', 'does', 'did', 'have', 'has', 'had', 'will',
    'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must', 'so', 'if', 'than', 'then', 'into', 'onto',
    'up', 'out', 'over', 'about', 'there', 'some', 'just'
]);


// A text's accented Latin letters without their accents - á as a, Ñ as N, and the ligatures as English says them
// (Cæsar, Œdipus; ß as s) - each letter one letter, so the text keeps its length (the front end's words are letters a
// to z)
const plainLetterMap = {'æ': 'e', 'Æ': 'E', 'œ': 'e', 'Œ': 'E', 'ø': 'o', 'Ø': 'O', 'ł': 'l', 'Ł': 'L', 'đ': 'd', 'Đ': 'D', 'ß': 's'};
function plainLetters(text) {
    return text.replace(/[\u00c0-\u024f]/g, (char) => plainLetterMap[char] ?? char.normalize('NFD').replace(/[\u0300-\u036f]/g, ''));
}


// Question words - a question starting with one ends with a fall
export const whWords = new Set(['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']);


// Abbreviations
const abbreviations = {'mr': 'mister', 'mrs': 'missus', 'dr': 'doctor', 'st': 'saint', 'vs': 'versus', 'etc': 'etcetera'};

// Abbreviations only with their period - the months and days ("mon" is also French, so "Mon." only; "mar", "sat",
// "sun", and "wed" are words, so they aren't abbreviations)
const periodAbbreviations = {
    'jan': 'january', 'feb': 'february', 'apr': 'april', 'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'sept': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december', 'mon': 'monday', 'tue': 'tuesday',
    'tues': 'tuesday', 'thu': 'thursday', 'thur': 'thursday', 'thurs': 'thursday', 'fri': 'friday'
};


// Number words
const ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];


// Spell an integer in words
export function numberWords(value) {
    if (value < 20) {
        return [ones[value]];
    }
    if (value < 100) {
        return value % 10 === 0 ? [tens[Math.floor(value / 10)]] : [tens[Math.floor(value / 10)], ones[value % 10]];
    }
    for (const [size, name] of [[1e12, 'trillion'], [1e9, 'billion'], [1e6, 'million'], [1e3, 'thousand'], [100, 'hundred']]) {
        if (value >= size) {
            const rest = value % size;
            return [...numberWords(Math.floor(value / size)), name, ...(rest === 0 ? [] : numberWords(rest))];
        }
    }
    return [];
}


// Spell a year in words - by its hundreds (1972 as nineteen seventy-two, 1900 as nineteen hundred, 1905 as nineteen oh
// five), but a millennium's first years as a number (2000, 2001 as two thousand one)
export function yearWords(value) {
    const hundreds = Math.floor(value / 100);
    const rest = value % 100;
    if (hundreds % 10 === 0 && rest < 10) {
        return numberWords(value);
    }
    return [...numberWords(hundreds), ...(rest === 0 ? ['hundred'] : (rest < 10 ? ['oh', ones[rest]] : numberWords(rest)))];
}


// A number word as an ordinal (twenty as twentieth) or a plural (twenty as twenties)
const ordinalWords = {
    'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth', 'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'
};
function ordinalWord(word) {
    return ordinalWords[word] ?? (word.endsWith('y') ? `${word.slice(0, -1)}ieth` : `${word}th`);
}
function pluralWord(word) {
    return word.endsWith('y') ? `${word.slice(0, -1)}ies` : (word === 'six' ? 'sixes' : `${word}s`);
}


// A decimal's fraction in words - digit by digit after the point
function fractionWords(fraction) {
    return ['point', ...[...fraction].map((digit) => ones[Number(digit)])];
}


// A number token's words - its digits (commas grouping its thousands) in words, a four-digit number as a year, a
// decimal's fraction digit by digit, a percentage, and an ordinal (1st, 22nd) or a plural (1970s, 70's) as its last
// word's
export function numberTokenWords(token) {
    const [, digits, fraction, suffix] = token.match(/^([\d,]+)(?:\.(\d+))?(\D*)$/);
    const value = Number(digits.replace(/,/g, ''));
    const isYear = /^[1-9]\d{3}$/.test(digits) && fraction === undefined && (suffix === '' || suffix.endsWith('s'));
    const words = isYear ? yearWords(value) : numberWords(value);
    if (fraction !== undefined) {
        words.push(...fractionWords(fraction));
    }
    if (suffix === '%') {
        words.push('percent');
    } else if (suffix.endsWith('s')) {
        words.push(pluralWord(words.pop()));
    } else if (suffix !== '') {
        words.push(ordinalWord(words.pop()));
    }
    return words;
}


// The currencies' words - the unit, its plural, the hundredth, its plural
const currencies = {
    '$': ['dollar', 'dollars', 'cent', 'cents'], '£': ['pound', 'pounds', 'penny', 'pence'], '€': ['euro', 'euros', 'cent', 'cents']
};


// An amount of money's words - $1,234.56 as one thousand two hundred thirty-four dollars and fifty-six cents, $0.50 as
// fifty cents, $1.5 as one point five dollars, and $5 million as five million dollars
export function moneyWords(token) {
    const [, symbol, digits, fraction, scale] = token.match(/^(\D)([\d,]+)(?:\.(\d+))?(?:\s+([a-z]+))?$/);
    const [unit, units, hundredth, hundredths] = currencies[symbol];
    const value = Number(digits.replace(/,/g, ''));
    if (fraction?.length === 2 && scale === undefined) {
        const cents = Number(fraction);
        const words = value === 0 && cents !== 0 ? [] : [...numberWords(value), value === 1 ? unit : units];
        if (cents !== 0) {
            words.push(...(words.length === 0 ? [] : ['and']), ...numberWords(cents), cents === 1 ? hundredth : hundredths);
        }
        return words;
    }
    const words = numberWords(value);
    if (fraction !== undefined) {
        words.push(...fractionWords(fraction));
    }
    if (scale !== undefined) {
        words.push(scale);
    }
    words.push(value === 1 && fraction === undefined && scale === undefined ? unit : units);
    return words;
}


// The letters of a.m. and p.m. as words
const meridiemWords = {'a': ['ay', 'em'], 'p': ['pee', 'em']};


// A time's words - 10:30 as ten thirty, 10:05 as ten oh five, 10:00 as ten o'clock (14:00 as fourteen hundred), and
// with a.m. or p.m. (10:00 p.m. and 10pm as ten pee em)
export function timeWords(token) {
    const [, hours, minutes, meridiem] = token.match(/^(\d+)(?::(\d+))?[ \t]*(?:([ap])\.?m\.?)?$/);
    const words = numberWords(Number(hours));
    if (minutes !== undefined && Number(minutes) !== 0) {
        words.push(...(Number(minutes) < 10 ? ['oh', ones[Number(minutes)]] : numberWords(Number(minutes))));
    } else if (meridiem === undefined) {
        words.push(Number(hours) >= 1 && Number(hours) <= 12 ? 'o\'clock' : 'hundred');
    }
    return meridiem === undefined ? words : [...words, ...meridiemWords[meridiem]];
}


// The context marks of the phrase ends - a phrase end's item in a text's items (the G2P's context network reads them)
const phraseMarks = {'comma': ',', 'period': '.', 'question': '?', 'exclamation': '!'};


// The text's tokens - a.m. or p.m., a word, a time (10:30, 10:30 a.m., 10pm), an amount of money (with its scale - $5
// million), a number (its thousands grouped by commas or not; with a decimal fraction, a percent sign, or an ordinal or
// plural suffix), or a punctuation mark (a parenthesis too)
const digitsPattern = String.raw`(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)`;
const meridiemPattern = String.raw`[ap]\.?m\.?(?![a-z])`;
const tokenPattern = new RegExp([
    String.raw`[ap]\.m\.?(?![a-z])`,
    String.raw`[a-z][a-z']*\.?`,
    String.raw`\d{1,2}(?::\d{2}(?!\d)(?:[ \t]*${meridiemPattern})?|[ \t]*${meridiemPattern})`,
    String.raw`[$£€]${digitsPattern}(?:\.\d+)?(?:[ \t\r\n]+(?:thousand|million|billion|trillion)(?![a-z]))?`,
    String.raw`${digitsPattern}(?:\.\d+%?|%|(?:st|nd|rd|th|'?s)(?![a-z]))?`,
    '[.,;:!?()]'
].join('|'), 'g');


// Tokenize text into phrases - [{words: [{text, phones: [{ph, stress}], isFunction, syllables}], end, major}]
// The phrase end is 'comma', 'period', 'exclamation', 'question' (yes/no), or 'wh' (a question-word question). A comma
// phrase ended by a semicolon or colon is a major break (major true). A parenthesis - opening or closing an aside - is a
// comma's break, unless a punctuation mark right after it ends the phrase ("(1818)." ends with a period, "(1818);" with a
// semicolon's major break). A word the pronouncer finds ambiguous (the G2P network's ambiguity head -
// pronounce.ambiguous) takes the reading its context network chooses from the text's items (pronounce.chooseReading).
export function textToPhrases(text) {
    const pronounce = getPronouncer();
    const tokens = plainLetters(text).toLowerCase().replace(/[‘’]/g, '\'').match(tokenPattern) || [];

    // The text's items - its words, and its phrase ends ({end, major}, only after a word - a parenthesis's break gives way
    // to a punctuation mark right after it)
    const items = [];
    let parenthesisEnd = false;
    const endPhrase = (end, major = false, parenthesis = false) => {
        if (items.length !== 0 && typeof items[items.length - 1] === 'string') {
            items.push({end, major});
            parenthesisEnd = parenthesis;
        } else if (parenthesisEnd && !parenthesis) {
            items[items.length - 1] = {end, major};
            parenthesisEnd = false;
        }
    };
    const addWord = (word) => {
        items.push(word.replace(/^'+|'+$/g, ''));
    };
    for (const token of tokens) {
        if (/^\d+(?::|[ \t]*[ap])/.test(token)) {
            timeWords(token).forEach(addWord);
        } else if (/^\d/.test(token)) {
            numberTokenWords(token).forEach(addWord);
        } else if (/^[$£€]/.test(token)) {
            moneyWords(token).forEach(addWord);
        } else if (/^[ap]\.m/.test(token)) {
            meridiemWords[token[0]].forEach(addWord);
        } else if (/^[a-z]/.test(token)) {
            const bare = token.replace(/\.$/, '');
            if (Object.hasOwn(abbreviations, bare)) {
                addWord(abbreviations[bare]);
            } else if (token.endsWith('.') && Object.hasOwn(periodAbbreviations, bare)) {
                addWord(periodAbbreviations[bare]);
            } else {
                addWord(bare);
                if (token.endsWith('.')) {
                    endPhrase('period');
                }
            }
        } else if (token === '?') {
            endPhrase('question');
        } else if (token === '!') {
            endPhrase('exclamation');
        } else if (token === '.') {
            endPhrase('period');
        } else if (token === '(' || token === ')') {
            endPhrase('comma', false, true);
        } else {
            endPhrase('comma', token === ';' || token === ':');
        }
    }
    endPhrase('period');

    // The phrases - each word pronounced (in the reading its context chooses, if ambiguous)
    const context = items.map((item) => (typeof item === 'string' ? item : phraseMarks[item.end]));
    const phrases = [];
    let words = [];
    for (const [index, item] of items.entries()) {
        if (typeof item !== 'string') {
            const isQuestion = item.end === 'question';
            phrases.push({words, 'end': isQuestion && whWords.has(words[0].text) ? 'wh' : item.end, 'major': item.major});
            words = [];
            continue;
        }
        const reading = pronounce.ambiguous?.(item) ? pronounce.chooseReading(context, index) : 0;
        const phones = pronounce(item, reading).map((phone) => {
            const match = phone.match(/^([A-Z]+)(\d)?$/);
            return {'ph': match[1], 'stress': match[2] === undefined ? null : Number(match[2])};
        });
        const isFunction = functionWords.has(item);
        if (isFunction) {
            for (const phone of phones) {
                if (phone.stress !== null) {
                    phone.stress = 0;
                }
            }
        }
        words.push({'text': item, phones, isFunction, 'syllables': phones.filter((phone) => isVowel(phone.ph)).length});
    }
    return phrases;
}
