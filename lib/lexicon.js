// The text front end - normalization, tokenizing into phrases, and pronunciation. The pronouncer turns a word into
// ARPAbet phones with stress: the app sets the G2P network's (setPronouncer with lib/g2p.js's loadG2p) - no dictionary at
// runtime. Without one, under Node, the CMU Pronouncing Dictionary is used - the G2P network's training labels (the voice
// model's training data is pronounced by the network, as the app pronounces - nn/frontend.py, nn/js/tokenizeLines.js).

import {isVowel} from './phonemes.js';


// The pronouncer (null - the dictionary), and the pronouncing dictionary - word to ARPAbet phones (first pronunciation)
let pronouncer = null;
let dictionary = null;


// Set the pronouncer - pronounce(word) returns the word's ARPAbet phones with stress (null restores the dictionary)
export function setPronouncer(pronounce) {
    pronouncer = pronounce;
}


// Set the pronouncing dictionary from the text of a CMU Pronouncing Dictionary file (cmudict.dict) - the caller loads it
// (in the browser, with fetch)
export function setDictionary(text) {
    dictionary = new Map();
    for (const line of text.split('\n')) {
        const hash = line.indexOf('#');
        const content = (hash >= 0 ? line.slice(0, hash) : line).trim();
        if (content !== '') {
            const [word, ...phones] = content.split(/\s+/);
            if (!word.includes('(')) {
                dictionary.set(word, phones);
            }
        }
    }
}


// Get the pronouncer - the one set, else the dictionary's (under Node, data/cmudict.dict is loaded on first use - no static
// node:fs import, so the module also loads in the browser)
function getPronouncer() {
    if (pronouncer !== null) {
        return pronouncer;
    }
    if (dictionary === null) {
        const fs = globalThis.process?.getBuiltinModule?.('node:fs');
        if (!fs) {
            throw new Error('No pronouncer - call setPronouncer(pronounce) first');
        }
        setDictionary(fs.readFileSync(new URL('../data/cmudict.dict', import.meta.url), 'utf8'));
    }
    return (word) => dictionary.get(word) || pronounceUnknown(word);
}


// Function words - unaccented, with reduced stress
const functionWords = new Set([
    'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'nor', 'is', 'are', 'was', 'were', 'be',
    'been', 'am', 'it', 'its', 'that', 'this', 'with', 'as', 'by', 'from', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'do', 'does', 'did', 'have', 'has', 'had', 'will',
    'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must', 'so', 'if', 'than', 'then', 'into', 'onto',
    'up', 'out', 'over', 'about', 'there', 'some', 'just'
]);


// Question words - a question starting with one ends with a fall
export const whWords = new Set(['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']);


// Abbreviations
const abbreviations = {'mr': 'mister', 'mrs': 'missus', 'dr': 'doctor', 'st': 'saint', 'vs': 'versus', 'etc': 'etcetera'};


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
    for (const [size, name] of [[1e9, 'billion'], [1e6, 'million'], [1e3, 'thousand'], [100, 'hundred']]) {
        if (value >= size) {
            const rest = value % size;
            return [...numberWords(Math.floor(value / size)), name, ...(rest === 0 ? [] : numberWords(rest))];
        }
    }
    return [];
}


// Pronounce a word not in the dictionary (training-time labels) - strip common suffixes, else map letters naively
function pronounceUnknown(word) {
    const dict = dictionary;
    const suffixes = [['s', ['Z']], ['es', ['IH0', 'Z']], ['ed', ['D']], ['ing', ['IH0', 'NG']], ['er', ['ER0']], ['ly', ['L', 'IY0']]];
    for (const [suffix, phones] of suffixes) {
        if (word.endsWith(suffix) && dict.has(word.slice(0, -suffix.length))) {
            return [...dict.get(word.slice(0, -suffix.length)), ...phones];
        }
    }
    const letters = {'a': ['AE1'], 'b': ['B'], 'c': ['K'], 'd': ['D'], 'e': ['EH1'], 'f': ['F'], 'g': ['G'], 'h': ['HH'], 'i': ['IH1'],
        'j': ['JH'], 'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['AA1'], 'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'],
        't': ['T'], 'u': ['AH1'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'], 'z': ['Z']};
    const phones = [];
    for (const letter of word) {
        if (letter in letters && (phones.length === 0 || letters[letter][0] !== phones[phones.length - 1])) {
            phones.push(...letters[letter]);
        }
    }
    return phones;
}


// Tokenize text into phrases - [{words: [{text, phones: [{ph, stress}], isFunction, syllables}], end, major}]
// The phrase end is 'comma', 'period', 'exclamation', 'question' (yes/no), or 'wh' (a question-word question). A comma
// phrase ended by a semicolon or colon is a major break (major true).
export function textToPhrases(text) {
    const pronounce = getPronouncer();
    const tokens = text.toLowerCase().replace(/[‘’]/g, '\'').match(/[a-z][a-z']*\.?|\d+(?:\.\d+)?|[.,;:!?]/g) || [];
    const phrases = [];
    let words = [];
    const endPhrase = (end, major = false) => {
        if (words.length !== 0) {
            const isQuestion = end === 'question';
            phrases.push({words, 'end': isQuestion && whWords.has(words[0].text) ? 'wh' : end, major});
            words = [];
        }
    };
    const addWord = (wordText) => {
        const word = wordText.replace(/^'+|'+$/g, '');
        const phones = pronounce(word).map((phone) => {
            const match = phone.match(/^([A-Z]+)(\d)?$/);
            return {'ph': match[1], 'stress': match[2] === undefined ? null : Number(match[2])};
        });
        const isFunction = functionWords.has(word);
        if (isFunction) {
            for (const phone of phones) {
                if (phone.stress !== null) {
                    phone.stress = 0;
                }
            }
        }
        words.push({'text': word, phones, isFunction, 'syllables': phones.filter((phone) => isVowel(phone.ph)).length});
    };
    for (const token of tokens) {
        if (/^\d/.test(token)) {
            const [whole, fraction] = token.split('.');
            for (const word of numberWords(Number(whole))) {
                addWord(word);
            }
            if (fraction !== undefined) {
                addWord('point');
                for (const digit of fraction) {
                    addWord(ones[Number(digit)]);
                }
            }
        } else if (/^[a-z]/.test(token)) {
            const bare = token.replace(/\.$/, '');
            if (bare in abbreviations) {
                addWord(abbreviations[bare]);
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
        } else {
            endPhrase('comma', token === ';' || token === ':');
        }
    }
    endPhrase('period');
    return phrases;
}
