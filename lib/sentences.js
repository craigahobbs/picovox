// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Text to sentences for speaking - the units of synthesis. Text is split into sentences (long ones at semicolons and
// colons, and paragraphs at blank lines), each spoken after a short pause - a longer one before a paragraph - and each
// sentence's typography is made speakable (dashes as pauses, ellipses as periods, Roman-numeral headings as numbers).

/** @module picovox/lib/sentences */


// Sentences longer than this many characters are split at semicolons and colons (phrase boundaries)
const longSentenceChars = 200;


// Abbreviations that don't end sentences
const abbreviations = new Set([
    'mr', 'mrs', 'ms', 'dr', 'st', 'vs', 'etc', 'jr', 'sr', 'prof', 'mt', 'no', 'vol', 'fig', 'ch', 'esq', 'rev', 'capt', 'col', 'gen', 'lt'
]);


/**
 * A sentence of text, from {@link splitSentences}
 *
 * @typedef {Object} Sentence
 * @property {number} index - The sentence's index in the text's sentences
 * @property {number} start - The sentence's start offset in the text
 * @property {number} end - The sentence's end offset in the text
 * @property {string} text - The sentence's text, trimmed, with its runs of whitespace as single spaces
 * @property {boolean} paragraphStart - Whether the sentence starts a paragraph (the text's first sentence does)
 */


/**
 * Split text into sentences - long sentences at semicolons and colons, and paragraphs at blank lines. Text without a
 * letter or a digit isn't a sentence.
 *
 * @param {string} text - The text
 * @returns {Sentence[]} The sentences
 */
export function splitSentences(text) {
    const sentences = [];
    let start = 0;
    let paragraphStart = true;
    const addSentence = (end, endsParagraph) => {
        const sentenceText = text.slice(start, end);
        if (/[A-Za-z0-9]/.test(sentenceText)) {
            const leading = sentenceText.length - sentenceText.trimStart().length;
            sentences.push({
                'index': sentences.length, 'start': start + leading, 'end': start + sentenceText.trimEnd().length,
                'text': sentenceText.trim().replace(/\s+/g, ' '), paragraphStart
            });
            paragraphStart = false;
        }
        if (endsParagraph) {
            paragraphStart = true;
        }
        start = end;
    };
    for (let ix = 0; ix < text.length; ix++) {
        const char = text[ix];
        if (char === '\n' && /^\n[ \t\r]*\n/.test(text.slice(ix, ix + 40))) {
            addSentence(ix, true);
        } else if ((char === ';' || char === ':') && ix - start > longSentenceChars && /\s/.test(text[ix + 1] || ' ')) {
            addSentence(ix + 1, false);
        } else if ('.!?'.includes(char)) {
            let end = ix + 1;
            while (end < text.length && '.!?"\')]”’'.includes(text[end])) {
                end++;
            }
            if (end < text.length && !/\s/.test(text[end])) {
                continue;
            }
            const word = text.slice(Math.max(start, ix - 20), ix).match(/([A-Za-z]+)$/);
            if (char === '.' && word !== null && (abbreviations.has(word[1].toLowerCase()) || /^[A-Z]$/.test(word[1]))) {
                continue;
            }
            addSentence(end, false);
            ix = end - 1;
        }
    }
    addSentence(text.length, true);
    return sentences;
}


// Headings numbered with Roman numerals ('Chapter IV') - the label and the numeral
const headingLabels = [
    'Chapter', 'CHAPTER', 'Book', 'BOOK', 'Part', 'PART', 'Volume', 'VOLUME', 'Act', 'ACT', 'Scene', 'SCENE', 'Stave', 'STAVE',
    'Letter', 'LETTER', 'Canto', 'CANTO', 'Section', 'SECTION'
];
const romanHeadingPattern = new RegExp(`\\b(${headingLabels.join('|')})\\s+([IVXLCDM]+)\\b`, 'g');


// A Roman numeral's value
function romanValue(roman) {
    const values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000};
    let total = 0;
    for (let ix = 0; ix < roman.length; ix++) {
        const value = values[roman[ix]];
        total += ix + 1 < roman.length && values[roman[ix + 1]] > value ? -value : value;
    }
    return total;
}


/**
 * A sentence's text for speaking - dashes as pauses, ellipses as periods, no invisible formatting characters, and
 * Roman-numeral headings as numbers ('Chapter IV' to 'Chapter 4')
 *
 * @param {string} text - The sentence's text
 * @returns {string} The text to speak
 */
export function speechText(text) {
    return text.
        replace(/[\u2060\u00ad\u200b\ufeff]/g, '').
        replace(/[\u200a\u2009\u202f\u00a0]/g, ' ').
        replace(/\s*[\u2014\u2013]+\s*/g, ', ').
        replace(/\u2026/g, '...').
        replace(romanHeadingPattern, (match, label, roman) => `${label} ${romanValue(roman)}`).
        replace(/^([IVXLCDM]+)\.$/, (match, roman) => `${romanValue(roman)}.`);
}
