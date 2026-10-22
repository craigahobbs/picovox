// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Text to sentences for speaking - the units of synthesis. Text is split into sentences (long ones at semicolons and
// colons, and paragraphs at blank lines), each spoken after a short pause - a longer one before a paragraph - and each
// sentence's typography is made speakable (dashes as pauses, ellipses as periods, Roman numerals as numbers).

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

            // a.m. and p.m. end a sentence only before a word that isn't lowercase
            if (char === '.' && /\b[AaPp]\.[Mm]$/.test(text.slice(start, ix)) && /^\s*[a-z]/.test(text.slice(end, end + 40))) {
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

// Roman numerals elsewhere ('World War II', 'Henry VIII') - an uppercase numeral of two or more letters, written as a
// numeral is (CMXC, not XM), with an I, V, or X (not the abbreviations CC, CD, DC, MC, MD, and MM), and not the word MIX
const romanPattern = /\b[IVXLCDM]{2,}\b/g;
const romanNumeralPattern = /^(?=.*[IVX])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$/;
const romanWords = new Set(['MIX']);


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
 * A span of a sentence, from {@link quoteSpans}
 *
 * @typedef {Object} SentenceSpan
 * @property {string} text - The span's text, trimmed (a quoted span with its quotation marks)
 * @property {boolean} quoted - Whether the span is inside a quotation
 * @property {?number} quotation - The quotation's index in the text's quotations (null for an unquoted span) - a
 *     quotation's spans are its quoted spans, in one sentence or across several
 */


// A closing double quotation mark - ” or a straight quote after a non-space and before a space, punctuation, or the end
function isCloser(text, index) {
    if (text[index] === '\u201d') {
        return true;
    }
    return text[index] === '"' && /\S/.test(text[index - 1] ?? ' ') && /^(?:[\s.,;:!?\u2014-]|$)/.test(text.slice(index + 1, index + 2));
}


/**
 * Sentences' spans - each sentence split at its double quotation marks (curly or straight) into quoted and unquoted
 * spans, the quotations numbered. A quotation runs across sentences until it closes, and a paragraph starts outside one
 * - unless its first quotation mark closes one; a speech's paragraph left open continues in the next paragraph's
 * opening quotation (one speaker's quotation). Single quotation marks aren't quotations (the closing one is the
 * apostrophe).
 *
 * @param {Sentence[]} sentences - The sentences ({@link splitSentences})
 * @returns {SentenceSpan[][]} Each sentence's spans, in order
 */
export function quoteSpans(sentences) {
    let inside = false;
    let quotation = -1;
    let continued = false;
    return sentences.map((sentence, index) => {
        // A paragraph's start - outside a quotation, unless the paragraph's first quotation mark closes one (a speech left
        // open in the last paragraph continues in this one's opening quotation)
        if (sentence.paragraphStart) {
            continued = inside && /^[\u201c"]/.test(sentence.text);
            inside = false;
            for (let next = index; next < sentences.length && (next === index || !sentences[next].paragraphStart); next++) {
                const mark = sentences[next].text.search(/[\u201c\u201d"]/);
                if (mark >= 0) {
                    inside = isCloser(sentences[next].text, mark);
                    break;
                }
            }
            if (inside) {
                quotation++;
            }
        }

        // The sentence's spans
        const {text} = sentence;
        const spans = [];
        let current = '';
        const push = () => {
            const trimmed = current.trim();
            if (trimmed !== '') {
                spans.push({'text': trimmed, 'quoted': inside, 'quotation': inside ? quotation : null});
            }
            current = '';
        };
        for (let ix = 0; ix < text.length; ix++) {
            const char = text[ix];
            if (!inside && (char === '\u201c' || (char === '"' && !isCloser(text, ix)))) {
                push();
                inside = true;
                quotation += continued ? 0 : 1;
                continued = false;
                current = char;
            } else if (inside && (char === '\u201d' || char === '"')) {
                current += char;
                push();
                inside = false;
            } else {
                current += char;
            }
        }
        push();
        return spans;
    });
}


/**
 * A sentence's text for speaking - dashes as pauses, ellipses as periods, no invisible formatting characters, and
 * Roman numerals as numbers ('Chapter IV' to 'Chapter 4', 'World War II' to 'World War 2', 'World War I' to 'World War 1')
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
        replace(/^([IVXLCDM]+)\.$/, (match, roman) => `${romanValue(roman)}.`).
        replace(/\b(World War|WORLD WAR) I\b/g, '$1 1').
        replace(romanPattern, (roman) => (romanNumeralPattern.test(roman) && !romanWords.has(roman) ? `${romanValue(roman)}` : roman));
}
