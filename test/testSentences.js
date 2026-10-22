// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {quoteSpans, speechText, splitSentences} from '../lib/sentences.js';
import {strict as assert} from 'node:assert';
import test from 'node:test';


// The sentences' texts and paragraph starts
function sentenceTexts(text) {
    return splitSentences(text).map((sentence) => [sentence.text, sentence.paragraphStart]);
}


test('splitSentences', () => {
    const text = '  Call me Ishmael.  Some years ago - never mind how long!\nIt was?  Yes.\n\n  \nA new paragraph.';
    assert.deepEqual(splitSentences(text), [
        {'index': 0, 'start': 2, 'end': 18, 'text': 'Call me Ishmael.', 'paragraphStart': true},
        {'index': 1, 'start': 20, 'end': 57, 'text': 'Some years ago - never mind how long!', 'paragraphStart': false},
        {'index': 2, 'start': 58, 'end': 65, 'text': 'It was?', 'paragraphStart': false},
        {'index': 3, 'start': 67, 'end': 71, 'text': 'Yes.', 'paragraphStart': false},
        {'index': 4, 'start': 76, 'end': 92, 'text': 'A new paragraph.', 'paragraphStart': true}
    ]);
    assert.equal(text.slice(20, 57), 'Some years ago - never mind how long!');
    assert.equal(text.slice(76, 92), 'A new paragraph.');
});


test('splitSentences, empty', () => {
    assert.deepEqual(splitSentences(''), []);
    assert.deepEqual(splitSentences(' ... -- !\n\n?'), []);

    // A paragraph with no letters or digits doesn't start the next sentence's paragraph twice
    assert.deepEqual(sentenceTexts('One.\n\n***\n\nTwo.'), [['One.', true], ['Two.', true]]);
});


test('splitSentences, closing quotes and brackets', () => {
    assert.deepEqual(sentenceTexts('"Stop!" she cried. (It was late.) \u201cWhy?\u201d \u2018Yes.\u2019 Go...'), [
        ['"Stop!"', true], ['she cried.', false], ['(It was late.)', false], ['\u201cWhy?\u201d', false], ['\u2018Yes.\u2019', false],
        ['Go...', false]
    ]);
});


test('splitSentences, not sentence ends', () => {
    // A period inside a word or number, abbreviations, and initials
    assert.deepEqual(sentenceTexts('It cost 3.50 at example.com today. Mr. Smith and Dr. Jones met J. R. Tolkien. no. 5 vs. 6.'), [
        ['It cost 3.50 at example.com today.', true], ['Mr. Smith and Dr. Jones met J. R. Tolkien.', false], ['no. 5 vs. 6.', false]
    ]);

    // An abbreviation is case-insensitive; a lone lowercase letter ends a sentence
    assert.deepEqual(sentenceTexts('See VOL. 2. Plan a. Then b.'), [['See VOL. 2.', true], ['Plan a.', false], ['Then b.', false]]);

    // a.m. and p.m. end a sentence only before a word that isn't lowercase
    assert.deepEqual(sentenceTexts('At 5 p.m. on Monday we met. At 9 a.m. Then 6 P.M. sharp. Go.'), [
        ['At 5 p.m. on Monday we met.', true], ['At 9 a.m.', false], ['Then 6 P.M. sharp.', false], ['Go.', false]
    ]);
});


test('splitSentences, long sentences', () => {
    // A sentence longer than 200 characters is split at a semicolon or a colon followed by a space
    const long = `${'word '.repeat(45)}end; ${'more '.repeat(10)}done: last;tight; and the rest.`;
    assert.deepEqual(splitSentences(long).map((sentence) => sentence.text), [
        `${'word '.repeat(45)}end;`,
        `${'more '.repeat(10)}done: last;tight; and the rest.`
    ]);

    // Short sentences aren't split
    assert.deepEqual(sentenceTexts('One; two: three.'), [['One; two: three.', true]]);

    // A semicolon at the end of the text
    assert.deepEqual(splitSentences(`${'word '.repeat(45)}end;`).map((sentence) => sentence.text), [`${'word '.repeat(45)}end;`]);
});


test('splitSentences, whitespace', () => {
    // Runs of whitespace become single spaces; a blank line with spaces or a carriage return is a paragraph break
    assert.deepEqual(sentenceTexts('One\n  two\tthree.\r\n \t\r\nFour.'), [['One two three.', true], ['Four.', true]]);
    assert.deepEqual(sentenceTexts('No end'), [['No end', true]]);
});


test('quoteSpans', () => {
    // Each sentence's quoted and unquoted spans, the quotations numbered
    const spans = (text) => quoteSpans(splitSentences(text)).map((sentence) => sentence.map((span) => [span.text, span.quotation]));
    assert.deepEqual(spans('\u201cMy dear,\u201d said his lady, \u201chave you heard?\u201d'), [
        [['\u201cMy dear,\u201d', 0], ['said his lady,', null], ['\u201chave you heard?\u201d', 1]]
    ]);

    // Straight quotation marks - opening after a space, closing before one or punctuation; a quotation ending its sentence
    assert.deepEqual(spans('"Who is there?" she asked. Nobody answered.'), [[['"Who is there?"', 0]], [['she asked.', null]],
        [['Nobody answered.', null]]]);
    assert.deepEqual(spans('He said "go" and left.'), [[['He said', null], ['"go"', 0], ['and left.', null]]]);
    assert.deepEqual(spans('A "b"c d.'), [[['A', null], ['"b"', 0], ['c d.', null]]]);

    // A quotation across sentences, and a speech left open in its paragraph continuing in the next one's opening quotation
    assert.deepEqual(spans('\u201cI am going. It is late.\n\n\u201cAnd I am staying,\u201d she said.'), [
        [['\u201cI am going.', 0]], [['It is late.', 0]], [['\u201cAnd I am staying,\u201d', 0], ['she said.', null]]
    ]);

    // A paragraph whose first quotation mark closes one starts inside it - curly or straight
    assert.deepEqual(spans('He began: \u201cIt was dark.\n\nAnd cold,\u201d he said.'), [
        [['He began:', null], ['\u201cIt was dark.', 0]], [['And cold,\u201d', 1], ['he said.', null]]
    ]);
    assert.deepEqual(spans('He began: "It was dark.\n\nAnd cold," he said.'), [
        [['He began:', null], ['"It was dark.', 0]], [['And cold,"', 1], ['he said.', null]]
    ]);

    // Single quotation marks aren't quotations; no sentences, no spans
    assert.deepEqual(spans('It\u2019s \u2018single\u2019 quoted.'), [[['It\u2019s \u2018single\u2019 quoted.', null]]]);
    assert.deepEqual(quoteSpans([]), []);
    assert.deepEqual(quoteSpans(splitSentences('\u201c')), []);
});


test('speechText', () => {
    // Invisible formatting characters are removed, thin and no-break spaces are spaces
    assert.equal(speechText('in\u00advisi\u200bble\u2060\ufeff'), 'invisible');
    assert.equal(speechText('a\u00a0b\u2009c\u200ad\u202fe'), 'a b c d e');

    // Dashes are pauses, ellipses are periods
    assert.equal(speechText('It was \u2014 so to speak \u2014 late\u2013ish'), 'It was, so to speak, late, ish');
    assert.equal(speechText('Wait\u2026 what?'), 'Wait... what?');

    // Roman-numeral headings are numbers
    assert.equal(speechText('Chapter IV. The End'), 'Chapter 4. The End');
    assert.equal(speechText('BOOK XLII and Stave MCMXCIX'), 'BOOK 42 and Stave 1999');
    assert.equal(speechText('XIV.'), '14.');
    assert.equal(speechText('I am here.'), 'I am here.');
    assert.equal(speechText('Chapter Ten'), 'Chapter Ten');

    // Other Roman numerals of two or more letters are numbers - not the pronoun I, the abbreviations, the word MIX, or
    // letters that aren't a numeral
    assert.equal(speechText('World War II, Henry VIII, and Louis XIV in MDCCCXCII.'), 'World War 2, Henry 8, and Louis 14 in 1892.');
    assert.equal(speechText('I saw DC, CD, MM, and MIX.'), 'I saw DC, CD, MM, and MIX.');
    assert.equal(speechText('IIII, XM, VIVID, Civil, and ii'), 'IIII, XM, VIVID, Civil, and ii');

    // World War I is one, not the pronoun
    assert.equal(speechText('World War I, WORLD WAR I, and War I'), 'World War 1, WORLD WAR 1, and War I');
});
