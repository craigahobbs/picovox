// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {strict as assert} from 'node:assert';
import {autoStyles} from '../lib/style.js';
import {splitSentences} from '../lib/sentences.js';
import test from 'node:test';


test('autoStyles', () => {
    // The quotations in the dialogue style - an exclaimed quoted sentence in the exclaim style - the narration neutral,
    // adjacent spans of one style merged; a sentence without a quotation has no spans
    const sentences = splitSentences('It was late. \u201cHello,\u201d \u201cthere,\u201d she said.\n\n\u201cRun! Now!\u201d');
    const auto = autoStyles(sentences);
    assert.equal(auto.spans(0), null);
    assert.deepEqual(auto.spans(1), [
        {'text': '\u201cHello,\u201d \u201cthere,\u201d', 'style': 'dialogue'}, {'text': 'she said.', 'style': 'neutral'}
    ]);
    assert.deepEqual(auto.spans(2), [{'text': '\u201cRun!', 'style': 'exclaim'}]);
    assert.deepEqual(auto.spans(3), [{'text': 'Now!\u201d', 'style': 'exclaim'}]);
    const mixed = splitSentences('\u201cOh,\u201d she said, \u201cfine!\u201d');
    assert.deepEqual(autoStyles(mixed).spans(0), [
        {'text': '\u201cOh,\u201d', 'style': 'dialogue'}, {'text': 'she said,', 'style': 'neutral'},
        {'text': '\u201cfine!\u201d', 'style': 'exclaim'}
    ]);
});

