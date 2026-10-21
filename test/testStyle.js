// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {autoStyles, paragraphBeat, paragraphBeats} from '../lib/style.js';
import {strict as assert} from 'node:assert';
import {splitSentences} from '../lib/sentences.js';
import test from 'node:test';


test('autoStyles', () => {
    // The quotations in the dialogue style, the narration neutral, adjacent spans of one style merged; a sentence without
    // a quotation has no spans
    const sentences = splitSentences('It was late. \u201cHello,\u201d \u201cthere,\u201d she said.\n\n\u201cRun! Now!\u201d');
    const auto = autoStyles(sentences);
    assert.equal(auto.spans(0), null);
    assert.deepEqual(auto.spans(1), [
        {'text': '\u201cHello,\u201d \u201cthere,\u201d', 'style': 'dialogue'}, {'text': 'she said.', 'style': 'neutral'}
    ]);
    assert.deepEqual(auto.spans(2), [{'text': '\u201cRun!', 'style': 'dialogue'}]);
    assert.deepEqual(auto.spans(3), [{'text': 'Now!\u201d', 'style': 'dialogue'}]);
});


test('paragraphBeats', () => {
    // Each paragraph as one arc - its first sentence lifted and louder, the pitch settling to -lift/2 by its last
    // sentence, which is slowed; a one-sentence paragraph is only lifted
    const {lift, level, settle} = paragraphBeat;
    const beats = paragraphBeats(splitSentences('One. Two. Three.\n\nFour.'));
    assert.deepEqual(beats, [
        {'transforms': [{'pitchScale': 1 + lift}, {level}, {'frication': level}], 'rate': 1},
        {'transforms': [{'pitchScale': 1 + lift - 1.5 * lift * 0.5}], 'rate': 1},
        {'transforms': [{'pitchScale': 1 + lift - 1.5 * lift}], 'rate': 1 - settle},
        {'transforms': [{'pitchScale': 1 + lift}, {level}, {'frication': level}], 'rate': 1}
    ]);

    // Sentences without a paragraph start begin one
    assert.deepEqual(paragraphBeats([{'index': 0, 'paragraphStart': false}]), [beats[3]]);
    assert.deepEqual(paragraphBeats([]), []);
});
