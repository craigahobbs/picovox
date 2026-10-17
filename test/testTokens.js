// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

import {strict as assert} from 'node:assert';
import {readFileSync} from 'node:fs';
import {setPronouncer} from '../lib/lexicon.js';
import test from 'node:test';
import {tokenize} from '../lib/tokens.js';


// A small pronouncer
const phones = {'hello': ['HH', 'AH0', 'L', 'OW1'], 'world': ['W', 'ER1', 'L', 'D'], 'why': ['W', 'AY1'], 'oy': ['OY2']};
setPronouncer((word) => phones[word]);


// The training vocabulary (picovox_train/vocab.json)
const vocabulary = JSON.parse(readFileSync(new URL('../train/src/picovox_train/vocab.json', import.meta.url), 'utf8')).tokens;


test('tokenize', () => {
    const {tokens, ids, words} = tokenize('Hello, world! Why?', vocabulary);
    assert.deepEqual(tokens, ['<bos>', 'HH', 'AH0', 'L', 'OW1', '<comma>', 'W', 'ER1', 'L', 'D', '<exclamation>', 'W', 'AY1', '<wh>']);
    assert.deepEqual(ids, tokens.map((token) => vocabulary.indexOf(token)));
    assert.deepEqual(words, [
        {'text': 'hello', 'start': 1, 'end': 5}, {'text': 'world', 'start': 6, 'end': 10}, {'text': 'why', 'start': 11, 'end': 13}
    ]);

    // Words in a phrase are separated by word boundaries
    assert.deepEqual(tokenize('hello world', vocabulary).tokens, ['<bos>', 'HH', 'AH0', 'L', 'OW1', '_', 'W', 'ER1', 'L', 'D', '<period>']);

    // No words
    assert.deepEqual(tokenize('', vocabulary), {'tokens': ['<bos>'], 'ids': [vocabulary.indexOf('<bos>')], 'words': []});
});


test('tokenize, unknown tokens', () => {
    // A token the vocabulary lacks is <unk>
    const small = ['<pad>', '<unk>', '<bos>', '_', '<period>', 'HH', 'AH0', 'L', 'OW1'];
    assert.deepEqual(tokenize('hello oy', small).ids, [2, 5, 6, 7, 8, 3, 1, 4]);

    // The same vocabulary again (its ids cached)
    assert.deepEqual(tokenize('hello', small).ids, [2, 5, 6, 7, 8, 4]);
});
