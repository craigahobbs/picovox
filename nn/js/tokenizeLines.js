// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// Tokenize texts with nn/js/tokens.js - one JSON string per input line in, one JSON line of {tokens, ids, words} out
// (used by `python -m nn.data verify-tokens` to compare with the Python front end)
//
//     node nn/js/tokenizeLines.js < texts.jsonl > tokens.jsonl

import {readFileSync, writeSync} from 'node:fs';
import {tokenize} from './tokens.js';


// Tokenize each line of standard input with the training vocabulary (nn/vocab.json)
const vocabulary = JSON.parse(readFileSync(new URL('../vocab.json', import.meta.url), 'utf8')).tokens;
const lines = readFileSync(0, 'utf8').split('\n').filter((line) => line.trim() !== '');
const output = lines.map((line) => JSON.stringify(tokenize(JSON.parse(line), vocabulary)));
writeSync(1, output.join('\n') + (output.length ? '\n' : ''));
