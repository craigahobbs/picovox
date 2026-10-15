// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// Tokenize texts with nn/js/tokens.js, every word pronounced by the app's G2P network (lib/g2p.js with
// lib/models/g2pModel.js) - one JSON string per input line in, one JSON line of {tokens, ids, words} out (used by
// `python -m nn.data verify-tokens` to compare with the Python front end)
//
//     node nn/js/tokenizeLines.js < texts.jsonl > tokens.jsonl

import {Buffer} from 'node:buffer';
import {g2pModel} from '../../lib/models/g2pModel.js';
import {loadG2p} from '../../lib/g2p.js';
import process from 'node:process';
import {readFileSync} from 'node:fs';
import {setPronouncer} from '../../lib/lexicon.js';
import {tokenize} from './tokens.js';


// The app's pronouncer
setPronouncer(await loadG2p(g2pModel));

// Read standard input (asynchronously - after the awaits above, a synchronous read of a pipe may find it not ready)
const chunks = [];
for await (const chunk of process.stdin) {
    chunks.push(chunk);
}

// Tokenize each line with the training vocabulary (nn/vocab.json)
const vocabulary = JSON.parse(readFileSync(new URL('../vocab.json', import.meta.url), 'utf8')).tokens;
const lines = Buffer.concat(chunks).toString('utf8').split('\n').filter((line) => line.trim() !== '');
const output = lines.map((line) => JSON.stringify(tokenize(JSON.parse(line), vocabulary)));
process.stdout.write(output.join('\n') + (output.length ? '\n' : ''));
