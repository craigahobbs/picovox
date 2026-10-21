// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// A text's quotations as the package finds them (lib/sentences.js splitSentences, lib/style.js textQuotations) - one
// JSON text per input line in, one JSON line of its quotations out ([{quotation, first, last (its sentences' indices),
// text, before, after (the narration beside it in its first and last sentences), previous, next (the sentences around
// it)}]) - for the style labels and the style network's training (picovox_train/style.py)
//
//     node nn/js/quoteLines.js < texts.jsonl > quotations.jsonl

import {Buffer} from 'node:buffer';
import process from 'node:process';
import {splitSentences} from '../../lib/sentences.js';
import {textQuotations} from '../../lib/style.js';


// Read standard input, then write each text's quotations
const chunks = [];
for await (const chunk of process.stdin) {
    chunks.push(chunk);
}
const lines = Buffer.concat(chunks).toString('utf8').split('\n').filter((line) => line !== '');
for (const line of lines) {
    process.stdout.write(`${JSON.stringify(textQuotations(splitSentences(JSON.parse(line))))}\n`);
}
