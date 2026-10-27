// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// Pronounce words with a G2P module as the package does (lib/g2p.js) - one word per input line in, one line of its phones
// (space-separated) out (used by `python -m picovox_train.context check` to measure a module on the book words)
//
//     node tools/pronounceLines.js lib/models/g2pModel.js < words.txt > phones.txt

import {Buffer} from 'node:buffer';
import {loadG2p} from '../lib/g2p.js';
import {pathToFileURL} from 'node:url';
import process from 'node:process';
import {resolve} from 'node:path';


// Load the module
const {g2pModel} = await import(pathToFileURL(resolve(process.argv[2])).href);
const pronounce = await loadG2p(g2pModel);

// Read standard input (asynchronously, after the awaits above)
const chunks = [];
for await (const chunk of process.stdin) {
    chunks.push(chunk);
}
const words = Buffer.concat(chunks).toString('utf8').split('\n').filter((line) => line !== '');
const output = words.map((word) => pronounce(word).join(' '));
process.stdout.write(output.join('\n') + (output.length ? '\n' : ''));
