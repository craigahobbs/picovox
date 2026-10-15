// Licensed under the MIT License
// https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

// The neural network's input tokens from text - lib/lexicon.js textToPhrases as the token ids of nn/vocab.json:
//
//     <bos>  phones of word 1  _  phones of word 2  ...  <phrase end>  phones ...  <phrase end>
//
// Phones are ARPAbet with the vowels' lexical stress (AH0, AH1, ...), "_" is a word boundary inside a phrase, and the
// phrase ends are <comma>, <period>, <question>, <wh>, and <exclamation>. The Python side, nn/frontend.py, makes the
// training data's tokens - `python -m nn.data verify-tokens` checks that both give the same ids.
//
// Under Node, lib/lexicon.js loads data/cmudict.dict itself; in the browser, call setDictionary(text) first.

import {textToPhrases} from '../../lib/lexicon.js';
import vocab from '../vocab.json' with {'type': 'json'};


// The vocabulary - its token names, a token name to id map, and the special token ids
export {vocab};
const tokenIds = new Map(vocab.tokens.map((token, index) => [token, index]));


// The token name of a phone ({ph, stress})
function phoneToken(phone) {
    return phone.stress === null ? phone.ph : `${phone.ph}${phone.stress}`;
}


// Tokenize text - {tokens: [name], ids: [id], words: [{text, start, end}]}, each word's token index range [start, end)
export function tokenize(text) {
    const tokens = ['<bos>'];
    const words = [];
    for (const phrase of textToPhrases(text)) {
        for (const [index, word] of phrase.words.entries()) {
            if (index > 0) {
                tokens.push('_');
            }
            const start = tokens.length;
            tokens.push(...word.phones.map(phoneToken));
            words.push({'text': word.text, start, 'end': tokens.length});
        }
        tokens.push(vocab.tokens[vocab.phraseEnds[phrase.end]]);
    }
    const ids = tokens.map((token) => tokenIds.get(token) ?? vocab.unk);
    return {tokens, ids, words};
}


// The token names of ids
export function tokenNames(ids) {
    return ids.map((id) => vocab.tokens[id]);
}
