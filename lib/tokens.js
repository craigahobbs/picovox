// Licensed under the MIT License
// https://github.com/craigahobbs/picovox/blob/main/LICENSE

// The neural network's input tokens from text - lib/lexicon.js textToPhrases as the token ids of a vocabulary, the
// token names a voice model was trained with (its manifest's vocab - nn/vocab.json at training time):
//
//     <bos>  phones of word 1  _  phones of word 2  ...  <phrase end>  phones ...  <phrase end>
//
// Phones are ARPAbet with the vowels' lexical stress (AH0, AH1, ...), "_" is a word boundary inside a phrase, and the
// phrase ends are <comma>, <period>, <question>, <wh>, and <exclamation>. The Python side, nn/frontend.py, makes the
// training data's tokens - `python -m nn.data verify-tokens` checks that both give the same ids.
//
// The words' pronunciations come from lib/lexicon.js's pronouncer - set the G2P network first (setPronouncer with
// lib/g2p.js loadG2p), as the package (lib/synthesis.js) and nn/js/tokenizeLines.js do: the training data is pronounced
// by it.

import {textToPhrases} from './lexicon.js';


// A vocabulary's token name to id map (cached per vocabulary)
const vocabularyIds = new WeakMap();
function tokenIds(vocabulary) {
    let ids = vocabularyIds.get(vocabulary);
    if (ids === undefined) {
        ids = new Map(vocabulary.map((token, index) => [token, index]));
        vocabularyIds.set(vocabulary, ids);
    }
    return ids;
}


// The token name of a phone ({ph, stress})
function phoneToken(phone) {
    return phone.stress === null ? phone.ph : `${phone.ph}${phone.stress}`;
}


// Tokenize text with a vocabulary (token names - a token's id is its index) - {tokens: [name], ids: [id], words:
// [{text, start, end}]}, each word's token index range [start, end)
export function tokenize(text, vocabulary) {
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
        tokens.push(`<${phrase.end}>`);
    }
    const ids = tokenIds(vocabulary);
    const unknown = ids.get('<unk>');
    return {tokens, 'ids': tokens.map((token) => ids.get(token) ?? unknown), words};
}
