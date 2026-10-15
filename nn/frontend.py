# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The network's input tokens - the vocabulary (nn/vocab.json) and a Python port of the text front end,
lib/lexicon.js textToPhrases, so training data is tokenized exactly as the app tokenizes at inference
(nn/js/tokens.js with the G2P network; `python -m nn.data verify-tokens` checks that the two agree).

A token sequence is:

    <bos>  phones of word 1  _  phones of word 2  ...  <phrase end>  phones ...  <phrase end>

- <bos> - the start (it carries the leading silence's duration)
- phones - ARPAbet with lexical stress on the vowels (AH0, AH1, AH2, ...; consonants have none), as textToPhrases
  gives them - every word pronounced by the app's G2P network (lib/models/g2pModel.js, run by nn/g2pmodel.py exactly
  as lib/g2p.js runs it - no dictionary), function words' vowels reduced to stress 0
- _ - a word boundary inside a phrase (it carries any pause between the words - usually none)
- <comma>, <period>, <question>, <wh>, <exclamation> - the phrase ends textToPhrases gives (comma for , ; : - wh for a
  question starting with a question word); each carries the pause after its phrase, and the last one the trailing
  silence

    python -m nn.frontend vocab                 # write nn/vocab.json
    python -m nn.frontend tokens 'Hello, world.'

The CMU Pronouncing Dictionary (data/cmudict.dict, get_dictionary) is the G2P network's training labels - a training
tool, not the front end's.
"""

import json
import multiprocessing
import os
import re
import sys

from . import g2pmodel


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICTIONARY_PATH = os.path.join(ROOT, 'data', 'cmudict.dict')
VOCAB_PATH = os.path.join(ROOT, 'nn', 'vocab.json')

# The ARPAbet phones (lib/phonemes.js) - the vowels carry stress 0, 1, or 2
VOWELS = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
CONSONANTS = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH',
              'V', 'W', 'Y', 'Z', 'ZH']

# The special tokens - padding, unknown, start, word boundary, and the phrase ends (textToPhrases' phrase 'end' values)
PAD, UNK, BOS, WORD = '<pad>', '<unk>', '<bos>', '_'
PHRASE_ENDS = {'comma': '<comma>', 'period': '<period>', 'question': '<question>', 'wh': '<wh>',
               'exclamation': '<exclamation>'}
SPECIALS = [PAD, UNK, BOS, WORD] + list(PHRASE_ENDS.values())
BOUNDARIES = set([BOS, WORD] + list(PHRASE_ENDS.values()))


# The vocabulary - the token names, their id the index
def vocabulary():
    return SPECIALS + CONSONANTS + [f'{vowel}{stress}' for vowel in VOWELS for stress in range(3)]


# Write nn/vocab.json
def write_vocab(path=VOCAB_PATH):
    tokens = vocabulary()
    vocab = {
        'version': 1,
        'description': 'The neural-network input tokens - the id of a token is its index in "tokens". Made by '
                       'nn/frontend.py from lib/lexicon.js textToPhrases: <bos>, then each phrase\'s words\' phones '
                       '(ARPAbet, vowels with stress) separated by "_", then its phrase-end token.',
        'tokens': tokens,
        'pad': tokens.index(PAD),
        'unk': tokens.index(UNK),
        'bos': tokens.index(BOS),
        'word': tokens.index(WORD),
        'phraseEnds': {end: tokens.index(token) for end, token in PHRASE_ENDS.items()},
        'vowels': VOWELS,
        'consonants': CONSONANTS
    }
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(vocab, indent=4) + '\n')


# The token name to id map
TOKEN_IDS = {token: index for index, token in enumerate(vocabulary())}


#
# The port of lib/lexicon.js
#

_DICTIONARY = None
_G2P = None


# The G2P network (lib/models/g2pModel.js) - loaded on first use
def get_g2p():
    global _G2P  # pylint: disable=global-statement
    if _G2P is None:
        _G2P = g2pmodel.G2p(g2pmodel.read_module())
    return _G2P


# The pronunciations of words - {word: phones}, pronounced in batches, by up to jobs worker processes (default: all
# cores; worth it for thousands of new words), and kept in the G2P network's cache
def pronounce_words(words, jobs=None):
    g2p = get_g2p()
    missing = sorted({word for word in words if word not in g2p.cache})
    jobs = min(jobs or os.cpu_count(), max(1, len(missing) // 1000))
    if jobs > 1:
        chunks = [missing[index::jobs] for index in range(jobs)]
        with multiprocessing.Pool(jobs) as pool:
            for chunk, phones in zip(chunks, pool.map(_pronounce_chunk, chunks)):
                g2p.cache.update(zip(chunk, phones))
    else:
        g2p.pronounce_many(missing)
    return {word: g2p.cache[word] for word in words}


# Pronounce words with the G2P network (a worker task)
def _pronounce_chunk(words):
    return get_g2p().pronounce_many(words)


# Add pronunciations to the G2P network's cache - {word: phones} (a worker process's initializer, so its texts'
# words are pronounced once, before the work is shared out)
def set_pronunciations(pronunciations):
    get_g2p().cache.update(pronunciations)


# The words of texts as the front end pronounces them (after numbers and abbreviations are spelled out) - distinct, in
# first-seen order
def text_words(texts):
    words = {}

    # Record a word (no phones)
    def record(word):
        words[word] = None
        return []

    for text in texts:
        text_to_phrases(text, record)
    return list(words)


# Pronounce every word of texts (pronounce_words) - {word: phones}, for set_pronunciations
def pronounce_texts(texts, jobs=None):
    return pronounce_words(text_words(texts), jobs)


# Load the CMU Pronouncing Dictionary - word to its first pronunciation; comments stripped (the G2P network's training
# labels)
def get_dictionary(path=DICTIONARY_PATH):
    global _DICTIONARY  # pylint: disable=global-statement
    if _DICTIONARY is None:
        dictionary = {}
        with open(path, encoding='utf-8') as file:
            for line in file.read().split('\n'):
                hash_index = line.find('#')
                content = (line[:hash_index] if hash_index >= 0 else line).strip()
                if content != '':
                    word, *phones = content.split()
                    if '(' not in word:
                        dictionary[word] = phones
        _DICTIONARY = dictionary
    return _DICTIONARY


# Function words - unaccented, with reduced stress
FUNCTION_WORDS = set([
    'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'nor', 'is', 'are', 'was', 'were', 'be',
    'been', 'am', 'it', 'its', 'that', 'this', 'with', 'as', 'by', 'from', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'do', 'does', 'did', 'have', 'has', 'had', 'will',
    'would', 'can', 'could', 'shall', 'should', 'may', 'might', 'must', 'so', 'if', 'than', 'then', 'into', 'onto',
    'up', 'out', 'over', 'about', 'there', 'some', 'just'
])

# Question words - a question starting with one ends with a fall
WH_WORDS = set(['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'])

# Abbreviations
ABBREVIATIONS = {'mr': 'mister', 'mrs': 'missus', 'dr': 'doctor', 'st': 'saint', 'vs': 'versus', 'etc': 'etcetera'}

# Number words
ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

TOKEN_REGEX = re.compile(r"[a-z][a-z']*\.?|\d+(?:\.\d+)?|[.,;:!?]", re.ASCII)
PHONE_REGEX = re.compile(r'^([A-Z]+)(\d)?$')


# Spell an integer in words (numberWords)
def number_words(value):
    if value < 20:
        return [ONES[value]]
    if value < 100:
        return [TENS[value // 10]] if value % 10 == 0 else [TENS[value // 10], ONES[value % 10]]
    for size, name in ((10 ** 9, 'billion'), (10 ** 6, 'million'), (10 ** 3, 'thousand'), (100, 'hundred')):
        if value >= size:
            rest = value % size
            return number_words(value // size) + [name] + ([] if rest == 0 else number_words(rest))
    return []


# Tokenize text into phrases (textToPhrases) - [{words: [{text, phones: [{ph, stress}], isFunction, syllables}], end}];
# pronounce_word(word) gives a word's ARPAbet phones (default: the G2P network's - lib/lexicon.js setPronouncer with
# lib/g2p.js loadG2p, as the app does)
def text_to_phrases(text, pronounce_word=None):
    if pronounce_word is None:
        pronounce_word = get_g2p().pronounce
    tokens = TOKEN_REGEX.findall(text.lower().replace('‘', '\'').replace('’', '\''))
    phrases = []
    words = []

    def end_phrase(end):
        nonlocal words
        if words:
            phrases.append({'words': words, 'end': 'wh' if end == 'question' and words[0]['text'] in WH_WORDS else end})
            words = []

    def add_word(word_text):
        word = word_text.strip('\'')
        phones = []
        for phone in pronounce_word(word):
            match = PHONE_REGEX.match(phone)
            phones.append({'ph': match.group(1), 'stress': None if match.group(2) is None else int(match.group(2))})
        is_function = word in FUNCTION_WORDS
        if is_function:
            for phone in phones:
                if phone['stress'] is not None:
                    phone['stress'] = 0
        words.append({'text': word, 'phones': phones, 'isFunction': is_function,
                      'syllables': sum(1 for phone in phones if phone['ph'] in VOWELS)})

    for token in tokens:
        if '0' <= token[0] <= '9':
            whole, _, fraction = token.partition('.')
            for word in number_words(int(whole)):
                add_word(word)
            if '.' in token:
                add_word('point')
                for digit in fraction:
                    add_word(ONES[int(digit)])
        elif 'a' <= token[0] <= 'z':
            bare = token[:-1] if token.endswith('.') else token
            if bare in ABBREVIATIONS:
                add_word(ABBREVIATIONS[bare])
            else:
                add_word(bare)
                if token.endswith('.'):
                    end_phrase('period')
        elif token == '?':
            end_phrase('question')
        elif token == '!':
            end_phrase('exclamation')
        elif token == '.':
            end_phrase('period')
        else:
            end_phrase('comma')
    end_phrase('period')
    return phrases


# The token name of a phone
def phone_token(phone):
    return phone['ph'] if phone['stress'] is None else f'{phone["ph"]}{phone["stress"]}'


# Tokenize text - {'tokens': [name], 'ids': [id], 'words': [{text, start, end}]} (each word's token index range, and
# the phrase-end tokens have no word); pronounce_word as text_to_phrases
def tokenize(text, pronounce_word=None):
    tokens = [BOS]
    words = []
    for phrase in text_to_phrases(text, pronounce_word):
        for index, word in enumerate(phrase['words']):
            if index > 0:
                tokens.append(WORD)
            start = len(tokens)
            tokens.extend(phone_token(phone) for phone in word['phones'])
            words.append({'text': word['text'], 'start': start, 'end': len(tokens)})
        tokens.append(PHRASE_ENDS[phrase['end']])
    return {'tokens': tokens, 'ids': [TOKEN_IDS.get(token, TOKEN_IDS[UNK]) for token in tokens], 'words': words}


# The command line - vocab (write nn/vocab.json) or tokens text...
def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == 'vocab':
        write_vocab()
        print(f'Wrote {VOCAB_PATH} ({len(vocabulary())} tokens)')
    elif args and args[0] == 'tokens':
        for text in args[1:]:
            result = tokenize(text)
            print(' '.join(result['tokens']))
            print(' '.join(str(token_id) for token_id in result['ids']))
    else:
        print('usage: python -m nn.frontend vocab | tokens text...', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
