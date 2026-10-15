# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The network's input tokens - the vocabulary (nn/vocab.json) and a Python port of the text front end,
lib/lexicon.js textToPhrases, so training data is tokenized exactly as the browser tokenizes at inference
(nn/js/tokens.js; `python -m nn.data verify-tokens` checks that the two agree).

A token sequence is:

    <bos>  phones of word 1  _  phones of word 2  ...  <phrase end>  phones ...  <phrase end>

- <bos> - the start (it carries the leading silence's duration)
- phones - ARPAbet with lexical stress on the vowels (AH0, AH1, AH2, ...; consonants have none), as textToPhrases
  gives them - CMU dictionary first pronunciations, function words' vowels reduced to stress 0, and the letter-to-sound
  fallback for unknown words
- _ - a word boundary inside a phrase (it carries any pause between the words - usually none)
- <comma>, <period>, <question>, <wh>, <exclamation> - the phrase ends textToPhrases gives (comma for , ; : - wh for a
  question starting with a question word); each carries the pause after its phrase, and the last one the trailing
  silence

    python -m nn.frontend vocab                 # write nn/vocab.json
    python -m nn.frontend tokens 'Hello, world.'
"""

import json
import os
import re
import sys


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


# Load the pronouncing dictionary (setDictionary) - word to its first pronunciation; comments stripped
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

# The letter-to-sound fallback
SUFFIXES = [('s', ['Z']), ('es', ['IH0', 'Z']), ('ed', ['D']), ('ing', ['IH0', 'NG']), ('er', ['ER0']), ('ly', ['L', 'IY0'])]
LETTERS = {'a': ['AE1'], 'b': ['B'], 'c': ['K'], 'd': ['D'], 'e': ['EH1'], 'f': ['F'], 'g': ['G'], 'h': ['HH'],
           'i': ['IH1'], 'j': ['JH'], 'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['AA1'], 'p': ['P'],
           'q': ['K'], 'r': ['R'], 's': ['S'], 't': ['T'], 'u': ['AH1'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'],
           'y': ['Y'], 'z': ['Z']}

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


# Pronounce a word not in the dictionary (pronounceUnknown)
def pronounce_unknown(word):
    dictionary = get_dictionary()
    for suffix, phones in SUFFIXES:
        if word.endswith(suffix) and word[:-len(suffix)] in dictionary:
            return dictionary[word[:-len(suffix)]] + phones
    phones = []
    for letter in word:
        if letter in LETTERS and (not phones or LETTERS[letter][0] != phones[-1]):
            phones.extend(LETTERS[letter])
    return phones


# Tokenize text into phrases (textToPhrases) - [{words: [{text, phones: [{ph, stress}], isFunction, syllables}], end}]
def text_to_phrases(text):
    dictionary = get_dictionary()
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
        for phone in dictionary[word] if word in dictionary else pronounce_unknown(word):
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
# the phrase-end tokens have no word)
def tokenize(text):
    tokens = [BOS]
    words = []
    for phrase in text_to_phrases(text):
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
