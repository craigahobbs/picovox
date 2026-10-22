# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The network's input tokens - the vocabulary (vocab.json) and a Python port of the text front end,
lib/lexicon.js textToPhrases, so training data is tokenized exactly as the app tokenizes at inference
(lib/tokens.js with the G2P network; `python -m picovox_train.multi verify-tokens` checks that the two agree).

A token sequence is:

    <bos>  phones of word 1  _  phones of word 2  ...  <phrase end>  phones ...  <phrase end>

- <bos> - the start (it carries the leading silence's duration)
- phones - ARPAbet with lexical stress on the vowels (AH0, AH1, AH2, ...; consonants have none), as textToPhrases
  gives them - every word pronounced by the app's G2P network (lib/models/g2pModel.js, run by g2pmodel.py exactly
  as lib/g2p.js runs it - no dictionary), function words' vowels reduced to stress 0
- _ - a word boundary inside a phrase (it carries any pause between the words - usually none)
- <comma>, <period>, <question>, <wh>, <exclamation> - the phrase ends textToPhrases gives (comma for , ; : - wh for a
  question starting with a question word); each carries the pause after its phrase, and the last one the trailing
  silence

    python -m picovox_train.frontend vocab                 # write vocab.json

The CMU Pronouncing Dictionary (data/cmudict.dict, get_dictionary) is the G2P network's training labels - a training
tool, not the front end's.
"""

import json
import multiprocessing
import os
import re
import sys
import unicodedata

from . import ROOT, g2pmodel


DICTIONARY_PATH = os.path.join(ROOT, 'data', 'cmudict.dict')
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')

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


# Write vocab.json
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
# cores; worth it for thousands of new words), and kept in the G2P network's cache (with its ambiguity decisions)
def pronounce_words(words, jobs=None):
    g2p = get_g2p()
    missing = sorted({word for word in words if word not in g2p.cache or word not in g2p.ambiguous_words})
    jobs = min(jobs or os.cpu_count(), max(1, len(missing) // 1000))
    if jobs > 1:
        chunks = [missing[index::jobs] for index in range(jobs)]
        with multiprocessing.Pool(jobs) as pool:
            for chunk, (phones, ambiguous) in zip(chunks, pool.map(_pronounce_chunk, chunks)):
                g2p.cache.update(zip(chunk, phones))
                g2p.ambiguous_words.update(zip(chunk, ambiguous))
    else:
        g2p.pronounce_many(missing)
    return {word: g2p.cache[word] for word in words}


# Pronounce words with the G2P network (a worker task) - (their phones, the ambiguity head's decisions)
def _pronounce_chunk(words):
    g2p = get_g2p()
    return g2p.pronounce_many(words), [g2p.ambiguous_words[word] for word in words]


class Pronunciations(dict):
    """Words' pronunciations - {word: phones} - and the G2P network's caches after pronouncing texts (state: the
    alternate readings, the ambiguity decisions, and the context network's choices), for set_pronunciations"""

    state = None


# Add pronunciations to the G2P network's cache - {word: phones}, or a Pronunciations with its state (a worker
# process's initializer, so its texts' words are pronounced once, before the work is shared out)
def set_pronunciations(pronunciations):
    g2p = get_g2p()
    g2p.cache.update(pronunciations)
    if getattr(pronunciations, 'state', None) is not None:
        g2p.set_state(pronunciations.state)


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


# Pronounce every word of texts (pronounce_words), and, with a context network, choose the readings of their ambiguous
# words and pronounce those - a Pronunciations, for set_pronunciations
def pronounce_texts(texts, jobs=None):
    pronunciations = Pronunciations(pronounce_words(text_words(texts), jobs))
    g2p = get_g2p()
    if g2p.context is not None:
        windows = []
        for text in texts:
            sequence = context_sequence(text_items(text))
            windows.extend(g2p.context.window_of(sequence, index) for index, item in enumerate(sequence) if g2p.ambiguous_words.get(item))
        by_reading = {}
        for window, slot in zip(windows, g2p.context.choose_windows(windows)):
            by_reading.setdefault(slot, set()).add(window[len(window) // 2])
        for reading, words in sorted(by_reading.items()):
            g2p.pronounce_many(sorted(words), reading)
    pronunciations.state = g2p.state()
    return pronunciations


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

# Abbreviations only with their period - the months and days (lib/lexicon.js's periodAbbreviations)
PERIOD_ABBREVIATIONS = {
    'jan': 'january', 'feb': 'february', 'apr': 'april', 'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'sept': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december', 'mon': 'monday', 'tue': 'tuesday',
    'tues': 'tuesday', 'thu': 'thursday', 'thur': 'thursday', 'thurs': 'thursday', 'fri': 'friday'
}

# Number words
ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

# The text's tokens - a word, a number (its thousands grouped by commas or not; with a decimal fraction, or an ordinal or
# plural suffix), or a punctuation mark (lib/lexicon.js's tokenPattern)
TOKEN_REGEX = re.compile(r"[a-z][a-z']*\.?|(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+|(?:st|nd|rd|th|'?s)(?![a-z]))?|[.,;:!?]",
                         re.ASCII)
NUMBER_REGEX = re.compile(r'^([\d,]+)(?:\.(\d+))?(\D*)$', re.ASCII)
YEAR_REGEX = re.compile(r'^[1-9]\d{3}$', re.ASCII)
ACCENTED_REGEX = re.compile('[\u00c0-\u024f]')
MARKS_REGEX = re.compile('[\u0300-\u036f]')
PLAIN_LETTERS = {'æ': 'e', 'Æ': 'E', 'œ': 'e', 'Œ': 'E', 'ø': 'o', 'Ø': 'O', 'ł': 'l', 'Ł': 'L', 'đ': 'd', 'Đ': 'D', 'ß': 's'}
PHONE_REGEX = re.compile(r'^([A-Z]+)(\d)?$')


# Spell an integer in words (numberWords)
def number_words(value):
    if value < 20:
        return [ONES[value]]
    if value < 100:
        return [TENS[value // 10]] if value % 10 == 0 else [TENS[value // 10], ONES[value % 10]]
    size, name = next((size, name) for size, name in ((10 ** 12, 'trillion'), (10 ** 9, 'billion'), (10 ** 6, 'million'),
                                                       (10 ** 3, 'thousand'), (100, 'hundred')) if value >= size)
    rest = value % size
    return number_words(value // size) + [name] + ([] if rest == 0 else number_words(rest))


# Spell a year in words - by its hundreds (1972 as nineteen seventy-two, 1900 as nineteen hundred, 1905 as nineteen oh
# five), but a millennium's first years as a number (2000, 2001 as two thousand one) (yearWords)
def year_words(value):
    hundreds, rest = divmod(value, 100)
    if hundreds % 10 == 0 and rest < 10:
        return number_words(value)
    return number_words(hundreds) + (['hundred'] if rest == 0 else (['oh', ONES[rest]] if rest < 10 else number_words(rest)))


# A number word as an ordinal (twenty as twentieth) or a plural (twenty as twenties) (ordinalWord, pluralWord)
ORDINAL_WORDS = {
    'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth', 'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'
}


def ordinal_word(word):
    return ORDINAL_WORDS.get(word) or (f'{word[:-1]}ieth' if word.endswith('y') else f'{word}th')


def plural_word(word):
    return f'{word[:-1]}ies' if word.endswith('y') else ('sixes' if word == 'six' else f'{word}s')


# A number token's words - its digits (commas grouping its thousands) in words, a four-digit number as a year, a
# decimal's fraction digit by digit, and an ordinal (1st, 22nd) or a plural (1970s, 70's) as its last word's
# (numberTokenWords)
def number_token_words(token):
    digits, fraction, suffix = NUMBER_REGEX.match(token).groups()
    value = int(digits.replace(',', ''))
    is_ordinal = suffix != '' and not suffix.endswith('s')
    words = year_words(value) if YEAR_REGEX.match(digits) and fraction is None and not is_ordinal else number_words(value)
    if fraction is not None:
        words.extend(['point'] + [ONES[int(digit)] for digit in fraction])
    elif suffix != '':
        words.append((ordinal_word if is_ordinal else plural_word)(words.pop()))
    return words


# The context marks of the phrase ends - a phrase end's item in the context sequence (the context network's input)
PHRASE_MARKS = {'comma': ',', 'period': '.', 'question': '?', 'exclamation': '!'}


# A text's accented Latin letters without their accents - á as a, Ñ as N, and the ligatures as English says them
# (Cæsar, Œdipus) - each letter one letter, so the text keeps its length (plainLetters)
def plain_letters(text):
    def plain(match):
        char = match.group(0)
        return PLAIN_LETTERS.get(char) or MARKS_REGEX.sub('', unicodedata.normalize('NFD', char))

    return ACCENTED_REGEX.sub(plain, text)


# The front end's items of a text, in order (textToPhrases' first pass) - [(item, offset)]: a word (its text after
# numbers and abbreviations are spelled out) or a phrase end ({'end': name}, only after a word), and the offset of its
# source token in the text. The context sequence is the items with the phrase ends as PHRASE_MARKS.
def text_items(text):
    items = []

    def end_phrase(end, offset):
        if items and isinstance(items[-1][0], str):
            items.append(({'end': end}, offset))

    for match in TOKEN_REGEX.finditer(plain_letters(text).lower().replace('‘', '\'').replace('’', '\'')):
        token, offset = match.group(0), match.start()
        if '0' <= token[0] <= '9':
            items.extend((word, offset) for word in number_token_words(token))
        elif 'a' <= token[0] <= 'z':
            bare = token[:-1] if token.endswith('.') else token
            if bare in ABBREVIATIONS:
                items.append((ABBREVIATIONS[bare], offset))
            elif token.endswith('.') and bare in PERIOD_ABBREVIATIONS:
                items.append((PERIOD_ABBREVIATIONS[bare], offset))
            else:
                items.append((bare.strip('\''), offset))
                if token.endswith('.'):
                    end_phrase('period', offset)
        elif token == '?':
            end_phrase('question', offset)
        elif token == '!':
            end_phrase('exclamation', offset)
        elif token == '.':
            end_phrase('period', offset)
        else:
            end_phrase('comma', offset)
    end_phrase('period', len(text))
    return items


# The context sequence of a text's items - the words, and the phrase ends as their marks
def context_sequence(items):
    return [item if isinstance(item, str) else PHRASE_MARKS[item['end']] for item, _ in items]


# Tokenize text into phrases (textToPhrases) - [{words: [{text, phones: [{ph, stress}], isFunction, syllables}], end}];
# pronounce_word(word) gives a word's ARPAbet phones (default: the G2P network's - lib/lexicon.js setPronouncer with
# lib/g2p.js loadG2p, as the app does: a word the network finds ambiguous takes the reading its context network chooses
# from the neighboring words); readings ({position: phones}, a word's index in the text's words) replaces words' phones
# (the voice training data's aligned readings - readings.py)
def text_to_phrases(text, pronounce_word=None, readings=None):
    items = text_items(text)
    if pronounce_word is None:
        g2p = get_g2p()
        sequence = context_sequence(items)

        # A word's phones - the reading the context network chooses for an ambiguous word, else the default reading
        def pronounce_item(word, index):
            return g2p.pronounce(word, g2p.choose_reading(sequence, index) if g2p.ambiguous(word) else 0)
    else:
        def pronounce_item(word, _index):
            return pronounce_word(word)
    phrases = []
    words = []
    position = 0
    for index, (item, _) in enumerate(items):
        if not isinstance(item, str):
            end = item['end']
            phrases.append({'words': words, 'end': 'wh' if end == 'question' and words[0]['text'] in WH_WORDS else end})
            words = []
            continue
        phones = []
        word_phones = readings[position] if readings and position in readings else pronounce_item(item, index)
        for phone in word_phones:
            match = PHONE_REGEX.match(phone)
            phones.append({'ph': match.group(1), 'stress': None if match.group(2) is None else int(match.group(2))})
        is_function = item in FUNCTION_WORDS
        if is_function:
            for phone in phones:
                if phone['stress'] is not None:
                    phone['stress'] = 0
        words.append({'text': item, 'phones': phones, 'isFunction': is_function,
                      'syllables': sum(1 for phone in phones if phone['ph'] in VOWELS)})
        position += 1
    return phrases


# The token name of a phone
def phone_token(phone):
    return phone['ph'] if phone['stress'] is None else f'{phone["ph"]}{phone["stress"]}'


# Tokenize text - {'tokens': [name], 'ids': [id], 'words': [{text, start, end}]} (each word's token index range, and
# the phrase-end tokens have no word); pronounce_word and readings as text_to_phrases
def tokenize(text, pronounce_word=None, readings=None):
    tokens = [BOS]
    words = []
    for phrase in text_to_phrases(text, pronounce_word, readings):
        for index, word in enumerate(phrase['words']):
            if index > 0:
                tokens.append(WORD)
            start = len(tokens)
            tokens.extend(phone_token(phone) for phone in word['phones'])
            words.append({'text': word['text'], 'start': start, 'end': len(tokens)})
        tokens.append(PHRASE_ENDS[phrase['end']])
    return {'tokens': tokens, 'ids': [TOKEN_IDS.get(token, TOKEN_IDS[UNK]) for token in tokens], 'words': words}


# The command line - vocab (write vocab.json)
def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args != ['vocab']:
        print('usage: python -m picovox_train.frontend vocab', file=sys.stderr)
        return 2
    write_vocab()
    print(f'Wrote {VOCAB_PATH} ({len(vocabulary())} tokens)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
