# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The network's input tokens - the vocabulary (vocab.json) and a Python port of the text front end,
lib/lexicon.js textToPhrases, so training data is tokenized exactly as the app tokenizes at inference
(lib/tokens.js with the G2P network; the parity check - tests/test_frontend.py - checks that the two agree).

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
"""

import json
import multiprocessing
import os
import re
import sys
import unicodedata

from . import g2pmodel


VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')

# The ARPAbet phones - the vowels carry stress 0, 1, or 2
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
                       'picovox_train/frontend.py from lib/lexicon.js textToPhrases: <bos>, then each phrase\'s words\' phones '
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

# Abbreviations only with their period - the months and days, and the titles and the rest that don't end a sentence;
# "No." before a number is "number" (lib/lexicon.js's periodAbbreviations)
PERIOD_ABBREVIATIONS = {
    'jan': 'january', 'feb': 'february', 'apr': 'april', 'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'sept': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december', 'mon': 'monday', 'tue': 'tuesday',
    'tues': 'tuesday', 'thu': 'thursday', 'thur': 'thursday', 'thurs': 'thursday', 'fri': 'friday',
    'prof': 'professor', 'capt': 'captain', 'col': 'colonel', 'lt': 'lieutenant', 'rev': 'reverend',
    'esq': 'esquire', 'jr': 'junior', 'sr': 'senior', 'mt': 'mount', 'vol': 'volume', 'fig': 'figure'
}

# Number words
ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

# The text's tokens - a.m. or p.m., a word, a time (10:30, 10:30 a.m., 10pm), an amount of money (with its scale - $5
# million), a number (its thousands grouped by commas or not; with a decimal fraction, a percent sign, or an ordinal or
# plural suffix), or a punctuation mark (a parenthesis too - lib/lexicon.js's tokenPattern)
DIGITS_PATTERN = r'(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)'
MERIDIEM_PATTERN = r'[ap]\.?m\.?(?![a-z])'
TOKEN_REGEX = re.compile('|'.join([
    r'[ap]\.m\.?(?![a-z])',
    r"[a-z][a-z']*\.?",
    rf'\d{{1,2}}(?::\d{{2}}(?!\d)(?:[ \t]*{MERIDIEM_PATTERN})?|[ \t]*{MERIDIEM_PATTERN})',
    rf'[$£€]{DIGITS_PATTERN}(?:\.\d+)?(?:[ \t\r\n]+(?:thousand|million|billion|trillion)(?![a-z]))?',
    rf"{DIGITS_PATTERN}(?:\.\d+%?|%|(?:st|nd|rd|th|'?s)(?![a-z]))?",
    '[.,;:!?()]'
]), re.ASCII)
NUMBER_REGEX = re.compile(r'^([\d,]+)(?:\.(\d+))?(\D*)$', re.ASCII)
MONEY_REGEX = re.compile(r'^(\D)([\d,]+)(?:\.(\d+))?(?:\s+([a-z]+))?$', re.ASCII)
YEAR_REGEX = re.compile(r'^[1-9]\d{3}$', re.ASCII)
TIME_REGEX = re.compile(r'^\d+(?::|[ \t]*[ap])', re.ASCII)
TIME_WORDS_REGEX = re.compile(r'^(\d+)(?::(\d+))?[ \t]*(?:([ap])\.?m\.?)?$', re.ASCII)
MERIDIEM_REGEX = re.compile(r'^[ap]\.m')
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


# Digits in words, digit by digit (digitWords)
def digit_words(digits):
    return [ONES[int(digit)] for digit in digits]


# An integer's digits (commas grouping its thousands or not) in words - digit by digit past 15 digits (a number that
# long isn't read as one, and is past a JavaScript double's exact integers) (integerWords)
def integer_words(digits):
    plain = digits.replace(',', '').lstrip('0') or '0'
    return digit_words(plain) if len(plain) > 15 else number_words(int(plain))


# A decimal's fraction in words - digit by digit after the point (fractionWords)
def fraction_words(fraction):
    return ['point'] + digit_words(fraction)


# A number token's words - its digits (commas grouping its thousands) in words (integer_words), a four-digit number as
# a year, a decimal's fraction digit by digit, a percentage, and an ordinal (1st, 22nd) or a plural (1970s, 70's) as its
# last word's (numberTokenWords)
def number_token_words(token):
    digits, fraction, suffix = NUMBER_REGEX.match(token).groups()
    is_year = YEAR_REGEX.match(digits) and fraction is None and (suffix == '' or suffix.endswith('s'))
    words = year_words(int(digits)) if is_year else integer_words(digits)
    if fraction is not None:
        words.extend(fraction_words(fraction))
    if suffix == '%':
        words.append('percent')
    elif suffix.endswith('s'):
        words.append(plural_word(words.pop()))
    elif suffix != '':
        words.append(ordinal_word(words.pop()))
    return words


# The currencies' words - the unit, its plural, the hundredth, its plural (currencies)
CURRENCIES = {
    '$': ('dollar', 'dollars', 'cent', 'cents'), '£': ('pound', 'pounds', 'penny', 'pence'), '€': ('euro', 'euros', 'cent', 'cents')
}


# An amount of money's words - $1,234.56 as one thousand two hundred thirty-four dollars and fifty-six cents, $0.50 as
# fifty cents, $1.5 as one point five dollars, and $5 million as five million dollars (moneyWords)
def money_words(token):
    symbol, digits, fraction, scale = MONEY_REGEX.match(token).groups()
    unit, units, hundredth, hundredths = CURRENCIES[symbol]
    value = int(digits.replace(',', ''))
    if fraction is not None and len(fraction) == 2 and scale is None:
        cents = int(fraction)
        words = [] if value == 0 and cents != 0 else integer_words(digits) + [unit if value == 1 else units]
        if cents != 0:
            words.extend(([] if not words else ['and']) + number_words(cents) + [hundredth if cents == 1 else hundredths])
        return words
    words = integer_words(digits)
    if fraction is not None:
        words.extend(fraction_words(fraction))
    if scale is not None:
        words.append(scale)
    words.append(unit if value == 1 and fraction is None and scale is None else units)
    return words


# The letters of a.m. and p.m. as words (meridiemWords)
MERIDIEM_WORDS = {'a': ['ay', 'em'], 'p': ['pee', 'em']}


# A time's words - 10:30 as ten thirty, 10:05 as ten oh five, 10:00 as ten o'clock (14:00 as fourteen hundred), and
# with a.m. or p.m. (10:00 p.m. and 10pm as ten pee em) (timeWords)
def time_words(token):
    hours, minutes, meridiem = TIME_WORDS_REGEX.match(token).groups()
    words = number_words(int(hours))
    if minutes is not None and int(minutes) != 0:
        words.extend(['oh', ONES[int(minutes)]] if int(minutes) < 10 else number_words(int(minutes)))
    elif meridiem is None:
        words.append("o'clock" if 1 <= int(hours) <= 12 else 'hundred')
    return words if meridiem is None else words + MERIDIEM_WORDS[meridiem]


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
# numbers and abbreviations are spelled out) or a phrase end ({'end': name}, only after a word - a parenthesis's comma
# gives way to a punctuation mark right after it), and the offset of its source token in the text. The context sequence
# is the items with the phrase ends as PHRASE_MARKS.
def text_items(text):
    items = []
    parenthesis_end = False

    def end_phrase(end, offset, parenthesis=False):
        nonlocal parenthesis_end
        if items and isinstance(items[-1][0], str):
            items.append(({'end': end}, offset))
            parenthesis_end = parenthesis
        elif parenthesis_end and not parenthesis:
            items[-1] = ({'end': end}, offset)
            parenthesis_end = False

    matches = list(TOKEN_REGEX.finditer(plain_letters(text).lower().replace('‘', '\'').replace('’', '\'')))
    for index, match in enumerate(matches):
        token, offset = match.group(0), match.start()
        if TIME_REGEX.match(token):
            items.extend((word, offset) for word in time_words(token))
        elif '0' <= token[0] <= '9':
            items.extend((word, offset) for word in number_token_words(token))
        elif token[0] in CURRENCIES:
            items.extend((word, offset) for word in money_words(token))
        elif MERIDIEM_REGEX.match(token):
            items.extend((word, offset) for word in MERIDIEM_WORDS[token[0]])
        elif 'a' <= token[0] <= 'z':
            bare = token[:-1] if token.endswith('.') else token
            if bare in ABBREVIATIONS:
                items.append((ABBREVIATIONS[bare], offset))
            elif token.endswith('.') and bare in PERIOD_ABBREVIATIONS:
                items.append((PERIOD_ABBREVIATIONS[bare], offset))
            elif token == 'no.' and index + 1 < len(matches) and '0' <= matches[index + 1].group(0)[0] <= '9':
                items.append(('number', offset))
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
        elif token in '()':
            end_phrase('comma', offset, True)
        else:
            end_phrase('comma', offset)
    end_phrase('period', len(text))
    return items


# The context sequence of a text's items - the words, and the phrase ends as their marks
def context_sequence(items):
    return [item if isinstance(item, str) else PHRASE_MARKS[item['end']] for item, _ in items]


# Tokenize text into phrases (textToPhrases) - [{words: [{text, phones: [{ph, stress}]}], end}];
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
        if item in FUNCTION_WORDS:
            for phone in phones:
                if phone['stress'] is not None:
                    phone['stress'] = 0
        words.append({'text': item, 'phones': phones})
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
