# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

from picovox_train import frontend

from . import data_fixtures, parity
from .helpers import captured


# Texts for the front end - numbers, decimals, abbreviations, quotes, every phrase end, heteronyms
TEXTS = [
    'Call me Ishmael.',
    'Mr. Smith paid $1,234.56 on 3.14 days in 1999; it was 2,000,000 or 1,000,000,000!',
    'Who read the book? He wound the clock, and lead the way.',
    '‘Tis the season’s end: what do you think?',
    ', Leading comma. 21 22 30 100 101 1000',
    'Dr. Who vs. St. Ives etc.',
    'Señor Amadís and Doña Œdipus saw Cæsar in Łódź - Straße, ñ, ŋ.',
    'In 1972, 1905, 1900, and 2001 - the 1970s and 80\'s - the 1st, 22nd, 3rd, and 12th of 23,000,000 (12,3456).',
    'At 10:30 a.m., 9:00 p.m., or 6pm, I am paying 50% - $1,234.56, $0.50, $5 million, £1.50, and €20 - in the a.m.',
    'He left the room (quietly, as always) and closed the door (1818); (Aside.) The end (truly)',
    'Pi is 3.14159265358979323846; 12345678901234567890 and 9,007,199,254,740,993 are long, $12345678901234567.89 too.',
    'Rev. Smith, Esq., Capt. Hook, and Ms. Ames saw No. 5, Vol. 2, and Fig. 3 on Mt. Ida; I said no. No. Gen. Lee, Ch. 4.'
]


# A fake G2P network (g2pmodel.G2p's interface) - each word's phones are its length's
class FakeG2p:

    def __init__(self, context=None):
        self.cache = {}
        self.ambiguous_words = {}
        self.context = context
        self.states = []

    def pronounce_many(self, words, reading=0):
        for word in words:
            self.cache[word] = ['K'] * len(word) + ([f'AE{reading}'])
            self.ambiguous_words[word] = word == 'read'
        return [self.cache[word] for word in words]

    def state(self):
        return {'readings': [self.cache]}

    def set_state(self, state):
        self.states.append(state)


class TestFrontend(unittest.TestCase):

    def test_vocabulary(self):
        tokens = frontend.vocabulary()
        self.assertEqual(tokens[:4], ['<pad>', '<unk>', '<bos>', '_'])
        self.assertEqual(len(tokens), 9 + 24 + 45)
        self.assertEqual(frontend.TOKEN_IDS['AH0'], tokens.index('AH0'))

    def test_write_vocab(self):
        # The committed vocab.json is the vocabulary's
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'vocab.json')
            frontend.write_vocab(path)
            with open(path, encoding='utf-8') as file, open(frontend.VOCAB_PATH, encoding='utf-8') as committed:
                self.assertEqual(file.read(), committed.read())

    def test_main(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'vocab.json')
            with unittest.mock.patch.object(frontend, 'VOCAB_PATH', path), captured() as (stdout, _):
                self.assertEqual(frontend.main(['vocab']), 0)
            self.assertEqual(stdout.getvalue(), f'Wrote {path} (78 tokens)\n')
            with open(path, encoding='utf-8') as file:
                self.assertEqual(json.load(file)['tokens'], frontend.vocabulary())
        with captured() as (_, stderr):
            self.assertEqual(frontend.main(['tokens']), 2)
        self.assertEqual(stderr.getvalue(), 'usage: python -m picovox_train.frontend vocab\n')

    def test_main_module(self):
        with unittest.mock.patch.object(sys, 'argv', ['frontend']), captured() as (_, stderr), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.frontend', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('usage', stderr.getvalue())

    def test_get_g2p(self):
        g2p = frontend.get_g2p()
        self.assertIs(frontend.get_g2p(), g2p)
        self.assertIsNotNone(g2p.context)

    def test_pronounce_words(self):
        fake = FakeG2p()
        with unittest.mock.patch.object(frontend, '_G2P', fake):
            self.assertEqual(frontend.pronounce_words(['ab', 'read']), {'ab': ['K', 'K', 'AE0'], 'read': ['K'] * 4 + ['AE0']})

            # Thousands of new words, in worker processes
            words = [f'w{index:05d}' for index in range(2500)]
            with unittest.mock.patch.object(frontend.multiprocessing, 'Pool', data_fixtures.FakePool) as pool:
                result = frontend.pronounce_words(words + ['ab'], jobs=4)
            self.assertIs(pool, data_fixtures.FakePool)
            self.assertEqual(len(result), 2501)
            self.assertEqual(result['w00001'], ['K'] * 6 + ['AE0'])
            self.assertFalse(fake.ambiguous_words['w00001'])

    def test_set_pronunciations(self):
        fake = FakeG2p()
        with unittest.mock.patch.object(frontend, '_G2P', fake):
            frontend.set_pronunciations({'x': ['K']})
            self.assertEqual(fake.cache, {'x': ['K']})
            self.assertEqual(fake.states, [])
            pronunciations = frontend.Pronunciations({'y': ['T']})
            frontend.set_pronunciations(pronunciations)
            self.assertEqual(fake.states, [])
            pronunciations.state = {'readings': []}
            frontend.set_pronunciations(pronunciations)
            self.assertEqual(fake.states, [{'readings': []}])

    def test_text_words(self):
        self.assertEqual(frontend.text_words(['Mr. Smith, 21.', 'Smith 3.5']),
                         ['mister', 'smith', 'twenty', 'one', 'three', 'point', 'five'])

    def test_text_words_period_abbreviations(self):
        # The months and days only with their period (as lib/lexicon.js)
        self.assertEqual(frontend.text_words(['On Mon. Nov. 5, mon ami, Aug. sat.']),
                         ['on', 'monday', 'november', 'five', 'mon', 'ami', 'august', 'sat'])
        self.assertEqual(frontend.PERIOD_ABBREVIATIONS['sept'], 'september')
        self.assertEqual(frontend.text_words(['Rev. Ames saw No. 5; I said no. No.']),
                         ['reverend', 'ames', 'saw', 'number', 'five', 'i', 'said', 'no'])

    def test_pronounce_texts(self):
        # Without a context network
        fake = FakeG2p()
        with unittest.mock.patch.object(frontend, '_G2P', fake):
            pronunciations = frontend.pronounce_texts(['I read it.'])
        self.assertEqual(pronunciations, {'i': ['K', 'AE0'], 'read': ['K'] * 4 + ['AE0'], 'it': ['K', 'K', 'AE0']})
        self.assertEqual(pronunciations.state, fake.state())

        # With the G2P network's context network, the ambiguous words' chosen readings are pronounced too
        g2p = frontend.get_g2p()
        pronunciations = frontend.pronounce_texts(['He will lead the way. Lead is heavy. I read it yesterday.'])
        self.assertTrue(g2p.ambiguous('lead'))
        self.assertEqual(pronunciations['lead'], g2p.pronounce('lead'))
        self.assertEqual(set(pronunciations.state), {'readings', 'ambiguous', 'choices'})
        self.assertTrue(any(window[len(window) // 2] == 'lead' for window in pronunciations.state['choices']))

    def test_number_words(self):
        self.assertEqual(frontend.number_words(0), ['zero'])
        self.assertEqual(frontend.number_words(19), ['nineteen'])
        self.assertEqual(frontend.number_words(40), ['forty'])
        self.assertEqual(frontend.number_words(42), ['forty', 'two'])
        self.assertEqual(frontend.number_words(100), ['one', 'hundred'])
        self.assertEqual(frontend.number_words(1999), ['one', 'thousand', 'nine', 'hundred', 'ninety', 'nine'])
        self.assertEqual(frontend.number_words(2000000), ['two', 'million'])
        self.assertEqual(frontend.number_words(3000000007), ['three', 'billion', 'seven'])
        self.assertEqual(frontend.number_words(4 * 10 ** 12 + 2), ['four', 'trillion', 'two'])

    def test_year_words(self):
        self.assertEqual(frontend.year_words(1972), ['nineteen', 'seventy', 'two'])
        self.assertEqual(frontend.year_words(1900), ['nineteen', 'hundred'])
        self.assertEqual(frontend.year_words(1905), ['nineteen', 'oh', 'five'])
        self.assertEqual(frontend.year_words(1066), ['ten', 'sixty', 'six'])
        self.assertEqual(frontend.year_words(2024), ['twenty', 'twenty', 'four'])

        # A millennium's first years are numbers
        self.assertEqual(frontend.year_words(2000), ['two', 'thousand'])
        self.assertEqual(frontend.year_words(2001), ['two', 'thousand', 'one'])
        self.assertEqual(frontend.year_words(1009), ['one', 'thousand', 'nine'])

    def test_number_token_words(self):
        # Four-digit numbers are years; with commas, or with leading zeros, numbers
        self.assertEqual(frontend.number_token_words('1972'), ['nineteen', 'seventy', 'two'])
        self.assertEqual(frontend.number_token_words('1,972'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two'])
        self.assertEqual(frontend.number_token_words('0972'), ['nine', 'hundred', 'seventy', 'two'])
        self.assertEqual(frontend.number_token_words('19720'), ['nineteen', 'thousand', 'seven', 'hundred', 'twenty'])
        self.assertEqual(frontend.number_token_words('23,000,000'), ['twenty', 'three', 'million'])

        # Past 15 digits, digit by digit (leading zeros aside)
        self.assertEqual(frontend.number_token_words('999,999,999,999,999')[:5], ['nine', 'hundred', 'ninety', 'nine', 'trillion'])
        self.assertEqual(frontend.number_token_words('0001234567890123456'),
                         ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'one', 'two', 'three',
                          'four', 'five', 'six'])
        self.assertEqual(len(frontend.number_token_words('7' * 400)), 400)
        self.assertEqual(frontend.number_token_words('0'), ['zero'])

        # Decimals
        self.assertEqual(frontend.number_token_words('3.14'), ['three', 'point', 'one', 'four'])
        self.assertEqual(frontend.number_token_words('1972.5'),
                         ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two', 'point', 'five'])

        # Ordinals
        self.assertEqual(frontend.number_token_words('1st'), ['first'])
        self.assertEqual(frontend.number_token_words('12th'), ['twelfth'])
        self.assertEqual(frontend.number_token_words('22nd'), ['twenty', 'second'])
        self.assertEqual(frontend.number_token_words('40th'), ['fortieth'])
        self.assertEqual(frontend.number_token_words('1900th'), ['one', 'thousand', 'nine', 'hundredth'])

        # Plurals
        self.assertEqual(frontend.number_token_words('1970s'), ['nineteen', 'seventies'])
        self.assertEqual(frontend.number_token_words('1900s'), ['nineteen', 'hundreds'])
        self.assertEqual(frontend.number_token_words("80's"), ['eighties'])
        self.assertEqual(frontend.number_token_words('6s'), ['sixes'])

        # Percentages
        self.assertEqual(frontend.number_token_words('50%'), ['fifty', 'percent'])
        self.assertEqual(frontend.number_token_words('3.5%'), ['three', 'point', 'five', 'percent'])
        self.assertEqual(frontend.number_token_words('1972%'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two', 'percent'])

    def test_money_words(self):
        # Dollars and cents
        self.assertEqual(frontend.money_words('$1,234.56'),
                         ['one', 'thousand', 'two', 'hundred', 'thirty', 'four', 'dollars', 'and', 'fifty', 'six', 'cents'])
        self.assertEqual(frontend.money_words('$1.01'), ['one', 'dollar', 'and', 'one', 'cent'])
        self.assertEqual(frontend.money_words('$' + '1' * 16), ['one'] * 16 + ['dollars'])
        self.assertEqual(frontend.money_words('$' + '2' * 16 + '.25'), ['two'] * 16 + ['dollars', 'and', 'twenty', 'five', 'cents'])
        self.assertEqual(frontend.money_words('$1.00'), ['one', 'dollar'])
        self.assertEqual(frontend.money_words('$0.50'), ['fifty', 'cents'])
        self.assertEqual(frontend.money_words('$0.00'), ['zero', 'dollars'])

        # Whole amounts, decimals, and scales
        self.assertEqual(frontend.money_words('$1'), ['one', 'dollar'])
        self.assertEqual(frontend.money_words('$0'), ['zero', 'dollars'])
        self.assertEqual(frontend.money_words('$1972'), ['one', 'thousand', 'nine', 'hundred', 'seventy', 'two', 'dollars'])
        self.assertEqual(frontend.money_words('$1.5'), ['one', 'point', 'five', 'dollars'])
        self.assertEqual(frontend.money_words('$1 million'), ['one', 'million', 'dollars'])
        self.assertEqual(frontend.money_words('$2.75\nbillion'), ['two', 'point', 'seven', 'five', 'billion', 'dollars'])

        # Pounds and euros
        self.assertEqual(frontend.money_words('£1.50'), ['one', 'pound', 'and', 'fifty', 'pence'])
        self.assertEqual(frontend.money_words('£0.01'), ['one', 'penny'])
        self.assertEqual(frontend.money_words('€20'), ['twenty', 'euros'])

    def test_time_words(self):
        self.assertEqual(frontend.time_words('10:30'), ['ten', 'thirty'])
        self.assertEqual(frontend.time_words('10:05'), ['ten', 'oh', 'five'])
        self.assertEqual(frontend.time_words('12:00'), ['twelve', "o'clock"])
        self.assertEqual(frontend.time_words('14:00'), ['fourteen', 'hundred'])
        self.assertEqual(frontend.time_words('0:00'), ['zero', 'hundred'])

        # With a.m. or p.m.
        self.assertEqual(frontend.time_words('9:00 p.m.'), ['nine', 'pee', 'em'])
        self.assertEqual(frontend.time_words('10:30am'), ['ten', 'thirty', 'ay', 'em'])
        self.assertEqual(frontend.time_words('6pm'), ['six', 'pee', 'em'])
        self.assertEqual(frontend.time_words('7 a.m'), ['seven', 'ay', 'em'])

    def test_text_items(self):
        self.assertEqual(frontend.text_items('Hi, Mr. Lee! 2.5 ‘ok’? ; .'), [
            ('hi', 0), ({'end': 'comma'}, 2), ('mister', 4), ('lee', 8), ({'end': 'exclamation'}, 11), ('two', 13),
            ('point', 13), ('five', 13), ('ok', 18), ({'end': 'question'}, 21)
        ])
        self.assertEqual(frontend.text_items('End.'), [('end', 0), ({'end': 'period'}, 0)])
        self.assertEqual(frontend.text_items('A word'), [('a', 0), ('word', 2), ({'end': 'period'}, 6)])
        self.assertEqual(frontend.text_items('Two . words'), [('two', 0), ({'end': 'period'}, 4), ('words', 6),
                                                              ({'end': 'period'}, 11)])
        self.assertEqual(frontend.text_items(''), [])

        # Numbers - commas group thousands, suffixes are ordinals and plurals, not words
        self.assertEqual(frontend.text_items('1,000,000 1st 4things 12,3456'), [
            ('one', 0), ('million', 0), ('first', 10), ('four', 14), ('things', 15), ('twelve', 22), ({'end': 'comma'}, 24),
            ('thirty', 25), ('four', 25), ('fifty', 25), ('six', 25), ({'end': 'period'}, 29)
        ])

        # Money, percentages, times, and a.m. - not the word am
        self.assertEqual(frontend.text_items('$5 million 50% 6pm a.m. am'), [
            ('five', 0), ('million', 0), ('dollars', 0), ('fifty', 11), ('percent', 11), ('six', 15), ('pee', 15), ('em', 15),
            ('ay', 19), ('em', 19), ('am', 24), ({'end': 'period'}, 26)
        ])

    def test_text_items_parentheses(self):
        # An aside's parentheses are comma breaks, unless a punctuation mark right after one ends the phrase
        self.assertEqual(frontend.text_items('Go (now, ok) home (1818); (Aside.) the end (truly)'), [
            ('go', 0), ({'end': 'comma'}, 3), ('now', 4), ({'end': 'comma'}, 7), ('ok', 9), ({'end': 'comma'}, 11),
            ('home', 13), ({'end': 'comma'}, 18), ('eighteen', 19), ('eighteen', 19), ({'end': 'comma'}, 24), ('aside', 27),
            ({'end': 'period'}, 27), ('the', 35), ('end', 39), ({'end': 'comma'}, 43), ('truly', 44), ({'end': 'period'}, 50)
        ])

    def test_text_items_accents(self):
        # Accented letters lose their accents and the ligatures read as English says them, the offsets unchanged
        self.assertEqual(frontend.text_items('Señor Amadís, Œdipus Cæsar Straße ŋa'), [
            ('senor', 0), ('amadis', 6), ({'end': 'comma'}, 12), ('edipus', 14), ('cesar', 21), ('strase', 27), ('a', 35),
            ({'end': 'period'}, 36)
        ])
        self.assertEqual(frontend.plain_letters('ÀÉÎÕÜÑçøÆœŁđ'), 'AEIOUNcoEeLd')

    def test_context_sequence(self):
        self.assertEqual(frontend.context_sequence(frontend.text_items('Hi, there! Ok? Yes.')),
                         ['hi', ',', 'there', '!', 'ok', '?', 'yes', '.'])

    def test_text_to_phrases(self):
        phones = {'who': ['HH', 'UW1'], 'is': ['IH1', 'Z'], 'it': ['IH1', 'T'], 'go': ['G', 'OW1']}
        phrases = frontend.text_to_phrases('Who is it? Go!', phones.get)
        self.assertEqual([phrase['end'] for phrase in phrases], ['wh', 'exclamation'])
        self.assertEqual(phrases[0]['words'][1], {'text': 'is', 'phones': [{'ph': 'IH', 'stress': 0}, {'ph': 'Z', 'stress': None}]})
        self.assertEqual(phrases[1]['words'][0]['phones'][1], {'ph': 'OW', 'stress': 1})
        self.assertEqual([phrase['end'] for phrase in frontend.text_to_phrases('Is it?', phones.get)], ['question'])

        # The aligned readings replace words' phones
        phrases = frontend.text_to_phrases('Go go.', phones.get, readings={1: ['G', 'UW1']})
        self.assertEqual([word['phones'][1]['ph'] for word in phrases[0]['words']], ['OW', 'UW'])

    def test_tokenize(self):
        result = frontend.tokenize('Go, go.', {'go': ['G', 'OW1']}.get)
        self.assertEqual(result['tokens'], ['<bos>', 'G', 'OW1', '<comma>', 'G', 'OW1', '<period>'])
        self.assertEqual(result['ids'], [frontend.TOKEN_IDS[token] for token in result['tokens']])
        self.assertEqual(result['words'], [{'text': 'go', 'start': 1, 'end': 3}, {'text': 'go', 'start': 4, 'end': 6}])
        result = frontend.tokenize('Big cat.', {'big': ['B', 'IH1', 'G'], 'cat': ['K', 'XX1']}.get)
        self.assertEqual(result['tokens'], ['<bos>', 'B', 'IH1', 'G', '_', 'K', 'XX1', '<period>'])
        self.assertEqual(result['ids'][-2], frontend.TOKEN_IDS['<unk>'])
        self.assertEqual(frontend.phone_token({'ph': 'K', 'stress': None}), 'K')

    def test_tokenize_javascript(self):
        # The front end tokenizes as lib/tokens.js does (the parity check - tools/parity.js - with the package's G2P network
        # and context network)
        parity.inputs(self, 'tokenize', {'texts': TEXTS})
        expected = parity.outputs('tokenize')
        self.assertEqual(len(expected), len(TEXTS))
        for text, js_result in zip(TEXTS, expected):
            self.assertEqual(frontend.tokenize(text), js_result, text)
