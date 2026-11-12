# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest
import unittest.mock

import numpy

from picovox_train import spelled

from . import helpers


# The letters' words' pronunciations ("a" the article - its name is "ay"'s; "i"'s is "eye"'s)
LETTERS = {
    'a': 'AH0', 'ay': 'EY1', 'b': 'B IY1', 'c': 'S IY1', 'd': 'D IY1', 'e': 'IY1', 'f': 'EH1 F', 'g': 'JH IY1', 'h': 'EY1 CH',
    'i': 'AY1', 'eye': 'AY1', 'j': 'JH EY1', 'k': 'K EY1', 'l': 'EH1 L', 'm': 'EH1 M', 'n': 'EH1 N', 'o': 'OW1', 'p': 'P IY1',
    'q': 'K Y UW1', 'r': 'AA1 R', 's': 'EH1 S', 't': 'T IY1', 'u': 'Y UW1', 'v': 'V IY1', 'w': 'D AH1 B AH0 L Y UW0', 'x': 'EH1 K S',
    'y': 'W AY1', 'z': 'Z IY1'
}

# Tiny settings
SETTINGS = {'shortCounts': {'2': 20, '3': 1}, 'synthetic': 5, 'alpha': 0.5}


# Tiny data (g2p.py load_data's format) - the letters, and a few words: each a list of its pronunciations
def tiny_data():
    words = {
        **{word: [phones] for word, phones in LETTERS.items()},
        'tv': ['T IY1 V IY1', 'T EH1 L AH0 V IH2 ZH AH0 N'], 'us': ['AH1 S', 'Y UW2 EH1 S'], 'ai': ['AY1', 'EY1 AY1'], 'cat': ['K AE1 T'],
        'ads': ['AE1 D Z'], 'html': ['EY1 CH T IY1 EH1 M EH1 L'], 'noaa': ['N OW1 AH0', 'EH1 N OW1 EY1 EY1'],
        'asea': ['AH0 S IY1', 'EY1 EH1 S IY1 EY1'], 'nasa': ['N AE1 S AH0'], 'stop': ['S T AA1 P'], 'whip': ['W IH1 P'],
        'isbn': ['AY1 EH1 S B IY1 EH1 N'], 'don\'t': ['D OW1 N T'], 'z' * 33: ['Z IY1']
    }
    pronunciations = {word: [variant.split() for variant in variants] for word, variants in words.items()}
    return {
        'first': {word: variants[0] for word, variants in pronunciations.items()}, 'all': pronunciations,
        'labels': {'frodo': ['F R OW1 D OW0']}, 'monitor': ['whip', 'isbn'],
        # (the listeners' texts' counts aren't the labels' - "ai" stays spelled)
        'weightCounts': {'ai': 500}, 'evalCounts': {}
    }


# The words spelled alone (train/data/g2p-capitals.json) - a common word, a spelled string, and a string never labeled -
# and the words Wiktionary spells - a word no dictionary has, and a common short word
CAPITALS = {'alone': {'us': {}, 'tv': {}, 'whip': {}}, 'spelled': ['ucla', 'us']}

# The books' word counts (train/data/g2p-words.json)
WORDS = {'weightCounts': {'tv': 50, 'us': 100, 'ai': 5, 'cat': 3, 'asea': 9, 'gandalf': 2, 'stop': 40}, 'evalCounts': {'us': 20, 'nasa': 1}}


class TestSpelled(unittest.TestCase):

    def setUp(self):
        patch = helpers.json_files(spelled, {spelled.CAPITALS_PATH: CAPITALS, spelled.g2p.WORDS_PATH: WORDS})
        patch.start()
        self.addCleanup(patch.stop)

    def test_letter_names(self):
        names = spelled.letter_names(tiny_data()['first'])
        self.assertEqual(len(names), 26)
        self.assertEqual(names['a'], ['EY'])
        self.assertEqual(names['i'], ['AY'])
        self.assertEqual(names['w'], ['D', 'AH', 'B', 'AH', 'L', 'Y', 'UW'])
        self.assertEqual(spelling_text('toc', names), 'T IY OW S IY')
        self.assertEqual(spelled.unstressed(['EY1', 'CH']), ['EY', 'CH'])

    def test_spelled_labels(self):
        result = spelled.spelled_labels(tiny_data(), SETTINGS)
        labels = result['labels']

        # The short strings - every one: a common word read, unless the dictionary's first pronunciation spells it
        self.assertEqual(sum(1 for word in labels if len(word) <= 3), 26 ** 2 + 26 ** 3)
        self.assertEqual({word: labels[word] for word in ('us', 'tv', 'ai', 'toc', 'cat', 'ads', 'qx')},
                         {'us': False, 'tv': True, 'ai': True, 'toc': True, 'cat': False, 'ads': True, 'qx': True})

        # The long words - spelled by the dictionary's first pronunciation, or by another when the word isn't common; the
        # monitoring set's read words held out; the words that aren't letters, and the longest, not labeled
        self.assertEqual({word: labels[word] for word in ('html', 'noaa', 'asea', 'nasa', 'stop', 'frodo', 'gandalf', 'isbn')},
                         {'html': True, 'noaa': True, 'asea': False, 'nasa': False, 'stop': False, 'frodo': False, 'gandalf': False,
                          'isbn': True})
        self.assertEqual(result['heldOut'], ['whip'])
        self.assertNotIn('whip', labels)
        self.assertNotIn('don\'t', labels)
        self.assertNotIn('z' * 33, labels)
        self.assertEqual(result['counts']['us'], 120)
        self.assertEqual(result['alpha'], 0.5)

        # The words spelled alone - the file's that are read beside a word in capitals
        self.assertEqual(result['alone'], ['us'])

        # The words Wiktionary spells - spelled, but a common short word
        self.assertTrue(labels['ucla'])
        self.assertFalse(labels['us'])

        # The groups - the synthetic strings (consonants, unknown), spelled
        groups = result['groups']
        self.assertEqual(len(groups['synthetic']), 5)
        self.assertTrue(all(labels[word] and len(word) in (4, 5) and set(word) <= set(spelled.CONSONANTS) for word in groups['synthetic']))
        self.assertEqual(groups['longSpelled'], ['html', 'isbn', 'noaa', 'ucla'])
        self.assertEqual(groups['longWords'], ['asea', 'frodo', 'gandalf', 'nasa', 'stop'])
        self.assertEqual(groups['shortWords'], ['cat', 'us'])
        self.assertEqual(groups['shortDictionary'], ['ai', 'tv'])
        self.assertEqual(len(groups['shortSpelled']), 26 ** 2 + 26 ** 3 - 2)
        self.assertEqual(sorted(word for group in ('shortSpelled', 'longSpelled', 'synthetic', 'shortWords', 'longWords')
                                for word in groups[group]), sorted(labels))

        # A synthetic string that is a known word is drawn again
        data = tiny_data()
        data['labels'][groups['synthetic'][0]] = ['X']
        again = spelled.spelled_labels(data, SETTINGS)
        self.assertFalse(again['labels'][groups['synthetic'][0]])
        self.assertEqual(len(again['groups']['synthetic']), 5)
        self.assertNotIn(groups['synthetic'][0], again['groups']['synthetic'])

    def test_batch_sampler(self):
        result = spelled.spelled_labels(tiny_data(), SETTINGS)
        sample = spelled.batch_sampler(result)
        words, targets = sample(numpy.random.default_rng(1), 64)
        self.assertEqual(len(words), 64)
        self.assertEqual([beside for _, beside in targets], [1.0 if result['labels'][word] else 0.0 for word in words])
        self.assertEqual(sum(beside for _, beside in targets), 32)
        self.assertEqual({word: alone for word, (alone, beside) in zip(words, targets) if alone != beside}, {'us': 1.0})

        # The frequent long words by their counts - a word in no book never
        frequent = words[44:56]
        self.assertTrue(set(frequent) <= {'asea', 'gandalf', 'nasa', 'stop'}, frequent)
        self.assertEqual(sample(numpy.random.default_rng(1), 64), (words, targets))

    def test_spelled_errors(self):
        result = spelled.spelled_labels(tiny_data(), SETTINGS)

        # A head that finds three-letter strings and "stop" and "whip" spelled - alone too, but "us" and "tv"
        def decisions(words):
            return [[beside and word not in ('us', 'tv'), beside] for word in words
                    for beside in [len(word) == 3 or word in ('stop', 'us', 'whip')]]
        errors = spelled.spelled_errors(result, decisions)
        self.assertEqual(errors['aloneMissed'], ['us'])
        self.assertEqual(len(errors['aloneDiffers']), 26 ** 2 - 1 + 5 + 4 + 2)
        self.assertEqual(len(errors['spelledMissed']), 26 ** 2 - 1 + 5 + 4)
        self.assertEqual(errors['wordsSpelled'], ['us', 'stop', 'cat'])
        self.assertEqual(errors['heldOutSpelled'], ['whip'])
        self.assertEqual((errors['words'], errors['spelled'], errors['heldOut']), (7, 26 ** 2 + 26 ** 3 - 2 + 4 + 5, 1))

        # No held-out words
        result['heldOut'] = []
        errors = spelled.spelled_errors(result, decisions)
        self.assertEqual((errors['heldOutSpelled'], errors['heldOut']), ([], 0))


# A word's spelling, as text
def spelling_text(word, names):
    return ' '.join(spelled.spelling(word, names))
