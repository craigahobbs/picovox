# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy
import torch

from picovox_train import RECIPE_PATH, corpora, ctc, frontend, g2p, g2paudit, measure, read_json, tts
from picovox_train import books as books_module

from . import ctc_fixtures, data_fixtures, helpers
from .helpers import captured


# The tiny dictionary - a few dozen words, some with variants (hobbes's first is the error)
DICTIONARY = {
    'cat': [['K', 'AE1', 'T']],
    'cats': [['K', 'AE1', 'T', 'S']],
    "cat's": [['K', 'AE1', 'T', 'S']],
    'dog': [['D', 'AO1', 'G']],
    'dogs': [['D', 'AO1', 'G', 'Z']],
    'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']],
    'hobbes': [['HH', 'AA1', 'B', 'IY0', 'Z'], ['HH', 'AA1', 'B', 'Z']],
    'hobbs': [['HH', 'AA1', 'B', 'Z']],
    'sat': [['S', 'AE1', 'T']],
    'mat': [['M', 'AE1', 'T']],
    'hat': [['HH', 'AE1', 'T']],
    'bat': [['B', 'AE1', 'T']],
    'rat': [['R', 'AE1', 'T']],
    'log': [['L', 'AO1', 'G']],
    'fog': [['F', 'AO1', 'G']],
    'bog': [['B', 'AO1', 'G']],
    'hog': [['HH', 'AO1', 'G']],
    'tin': [['T', 'IH1', 'N']],
    'pin': [['P', 'IH1', 'N']],
    'win': [['W', 'IH1', 'N']],
    'bin': [['B', 'IH1', 'N']],
    'fin': [['F', 'IH1', 'N']],
    'frog': [['F', 'R', 'AA1', 'G']],
    'the': [['DH', 'AH0']]
}

# The words file - unknown words' labels, the derived words' stems, and the book counts
WORDS = {
    'labels': {'hobbes\'s': 'HH AA1 B IY0 Z IH0 Z', 'catz': 'K AE1 T S'},
    'stems': {'hobbes\'s': ['hobbes', 'derived'], 'catz': ['cat', 'spelling']},
    'evalCounts': {'the': 100, 'cat': 30, 'read': 25, 'dog': 5, 'hobbes': 3},
    'weightCounts': {'mat': 40, 'log': 30, 'hobbs': 22, 'frog': 21}
}

# The settings - a tiny model
CONFIG = {'model': {'embed': 4, 'hidden': 4}, 'batch': 8, 'alpha': 0.5, 'uniform': 0.5, 'unknown': 0.3, 'threads': 1,
          'stages': [{'name': 'base', 'steps': 2, 'lr': 0.003, 'warmup': 1, 'seed': 1}]}


# A tiny screening model (untrained, seeded)
def tiny_model():
    torch.manual_seed(0)
    return g2p.build_model({'embed': 4, 'hidden': 4})


class TestG2pAudit(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        directory = self.directory.name
        self.out = os.path.join(directory, 'out')
        # The G2P's data and settings, as objects
        self.patches = [helpers.json_files(g2p, {g2p.WORDS_PATH: WORDS, g2p.RECIPE_PATH: {'g2p': CONFIG}}),
                        unittest.mock.patch.object(corpora, 'read_dictionary', lambda: DICTIONARY)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.directory.cleanup()

    def test_unstressed_group_partition(self):
        self.assertEqual(g2paudit.unstressed('HH AA1 B Z'), 'HH AA B Z')
        self.assertEqual(g2paudit.group_key('hobbes\'s'), 'hobbes')
        self.assertEqual(g2paudit.group_key('forbes\''), 'forbes')
        self.assertEqual(g2paudit.group_key('jones\''), 'jones')
        self.assertEqual(g2paudit.group_key('o\''), 'o')
        self.assertEqual(g2paudit.group_key('\''), '\'')
        self.assertEqual(g2paudit.group_key('cat'), 'cat')
        words = ['cat', 'cat\'s', 'cats', 'dog', 'hobbes', 'hobbes\'s', 'read', 'sat', 'mat', 'tin', 'pin']
        folds = g2paudit.partition(words, 0)
        self.assertEqual(folds['cat'], folds['cat\'s'])
        self.assertEqual(folds['hobbes'], folds['hobbes\'s'])
        self.assertEqual(set(folds.values()), set(range(g2paudit.FOLDS)))
        self.assertEqual(folds, g2paudit.partition(list(reversed(words)), 0))
        self.assertNotEqual(folds, g2paudit.partition(words, 1))

    def train_folds(self, jobs=1):
        args = argparse.Namespace(out=self.out, repeats=2, repeat=None, steps=2, jobs=jobs)
        with captured(), unittest.mock.patch.object(g2paudit.multiprocessing, 'Pool', data_fixtures.FakePool):
            self.assertEqual(g2paudit.command_folds(args), 0)

    def test_folds(self):
        self.train_folds(jobs=2)
        with open(os.path.join(self.out, 'fold-1-4.pt.json'), encoding='utf-8') as file:
            result = json.load(file)
        self.assertGreater(result['heldOut'], 0)
        self.assertIn('heldOutWer', result)
        checkpoint = torch.load(os.path.join(self.out, 'fold-0-0.pt'), weights_only=False)
        self.assertEqual(checkpoint['settings']['unknown'], 0.0)
        self.assertEqual(checkpoint['settings']['steps'], 2)

        # Resumed - done before
        mtime = os.path.getmtime(os.path.join(self.out, 'fold-0-0.pt'))
        with captured():
            g2paudit.command_folds(argparse.Namespace(out=self.out, repeats=2, repeat=0, steps=None, jobs=1))
        self.assertEqual(os.path.getmtime(os.path.join(self.out, 'fold-0-0.pt')), mtime)

    def test_sequence_scores(self):
        model = tiny_model()
        words = ['cat', 'hobbes']
        every = {'cat': [['K', 'AE1', 'T']], 'hobbes': [['HH', 'AA1', 'B', 'IY0', 'Z'], ['HH', 'AA1', 'B', 'Z']]}
        scores = g2paudit.fold_scores(model, words, every)
        self.assertEqual(set(scores), set(words))
        prediction, label_score, prediction_score, variant_scores = scores['hobbes']
        self.assertIsInstance(prediction, str)
        self.assertLess(label_score, 0)
        self.assertLess(prediction_score, 0)
        self.assertEqual(len(variant_scores), 1)
        self.assertEqual(scores['cat'][3], [])
        self.assertEqual(g2paudit.fold_scores(model, ['cat'], every)['cat'][3], [])

        # A sequence's log probability is its steps' (and <eos>'s)
        one = g2paudit.sequence_log_probs(model, ['cat'], [['K', 'AE1', 'T']])[0]
        letters, targets = g2p.make_batch(['cat'], [['K', 'AE1', 'T']])
        with torch.no_grad():
            steps = g2paudit.step_log_probs(model, letters, targets)[0]
        self.assertAlmostEqual(one, float(sum(steps[index, target] for index, target in enumerate(targets[0].tolist()))), places=4)

    def test_second_guess(self):
        model = tiny_model()
        guess = g2paudit.second_guess(model, 'cat', ['K', 'AE1', 'T'])
        self.assertTrue(guess)
        self.assertNotEqual(guess, ['K', 'AE1', 'T'])

        # Greedy continuation stops at <eos>, or at the length limit
        letters = g2p.make_batch(['cat'], [[]])[0]
        with torch.no_grad():
            model.output.bias.fill_(0)
            model.output.bias[g2p.EOS] = 1000.0
        self.assertEqual(g2paudit.continue_greedy(model, letters, [g2p.PHONE_IDS['K']]), ['K'])
        with torch.no_grad():
            model.output.bias[g2p.EOS] = 0.0
            model.output.bias[g2p.PHONE_IDS['T']] = 1000.0
        self.assertEqual(g2paudit.continue_greedy(model, letters, [g2p.PHONE_IDS['K']]), ['K'] + ['T'] * (3 * 3 + 6 - 1))

    def test_book_capitals_and_catalog(self):
        books = os.path.join(self.directory.name, 'books')
        os.makedirs(books)
        with open(os.path.join(books, 'one.txt'), 'w', encoding='utf-8') as file:
            file.write('Hobbes said so. Then Mr. Hobbes and hobbes, Hobbes;\nHobbes\n\nHobbes went. HOBBES and Hobbes the cat, HOBBES.\n')
        with open(os.path.join(books, 'skip.md'), 'w', encoding='utf-8') as file:
            file.write('a Hobbes')
        counts = books_module.book_capitals(books, ['hobbes', 'cat'])
        # (mid-sentence: "Mr. Hobbes" no - after a period; "and hobbes" lowercase; ", Hobbes" yes; "; Hobbes" no; "and Hobbes" yes)
        self.assertEqual(counts['hobbes'], [2, 1])
        self.assertEqual(counts['cat'], [0, 1])
        catalog = os.path.join(self.directory.name, 'catalog.json')
        with open(catalog, 'w', encoding='utf-8') as file:
            json.dump({'books': [{'author': 'Charlotte Brontë', 'title': 'Jane Eyre'}, {'title': 'Leviathan'}, {'author': None}]}, file)
        self.assertEqual(g2paudit.catalog_words(catalog), {'charlotte', 'bronte', 'jane', 'eyre', 'leviathan'})

    def test_judge(self):
        variants = ['HH AA1 B IY0 Z', 'HH AA1 B Z']
        scores = [['HH AA1 B Z', -9.0, -1.0, [-1.0]], ['HH AA1 B Z', -8.0, -2.0, [-2.0]]]
        flagged, entry = g2paudit.judge(variants, scores)
        self.assertTrue(flagged)
        self.assertEqual(entry['flags'], ['prediction', 'variant'])
        self.assertEqual(entry['preferred'], ['HH AA1 B Z'])
        self.assertEqual(entry['models'], [['HH AA1 B Z', -9.0, -1.0], ['HH AA1 B Z', -8.0, -2.0]])
        self.assertFalse(entry['agree'])

        # One model unsure - not flagged by the prediction; the variant still preferred
        scores[1] = ['HH AA1 B Z', -3.0, -2.0, [-2.0 - g2paudit.MARGIN]]
        flagged, entry = g2paudit.judge(variants, scores)
        self.assertFalse(flagged)
        self.assertEqual(entry['flags'], [])

        # A stress-only variant isn't a flag; agreement
        flagged, entry = g2paudit.judge(['K AE1 T', 'K AE2 T'], [['K AE2 T', -5.0, -1.0, [-1.0]]])
        self.assertFalse(flagged)
        self.assertTrue(entry['agree'])

    def test_held_out_accuracy(self):
        scores = {'read': [['R EH1 D', -1, -1, []], ['R AY1 D', -1, -1, []]], 'zork': [['Z AO1 R K', -1, -1, []]]}
        labels = {'read': 'R IY1 D', 'zork': 'Z AO1 R K'}
        every = {'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']]}
        self.assertAlmostEqual(g2paudit.held_out_accuracy(scores, labels, every), 2 / 3)
        self.assertEqual(g2paudit.held_out_accuracy({}, {}, {}), 0)

    def test_suspects(self):
        self.train_folds()
        books = os.path.join(self.directory.name, 'books')
        os.makedirs(books)
        with open(os.path.join(books, 'one.txt'), 'w', encoding='utf-8') as file:
            file.write('Then Hobbes sat on the mat, and the cat sat too.\n')
        catalog = os.path.join(self.directory.name, 'catalog.json')
        with open(catalog, 'w', encoding='utf-8') as file:
            json.dump({'books': [{'author': 'Thomas Hobbes', 'title': 'Leviathan'}]}, file)
        relabel = os.path.join(self.directory.name, 'relabel.json')
        with open(relabel, 'w', encoding='utf-8') as file:
            json.dump({'relabel': {'read': 'R EH1 D'}}, file)

        # Every word flagged (the margin met by any prediction), every control agreed (the second guesses: one the label)
        def judge(variants, scores):
            return True, {'label': variants[0], 'models': [score[:3] for score in scores], 'agree': False, 'flags': ['prediction'],
                          'preferred': []}
        heteronyms_path = os.path.join(self.directory.name, 'labels.json')
        with open(heteronyms_path, 'w', encoding='utf-8') as file:
            json.dump({'words': {'read': {}}}, file)
        args = argparse.Namespace(out=self.out, repeats=2, books=books, catalog=catalog, relabel=relabel, heteronyms=heteronyms_path,
                                  threads=1, log=None)
        with captured(), unittest.mock.patch.object(g2paudit, 'judge', judge):
            self.assertEqual(g2paudit.command_suspects(args), 0)
        with open(os.path.join(self.out, 'suspects.json'), encoding='utf-8') as file:
            suspects = json.load(file)
        self.assertEqual(suspects[0]['word'], 'the')
        entries = {entry['word']: entry for entry in suspects}
        self.assertEqual(len(entries['hobbes']['models']), 2)
        self.assertEqual(entries['hobbes']['variants'], ['HH AA1 B IY0 Z', 'HH AA1 B Z'])
        self.assertEqual((entries['hobbes']['capitalized'], entries['hobbes']['proper'], entries['hobbes']['catalog']), (1, 1.0, True))
        self.assertEqual((entries['cat']['capitalized'], entries['cat']['lowercase'], entries['cat']['proper']), (0, 1, 0.0))
        self.assertIsNone(entries['dog']['proper'])
        self.assertTrue(entries['read']['relabeled'])
        self.assertTrue(entries['read']['heteronym'])
        self.assertFalse(entries['cat']['heteronym'])
        self.assertTrue(os.path.exists(os.path.join(self.out, 'scores', 'fold-1-4.json')))

        # The controls - agreed words in the books often enough, of one pronunciation; their second guesses
        def agree(variants, scores):
            return False, {'label': variants[0], 'models': scores, 'agree': True, 'flags': [], 'preferred': []}
        args = argparse.Namespace(out=self.out, repeats=2, books=None, catalog=None, relabel=None, heteronyms=None, threads=1, log=None)
        with captured(), unittest.mock.patch.object(g2paudit, 'judge', agree), \
                unittest.mock.patch.object(g2paudit, 'CONTROL_COUNT', 20), \
                unittest.mock.patch.object(g2paudit, 'second_guess',
                                           lambda model, word, phones: phones if word == 'frog' else ['HH', 'OW1', 'B', 'Z']):
            g2paudit.command_suspects(args)
        with open(os.path.join(self.out, 'controls.json'), encoding='utf-8') as file:
            controls = json.load(file)
        with open(os.path.join(self.out, 'suspects.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file), [])
        self.assertEqual(controls, [{'word': 'hobbs', 'label': 'HH AA1 B Z', 'alternative': 'HH OW1 B Z', 'count': 22}])

    # A clip's alignment record (tts.py's) - its tokens, and its phones' labels ([phone, start, end] - None: not aligned)
    @staticmethod
    def alignment(tokens, labels, seconds):
        return {'id': 'unused', 'tokens': tokens, 'seconds': seconds, 'labels': labels}

    def test_tokens(self):
        # A clip's words by its alignment - a word without phones, and the words past its alignment's, have no tokens
        hobbes = ['HH', 'AA1', 'B', 'Z']
        first = {'id': 'c1', 'book': 'pg1.txt', 'voice': 'Trevor', 'text': 'Hobbes saw Hobbes.'}
        first_labels = [['pau', 0.0, 0.1]] + [[phone, 0.1 * index, 0.1 * index + 0.1] for index, phone in enumerate(hobbes, 1)] + \
            [[phone, 0.1 * index + 0.1, 0.1 * index + 0.2] for index, phone in enumerate(hobbes, 5)] + [['pau', 1.0, 1.2]]
        first_record = self.alignment(['<bos>', *hobbes, '_', '_', *hobbes, '<period>'], first_labels, 1.2)
        second = {'id': 'c2', 'book': 'pg2.txt', 'voice': 'Harper', 'text': 'The cat sat and Hobbes.'}
        second_labels = [['pau', 0.0, 0.1], ['DH', 0.1, 0.2], ['AH0', 0.2, 0.3], ['K', 0.3, 0.4], ['AE1', 0.4, 0.5], ['T', 0.5, 0.6],
                         ['pau', 0.6, 0.8]]
        second_record = self.alignment(['<bos>', 'DH', 'AH0', '_', 'K', 'AE1', 'T', '<period>'], second_labels, 0.8)
        # (clips not aligned - no record, or no labels)
        unaligned = [({'id': 'c3', 'book': 'pg1.txt', 'text': 'Hobbes.'}, None),
                     ({'id': 'c4', 'book': 'pg1.txt', 'text': 'Hobbes.'}, self.alignment(['<bos>', *hobbes, '<period>'], None, 1.0))]
        clips = [(first, first_record), (second, second_record)] + unaligned
        found = g2paudit.find_tokens({'hobbes', 'saw', 'cat'}, clips, lambda clip: f'{clip["voice"]}/{clip["id"]}.wav')
        self.assertEqual(sorted(found), ['cat', 'hobbes'])
        self.assertEqual([(token['clip'], token['wav'], token['book']) for token in found['hobbes']],
                         [('c1', 'Trevor/c1.wav', 'pg1.txt'), ('c1', 'Trevor/c1.wav', 'pg1.txt')])
        timed = measure.word_times(first_record)
        self.assertEqual([token['segment'] for token in found['hobbes']],
                         [measure.word_context(timed, 0, 1.2), measure.word_context(timed, 2, 1.2)])
        self.assertEqual([round(seconds, 3) for seconds in found['hobbes'][0]['segment'][:2]], [0.0, 0.62])
        start, end, before, after = found['cat'][0]['segment']
        self.assertEqual((round(start, 3), round(end, 3), before, after), (0.1, 0.72, ['DH', 'AH0'], []))

        # Up to count tokens, a book at a time
        tokens = [{'clip': f'{book}-{number}', 'book': book} for book in ('pg1.txt', 'pg2.txt') for number in range(3)]
        chosen = g2paudit.choose_tokens(tokens, 3, g2paudit.random.Random(1))
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len({token['book'] for token in chosen[:2]}), 2)
        self.assertEqual(len(g2paudit.choose_tokens(tokens[:2], 5, g2paudit.random.Random(1))), 2)

    def test_audio_plan(self):
        suspects = [{'word': 'hobbes', 'label': 'HH AA1 B IY0 Z', 'models': [['HH AA1 B Z', -9, -1], ['HH AA1 B Z', -9, -1]],
                     'variants': ['HH AA1 B IY0 Z', 'HH AA2 B Z', 'HH OW1 B Z']}]
        controls = [{'word': 'cat', 'label': 'K AE1 T', 'alternative': 'K AA1 T'},
                    {'word': 'dog', 'label': 'D AO1 G', 'alternative': 'D AA1 G'}]
        token = {'clip': 'c1', 'book': 'pg1.txt'}
        other = {'clip': 'c2', 'book': 'pg2.txt'}
        found = {'hobbes': [token, other], 'hobbs': [token], 'cat': [token, other], 'dog': [token]}
        suspects.append(dict(suspects[0], word='hobbs'))
        suspects.append({'word': 'the', 'label': 'DH AH0', 'models': [['DH IY0', -9, -1]], 'variants': ['DH AH0']})
        found['the'] = [token, other]
        controls.append({'word': 'often', 'label': 'AO1 F AH0 N', 'alternative': 'AO1 F IH0 N'})
        found['often'] = [token, other]
        with unittest.mock.patch.object(g2paudit, 'MIN_BOOKS', 2):
            plan = g2paudit.audio_plan(suspects, controls, found)
        self.assertEqual([(item['word'], item['control']) for item in plan],
                         [('hobbes', False), ('hobbes', False), ('cat', True), ('cat', True)])
        self.assertEqual(plan[0]['candidates'], ['HH AA1 B IY0 Z', 'HH AA1 B Z', 'HH OW1 B Z'])
        with unittest.mock.patch.object(g2paudit, 'CONTROL_WORDS', 0), unittest.mock.patch.object(g2paudit, 'MIN_BOOKS', 2):
            self.assertEqual(len(g2paudit.audio_plan(suspects, controls, found)), 2)

    def test_unfixable(self):
        self.assertEqual(g2paudit.unfixable('and', 'AH0 N D', 'AE1 N D'), 'function word')
        self.assertEqual(g2paudit.unfixable('cat', 'K AE1 T', 'K AE2 T'), 'stress only')
        self.assertEqual(g2paudit.unfixable('xi', 'SH IY1', 'Z IY1'), 'a letter or an abbreviation')
        self.assertEqual(g2paudit.unfixable('eighth', 'EY1 T TH', 'EY1 TH'), 'consonant-drop only')
        self.assertEqual(g2paudit.unfixable('often', 'AO1 F AH0 N', 'AO1 F IH0 N'), 'reduced only')
        self.assertEqual(g2paudit.unfixable('produce', 'P R AH0 D UW1 S', 'P R OW0 D UW1 S'), 'reduction only')
        self.assertIsNone(g2paudit.unfixable('hobbes', 'HH AA1 B IY0 Z', 'HH AA1 B Z'))
        self.assertIsNone(g2paudit.unfixable('cat', 'K AE1 T', 'K AA1 T'))
        self.assertEqual(g2paudit.unfixable('character', 'K EH1 R IH0 K T ER0', 'K AE1 R IH0 K T ER0'), 'merger only')
        self.assertEqual(g2paudit.unfixable('longer', 'L AO1 NG G ER0', 'L AA1 NG G ER0'), 'merger only')
        self.assertIsNone(g2paudit.unfixable('bet', 'B EH1 T', 'B AE1 T'))
        self.assertEqual(g2paudit.unfixable('character', 'K EH1 R IH0 K T ER0', 'K AE1 R AH0 K T ER0'), 'merger only')

    def test_audio_and_adjudicate(self):
        os.makedirs(self.out)
        suspects = [
            {'word': 'hobbes', 'label': 'HH AA1 B IY0 Z', 'models': [['HH AA1 B Z', -9.0, -1.0], ['HH AA1 B Z', -8.0, -1.0]],
             'variants': ['HH AA1 B IY0 Z', 'HH AA1 B Z'], 'relabeled': False, 'heteronym': False, 'count': 3, 'proper': 1.0,
             'catalog': True},
            {'word': 'sat', 'label': 'S AE1 T', 'models': [['S AA1 T', -9.0, -1.0], ['S AA1 T', -8.0, -1.0]],
             'variants': ['S AE1 T'], 'relabeled': False, 'heteronym': False, 'count': 0, 'proper': None, 'catalog': False},
            {'word': 'read', 'label': 'R IY1 D', 'models': [['R EH1 D', -9.0, -1.0]], 'variants': ['R IY1 D', 'R EH1 D'],
             'relabeled': True, 'heteronym': True, 'count': 25, 'proper': None, 'catalog': False},
            {'word': 'are', 'label': 'AA1 R', 'models': [['ER0', -9.0, -1.0]], 'variants': ['AA1 R', 'ER0'],
             'relabeled': False, 'heteronym': False, 'count': 99, 'proper': None, 'catalog': False},
            {'word': 'close', 'label': 'K L OW1 S', 'models': [['K L OW1 Z', -9.0, -1.0]], 'variants': ['K L OW1 S', 'K L OW1 Z'],
             'relabeled': False, 'heteronym': True, 'count': 9, 'proper': None, 'catalog': False}
        ]
        with open(os.path.join(self.out, 'suspects.json'), 'w', encoding='utf-8') as file:
            json.dump(suspects, file)
        with open(os.path.join(self.out, 'controls.json'), 'w', encoding='utf-8') as file:
            json.dump([{'word': 'cat', 'label': 'K AE1 T', 'alternative': 'K AA1 T', 'count': 30}], file)

        # The recognizer (its posteriors one row per 20 ms) prefers hobbes without the vowel, and cat as itself
        heard = {'HH AA1 B Z', 'K AE1 T'}
        segments, contexts = [], []

        def posteriors(model, device, samples_list):  # pylint: disable=unused-argument
            segments.extend(len(samples) for samples in samples_list)
            return [numpy.zeros((len(samples) // 320, len(ctc_fixtures.VOCAB))) for samples in samples_list]

        def likelihood(log_posteriors, before, phones, after, vocab):  # pylint: disable=unused-argument
            contexts.append((before, after))
            return 0.0 if ' '.join(phones) in heard else -10.0

        def audio(**options):
            args = argparse.Namespace(out=self.out, batch=2, threads=1, log=None, speak=False, jobs=None)
            vars(args).update(options)
            with captured() as (_, stderr), \
                    unittest.mock.patch.object(ctc, 'load_ctc', lambda device: (None, ctc_fixtures.VOCAB, device)), \
                    unittest.mock.patch.object(ctc, 'ctc_log_posteriors', posteriors), \
                    unittest.mock.patch.object(g2paudit, 'candidate_log_likelihood', likelihood), \
                    unittest.mock.patch.object(g2paudit, 'MIN_BOOKS', 1):
                self.assertEqual(g2paudit.command_audio(args), 0)
            return stderr.getvalue()

        texts = [('male', 'Hobbes ran on the mat.', 'train'), ('female', 'The cat and Hobbes ran.', 'train'),
                 ('male', 'Then Hobbes left.', 'train')]
        with ctc_fixtures.tts_corpus(texts) as (directory, settings, clips):
            log = audio()
            with open(os.path.join(self.out, 'audio.jsonl'), encoding='utf-8') as file:
                items = [json.loads(line) for line in file]
            # (sat - no tokens; read and close - the relabeled and the heteronym words, the G2P's own models')
            self.assertIn('audio: 4 tokens to score (0 before), 1 suspects', log)
            self.assertEqual(sorted((item['word'], item['control'], item['clip']) for item in items),
                             sorted([('hobbes', False, clip['id']) for clip in clips] + [('cat', True, clips[1]['id'])]))
            hobbes = next(item for item in items if item['word'] == 'hobbes')
            self.assertEqual((hobbes['candidates'], hobbes['scores'], hobbes['vote'], hobbes['book']),
                             (['HH AA1 B IY0 Z', 'HH AA1 B Z'], [-10.0, 0.0], 1, 'pg1.txt'))
            self.assertEqual(next(item['vote'] for item in items if item['word'] == 'cat'), 0)
            # (each token's segment, its neighbors' phones the context)
            self.assertEqual(len(segments), 4)
            self.assertTrue(all(before or after for before, after in contexts))

            # Resumed - the tokens before kept; with --speak, the extra clips' tokens (the suspects' words of too few)
            audit = os.path.join(directory, 'audit')
            extra_clip = {'id': 'extra1', 'voice': 'Harper', 'book': 'pg2.txt', 'text': 'Hobbes sat.'}
            labels, seconds = ctc_fixtures.clip_labels(extra_clip['text'])
            extra_record = {'id': 'extra1', 'tokens': frontend.tokenize(extra_clip['text'])['tokens'], 'seconds': seconds, 'labels': labels}
            os.makedirs(os.path.dirname(tts.wav_path(audit, extra_clip)))
            helpers.write_wav(tts.wav_path(audit, extra_clip), numpy.zeros(int(seconds * 24000)), 24000)
            calls = []

            def speak_extra(words, found, extra_settings, extra_directory, jobs):
                calls.append((words, sorted(found), extra_settings, extra_directory, jobs))
                return [(extra_clip, extra_record)]
            with unittest.mock.patch.object(g2paudit, 'speak_extra', speak_extra):
                log = audio(speak=True, jobs=3)
            self.assertEqual(calls, [({'hobbes', 'sat', 'are'}, ['cat', 'hobbes'], settings, directory, 3)])
            self.assertIn('audio: 2 tokens to score (4 before), 2 suspects', log)
            with open(os.path.join(self.out, 'audio.jsonl'), encoding='utf-8') as file:
                items = [json.loads(line) for line in file]
            # (sat - neither candidate heard: no vote)
            self.assertEqual(sorted((item['word'], item['clip'], item['book'], item['vote']) for item in items[4:]),
                             [('hobbes', 'extra1', 'pg2.txt', 1), ('sat', 'extra1', 'pg2.txt', None)])
            self.assertIn('0 tokens to score (6 before)', audio())

        # The verdicts - more votes, from several books, for the error (the votes added here)
        votes = [{'word': 'hobbes', 'control': False, 'clip': f'{book}-{number}', 'book': f'pg{book}.txt',
                  'candidates': ['HH AA1 B IY0 Z', 'HH AA1 B Z'], 'scores': [-9.0, -1.0], 'vote': 1}
                 for book in range(3) for number in range(2)]
        votes += [{'word': 'hobbes', 'control': False, 'clip': '9-0', 'book': 'pg9.txt',
                   'candidates': ['HH AA1 B IY0 Z', 'HH OW1 B Z', 'HH AA1 B Z'], 'scores': [-9.0, -5.0, -1.0], 'vote': 2}]
        votes += [{'word': 'are', 'control': False, 'clip': f'{book}-0', 'book': f'pg{book}.txt',
                   'candidates': ['AA1 R', 'ER0'], 'scores': [-9.0, -1.0], 'vote': 1} for book in range(5)]
        votes += [{'word': 'cat', 'control': True, 'clip': f'{book}-0', 'book': f'pg{book}.txt',
                   'candidates': ['K AE1 T', 'K AA1 T'], 'scores': [-1.0, -9.0], 'vote': vote}
                  for book, vote in [(number, 0) for number in range(8)] + [(8, 1), (9, None)]]
        with open(os.path.join(self.out, 'audio.jsonl'), 'a', encoding='utf-8') as file:
            file.write(''.join(json.dumps(item) + '\n' for item in votes))
        extra = os.path.join(self.directory.name, 'extra.json')
        with open(extra, 'w', encoding='utf-8') as file:
            json.dump({'hobbs': {'phones': 'HH AA1 B Z', 'evidence': {'was': 'HH AA1 B IY0 Z', 'note': 'hobbesian'}},
                       'hobbes': {'phones': 'HH AA1 B Z', 'evidence': {'was': 'HH AA1 B IY0 Z', 'date': '2026-10-17'}}}, file)
        with captured():
            self.assertEqual(g2paudit.command_adjudicate(argparse.Namespace(out=self.out, extra=extra, log=None)), 0)
        with open(os.path.join(self.out, 'verdicts.json'), encoding='utf-8') as file:
            verdicts = json.load(file)
        self.assertEqual(verdicts['summary'], {'error': 1, 'inconclusive': 1, 'excluded': 1, 'none': 2})
        self.assertEqual(next(item for item in verdicts['verdicts'] if item['word'] == 'are')['reason'], 'function word')
        self.assertEqual(verdicts['controls']['words'], 1)
        self.assertEqual(verdicts['controls']['falseErrors'], 0)
        self.assertEqual(verdicts['controls']['confirmedCorrect'], 1)
        self.assertEqual((verdicts['controls']['tokenVotes'], verdicts['controls']['tokenVotesForAlternative']), (10, 1))
        with open(os.path.join(self.out, 'fixes.json'), encoding='utf-8') as file:
            fixes = json.load(file)
        self.assertEqual(fixes['fixes'], {'hobbes': 'HH AA1 B Z', 'hobbs': 'HH AA1 B Z'})
        self.assertEqual(fixes['evidence']['hobbes']['audio']['votes'], [0, 11, 0])
        self.assertEqual(fixes['evidence']['hobbes']['audio']['books'], [0, 4, 0])
        self.assertEqual(fixes['evidence']['hobbes']['audio']['candidates'], ['HH AA1 B IY0 Z', 'HH AA1 B Z', 'HH OW1 B Z'])
        self.assertEqual(fixes['evidence']['hobbes']['folds'], [['HH AA1 B Z', 8.0], ['HH AA1 B Z', 7.0]])
        self.assertEqual(fixes['evidence']['hobbs']['note'], 'hobbesian')
        self.assertIn('date', fixes['evidence']['hobbs'])
        with captured():
            g2paudit.command_adjudicate(argparse.Namespace(out=self.out, extra=None, log=None))
        with open(os.path.join(self.out, 'fixes.json'), encoding='utf-8') as file:
            self.assertEqual(list(json.load(file)['fixes']), ['hobbes'])

    def test_speak_extra(self):
        settings = ctc_fixtures.tts_settings()
        directory = os.path.join(self.directory.name, 'corpus')
        audit = os.path.join(directory, 'audit')
        sentences = {
            os.path.join('books', 'pg5.txt'): ['Hobbes sat down.', 'Hobbes ran.'] + [f'Hobbes ran {number} times.' for number in range(8)],
            os.path.join('books', 'pg6.txt'): ['The cat sat.', 'A bad sentence sat.', 'They sat.']
        }
        sections, checks = [], []

        def book_paths(section):
            sections.append(section)
            return list(sentences)

        def checked(texts, check_settings, jobs):
            checks.append((texts, check_settings, jobs))
            return {sentence: None if 'bad' in sentence else sentence for sentence in texts}

        # The recognizer - 'They sat.' isn't aligned (its clip's 16 kHz samples are short)
        def read_16k(path):
            return numpy.zeros(8000 if path.endswith(f'{tts.clip_id("Harper", "They sat.")}.wav') else 16000)

        def ctc_labels(tokens, log_posteriors, vocab, samples):  # pylint: disable=unused-argument
            phones = [token for token in tokens if token not in frontend.BOUNDARIES]
            return None if len(samples) < 16000 else ([(phone, 0.1 * index, 0.1 * index + 0.1) for index, phone in enumerate(phones)],)
        speak = unittest.mock.Mock(return_value=b'RIFF-wav')
        check_engine = unittest.mock.Mock()
        load_ctc = unittest.mock.Mock(return_value=('model', ctc_fixtures.VOCAB, 'cpu'))
        posteriors = unittest.mock.Mock(side_effect=lambda model, device, samples_list: [numpy.zeros((50, 3)) for _ in samples_list])

        def extra(words, found):
            with unittest.mock.patch.object(tts, 'book_paths', book_paths), \
                    unittest.mock.patch.object(tts, 'book_candidates', lambda path: list(enumerate(sentences[path]))), \
                    unittest.mock.patch.object(tts, 'checked', checked), \
                    unittest.mock.patch.object(tts, 'check_engine', check_engine), \
                    unittest.mock.patch.object(tts, 'speak', speak), \
                    unittest.mock.patch.object(ctc, 'load_ctc', load_ctc), \
                    unittest.mock.patch.object(ctc, 'read_16k', read_16k), \
                    unittest.mock.patch.object(ctc, 'ctc_log_posteriors', posteriors), \
                    unittest.mock.patch.object(ctc, 'ctc_labels', ctc_labels), \
                    unittest.mock.patch.object(g2paudit, 'PER_WORD', 3):
                return g2paudit.speak_extra(words, found, settings, directory, 2)

        # No word of too few tokens - nothing spoken
        self.assertEqual(extra({'cat'}, {'cat': [{}] * 3}), [])
        self.assertEqual((sections, speak.call_count), ([], 0))

        # The words of too few tokens (PER_WORD less theirs) - the context labels' books' sentences with them (at most four
        # times the tokens wanted a word, checked by the front end), in turn by the voices; a sentence of two words once
        # in a voice
        result = extra({'hobbes', 'sat', 'cat'}, {'hobbes': [{}], 'cat': [{}] * 3})
        self.assertEqual(sections, [{'books': settings['texts']['books'], 'pins': settings['texts']['pins'],
                                     'texts': read_json(RECIPE_PATH)['context']['labels']['texts']}])
        self.assertEqual([(sorted(texts), check_settings, jobs) for texts, check_settings, jobs in checks],
                         [(sorted(['Hobbes sat down.', 'Hobbes ran.', 'Hobbes ran 0 times.', 'Hobbes ran 1 times.', 'Hobbes ran 2 times.',
                                   'Hobbes ran 3 times.', 'Hobbes ran 4 times.', 'Hobbes ran 5 times.', 'Hobbes sat down.', 'The cat sat.',
                                   'A bad sentence sat.', 'They sat.']), settings, 2)])
        self.assertEqual([(clip['voice'], clip['book'], clip['text']) for clip, _ in result],
                         [('Trevor', 'pg5.txt', 'Hobbes sat down.'), ('Harper', 'pg5.txt', 'Hobbes ran.'),
                          ('Trevor', 'pg6.txt', 'The cat sat.'), ('Harper', 'pg6.txt', 'They sat.')])
        self.assertEqual([clip['id'] for clip, _ in result], [tts.clip_id(clip['voice'], clip['text']) for clip, _ in result])
        check_engine.assert_called_once_with(settings['engine'])
        self.assertEqual([call.args for call in speak.call_args_list],
                         [(settings['engine'], clip['voice'], clip['text']) for clip, _ in result])
        for clip, _ in result:
            with open(tts.wav_path(audit, clip), 'rb') as file:
                self.assertEqual(file.read(), b'RIFF-wav')
        self.assertFalse(any(name.endswith('.part') for _, _, names in os.walk(audit) for name in names))

        # (each clip aligned - its tokens, seconds, and labels; None if the recognizer can't align it)
        load_ctc.assert_called_once_with()
        first, record = result[0]
        tokens = frontend.tokenize(first['text'])['tokens']
        self.assertEqual(record, {'id': first['id'], 'tokens': tokens, 'seconds': 1.0,
                                  'labels': [[phone, round(0.1 * index, 4), round(0.1 * index + 0.1, 4)] for index, phone in
                                             enumerate(token for token in tokens if token not in frontend.BOUNDARIES)]})
        self.assertEqual([record['labels'] is None for _, record in result], [False, False, False, True])
        self.assertEqual(result[3][1]['seconds'], 0.5)
        self.assertEqual(tts.read_alignments(audit), {clip['id']: record for clip, record in result})

        # Kept - spoken and aligned once
        self.assertEqual(extra({'hobbes', 'sat', 'cat'}, {'hobbes': [{}], 'cat': [{}] * 3}), result)
        self.assertEqual((speak.call_count, load_ctc.call_count, posteriors.call_count), (4, 1, 1))

    def test_verdict(self):
        candidates = ['A', 'B', 'C']
        self.assertEqual(g2paudit.verdict(candidates, [(1, book) for book in 'abcd'])['verdict'], 'error')
        self.assertEqual(g2paudit.verdict(candidates, [(1, 'a')] * 3 + [(1, 'b')] * 3)['verdict'], 'inconclusive')
        result = g2paudit.verdict(candidates, [(0, book) for book in 'abcdefgh'] + [(2, 'i'), (None, 'j')])
        self.assertEqual((result['verdict'], result['best'], result['votes'], result['books']), ('correct', 2, [8, 0, 1], [8, 0, 1]))
        self.assertEqual(g2paudit.verdict(candidates, [])['verdict'], 'none')
        self.assertIsNone(g2paudit.verdict(candidates, [])['lower'])

        # The correction - the dictionary's variant (its stress), else the best prediction
        entry = {'variants': ['HH AA1 B IY0 Z', 'HH AA1 B Z'], 'models': [['HH AA2 B Z', -9, -2], ['HH AA0 B Z', -9, -1]]}
        self.assertEqual(g2paudit.corrected(entry, 'HH AA2 B Z'), 'HH AA1 B Z')
        entry['variants'] = ['HH AA1 B IY0 Z']
        self.assertEqual(g2paudit.corrected(entry, 'HH AA2 B Z'), 'HH AA0 B Z')

    def test_main(self):
        with unittest.mock.patch.object(g2paudit, 'command_folds', lambda args: 7):
            self.assertEqual(g2paudit.main(['folds', '--out', self.out]), 7)
        with unittest.mock.patch.object(sys, 'argv', ['g2paudit']), captured(), \
                warnings.catch_warnings(), self.assertRaises(SystemExit) as raised:
            warnings.simplefilter('ignore', RuntimeWarning)
            runpy.run_module('picovox_train.g2paudit', run_name='__main__')
        self.assertEqual(raised.exception.code, 2)
