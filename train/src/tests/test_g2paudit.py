# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import json
import math
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy
import soundfile
import torch

from picovox_train import cremad, g2p, g2paudit, heteronyms

from .helpers import captured


# The tiny dictionary - a few dozen words, some with variants (hobbes's first is the error)
DICTIONARY = '''
cat K AE1 T
cats K AE1 T S
cat's K AE1 T S
dog D AO1 G
dogs D AO1 G Z
read R IY1 D
read(2) R EH1 D
hobbes HH AA1 B IY0 Z
hobbes(2) HH AA1 B Z
hobbs HH AA1 B Z
sat S AE1 T
mat M AE1 T
hat HH AE1 T
bat B AE1 T
rat R AE1 T
log L AO1 G
fog F AO1 G
bog B AO1 G
hog HH AO1 G
tin T IH1 N
pin P IH1 N
win W IH1 N
bin B IH1 N
fin F IH1 N
frog F R AA1 G
the DH AH0
'''

# The words file - unknown words' labels, the derived words' stems, and the book counts
WORDS = {
    'labels': {'hobbes\'s': 'HH AA1 B IY0 Z IH0 Z', 'catz': 'K AE1 T S'},
    'stems': {'hobbes\'s': ['hobbes', 'derived'], 'catz': ['cat', 'spelling']},
    'evalCounts': {'the': 100, 'cat': 30, 'read': 25, 'dog': 5, 'hobbes': 3},
    'weightCounts': {'mat': 40, 'log': 30, 'hobbs': 22, 'frog': 21}
}

# The settings - a tiny model
CONFIG = {'model': {'embed': 4, 'hidden': 4}, 'batch': 8, 'alpha': 0.5, 'uniform': 0.5, 'unknown': 0.3, 'threads': 1,
          'stages': [{'name': 'base', 'steps': 2, 'lr': 0.003, 'warmup': 1, 'seed': 1}], 'export': {'bits': 6}}


class FakePool:
    """multiprocessing.Pool in the test's process (so the workers' code is measured)"""

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, function, items):
        return [function(item) for item in items]


# A tiny screening model (untrained, seeded)
def tiny_model():
    torch.manual_seed(0)
    return g2p.build_model({'embed': 4, 'hidden': 4})


# The recognizer's fake vocabulary - every strict token, the blank, and a token no phone claims
def fake_vocab():
    tokens = sorted({token for tokens in g2paudit.STRICT_TOKENS.values() for token in tokens} | set(g2paudit.REDUCTIONS))
    return {'<pad>': 0, 'x': 1, **{token: index + 2 for index, token in enumerate(tokens)}}


# Log posteriors that spell tokens (each for frames frames, a blank frame between) - [frames, vocab]
def spelled(tokens, vocab, frames=3):
    rows = []
    for token in tokens:
        rows.extend([vocab[token]] * frames + [0])
    posteriors = numpy.full((len(rows), len(vocab)), 0.001)
    posteriors[numpy.arange(len(rows)), rows] = 1.0
    return numpy.log(posteriors / posteriors.sum(1, keepdims=True))


class TestG2pAudit(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        directory = self.directory.name
        dictionary_path = os.path.join(directory, 'cmudict.dict')
        with open(dictionary_path, 'w', encoding='utf-8') as file:
            file.write(DICTIONARY)
        words_path = os.path.join(directory, 'g2p-words.json')
        with open(words_path, 'w', encoding='utf-8') as file:
            json.dump(WORDS, file)
        config_path = os.path.join(directory, 'train.json')
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump({'g2p': CONFIG}, file)
        self.out = os.path.join(directory, 'out')
        self.patches = [unittest.mock.patch.object(g2p, 'WORDS_PATH', words_path),
                        unittest.mock.patch.object(g2p.read_pronunciations, '__defaults__', (dictionary_path,)),
                        unittest.mock.patch.object(g2p.load_config, '__defaults__', (config_path,))]
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
        with captured(), unittest.mock.patch.object(g2paudit.multiprocessing, 'Pool', FakePool):
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
        counts = g2paudit.book_capitals(books, ['hobbes', 'cat'])
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

    def test_ctc(self):
        vocab = fake_vocab()
        self.assertEqual(g2paudit.phone_token_ids('IY1', vocab), [vocab[token] for token in g2paudit.STRICT_TOKENS['IY']])
        self.assertIn(vocab['ə'], g2paudit.phone_token_ids('IY0', vocab))
        self.assertEqual(g2paudit.phone_token_ids('Z', {'z': 3}), [3])

        # Hobbes spoken without the vowel - the label (hobbies) less likely
        posteriors = spelled(['h', 'ɑː', 'b', 'z'], vocab)
        label = g2paudit.ctc_log_likelihood(posteriors, 'HH AA1 B IY0 Z'.split(), vocab)
        alternative = g2paudit.ctc_log_likelihood(posteriors, 'HH AA1 B Z'.split(), vocab)
        self.assertGreater(alternative, label + g2paudit.VOTE_MARGIN)
        self.assertEqual(g2paudit.vote([label, alternative]), 1)
        self.assertIsNone(g2paudit.vote([-1.0, -1.5]))
        self.assertEqual(g2paudit.ctc_log_likelihood(posteriors[:2], 'HH AA1 B Z'.split(), vocab), -math.inf)

        # An r-colored token of one frame - the vowel and its R as one
        posteriors = spelled(['s', 't', 'ɑːɹ', 't', 's'], vocab, frames=1)
        full = g2paudit.ctc_log_likelihood(posteriors, 'S T AA1 R T S'.split(), vocab)
        either = g2paudit.candidate_log_likelihood(posteriors, ['S'], 'T AA1 R T'.split(), ['S'], vocab)
        self.assertGreater(either, full + g2paudit.VOTE_MARGIN)
        self.assertGreater(either, g2paudit.candidate_log_likelihood(posteriors, ['S'], 'T ER1 T'.split(), ['S'], vocab))
        self.assertEqual(g2paudit.candidate_log_likelihood(posteriors, [], 'S T ER1 T S'.split(), [], vocab),
                         g2paudit.ctc_log_likelihood(posteriors, 'S T ER1 T S'.split(), vocab))

    def write_labels(self):
        labels = os.path.join(self.directory.name, 'lab')
        audio = os.path.join(self.directory.name, 'audio')
        utterances = {
            ('train-clean-100', '19', '198', '19_198_000000_000000'): (
                [(0.0, 0.3, 'mister'), (0.3, 0.7, 'hobbes'), (0.7, 0.9, 'sat'), (1.5, 1.8, 'hobbes')],
                [(0.0, 0.05, 'M'), (0.05, 0.1, 'IH1'), (0.1, 0.15, 'S'), (0.15, 0.2, 'T'), (0.2, 0.3, 'ER0'), (0.3, 0.4, 'HH'),
                 (0.4, 0.5, 'AA1'), (0.5, 0.6, 'B'), (0.6, 0.7, 'Z'), (0.7, 0.8, 'S'), (0.8, 0.85, 'AE1'), (0.85, 0.9, 'T'),
                 (0.9, 1.5, 'sil'), (1.5, 1.8, 'spn')]),
            ('dev-clean', '84', '121', '84_121_000000_000000'): (
                [(0.0, 0.4, 'hobbes'), (0.4, 0.6, 'cat')],
                [(0.0, 0.1, 'HH'), (0.1, 0.2, 'AA1'), (0.2, 0.3, 'B'), (0.3, 0.4, 'Z'), (0.4, 0.45, 'K'), (0.45, 0.5, 'AE1'),
                 (0.5, 0.6, 'T')])
        }
        for (subset, speaker, chapter, name), (words, phones) in utterances.items():
            for kind, rows in (('word', words), ('phone', phones)):
                directory = os.path.join(labels, kind, subset, speaker, chapter)
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, name + '.lab'), 'w', encoding='utf-8') as file:
                    file.write(''.join(f'{start}\t{end}\t{label}\n' for start, end, label in rows))
            with open(os.path.join(labels, 'word', subset, speaker, chapter, 'notes.txt'), 'w', encoding='utf-8') as file:
                file.write('not a label')
            directory = os.path.join(audio, subset, speaker, chapter)
            os.makedirs(directory, exist_ok=True)
            soundfile.write(os.path.join(directory, name + '.wav'), numpy.zeros(48000), 24000)
        return labels, audio

    def test_tokens(self):
        labels, _ = self.write_labels()
        found = g2paudit.find_tokens({'hobbes', 'cat'}, labels)
        self.assertEqual(len(found['hobbes']), 3)
        first = found['hobbes'][0]
        self.assertEqual(first['rel'], os.path.join('train-clean-100', '19', '198', '19_198_000000_000000.lab'))
        self.assertEqual((first['before'], first['after']), ((0.0, 0.3, 'mister'), (0.7, 0.9, 'sat')))
        self.assertIsNone(found['hobbes'][2]['before'])
        self.assertIsNone(found['cat'][0]['after'])

        # Contexts - adjacent neighbors' phones; a gap, an unknown word, or none: padding
        phones = heteronyms.read_lab(os.path.join(labels, 'phone', first['rel']))
        self.assertEqual(g2paudit.token_context(first, phones), (0.0, 0.9, ['M', 'IH1', 'S', 'T', 'ER0'], ['S', 'AE1', 'T']))
        last = found['hobbes'][1]
        start, end, before, after = g2paudit.token_context(last, phones)
        self.assertEqual((round(start, 3), round(end, 3), before, after), (1.38, 1.92, [], []))
        spoken = dict(first, after=(0.7, 0.9, 'sat'), before=(1.5, 1.8, 'hobbes'))
        spoken['start'] = 1.81
        self.assertEqual(g2paudit.token_context(spoken, phones)[2], [])
        gap = dict(first, after=(0.95, 1.0, 'sat'))
        self.assertEqual(g2paudit.token_context(gap, phones)[1], 0.82)
        unknown = dict(first, after=(0.7, 0.9, 'x'), end=0.7)
        self.assertEqual(g2paudit.token_context(dict(unknown, after=(1.5, 1.8, 'x'), end=1.49), phones)[3], [])

        # Up to count tokens, a speaker at a time
        tokens = [{'rel': os.path.join('s', speaker, 'c', f'{number}.lab')} for speaker in ('1', '2') for number in range(3)]
        chosen = g2paudit.choose_tokens(tokens, 3, g2paudit.random.Random(1))
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len({token['rel'].split(os.sep)[1] for token in chosen[:2]}), 2)
        self.assertEqual(len(g2paudit.choose_tokens(tokens[:2], 5, g2paudit.random.Random(1))), 2)

    def test_read_segment(self):
        path = os.path.join(self.directory.name, 'one.wav')
        soundfile.write(path, numpy.zeros((24000, 2)), 24000)
        self.assertEqual(len(g2paudit.read_segment(path, 0.25, 0.75)), 8000)

    def test_audio_plan(self):
        suspects = [{'word': 'hobbes', 'label': 'HH AA1 B IY0 Z', 'models': [['HH AA1 B Z', -9, -1], ['HH AA1 B Z', -9, -1]],
                     'variants': ['HH AA1 B IY0 Z', 'HH AA2 B Z', 'HH OW1 B Z']}]
        controls = [{'word': 'cat', 'label': 'K AE1 T', 'alternative': 'K AA1 T'},
                    {'word': 'dog', 'label': 'D AO1 G', 'alternative': 'D AA1 G'}]
        token = {'rel': os.path.join('s', '1', 'c', 'u.lab'), 'start': 0.0}
        other = dict(token, rel=os.path.join('s', '2', 'c', 'u.lab'))
        found = {'hobbes': [token, other], 'hobbs': [token], 'cat': [token, other], 'dog': [token]}
        suspects.append(dict(suspects[0], word='hobbs'))
        suspects.append({'word': 'the', 'label': 'DH AH0', 'models': [['DH IY0', -9, -1]], 'variants': ['DH AH0']})
        found['the'] = [token, other]
        controls.append({'word': 'often', 'label': 'AO1 F AH0 N', 'alternative': 'AO1 F IH0 N'})
        found['often'] = [token, other]
        with unittest.mock.patch.object(g2paudit, 'MIN_SPEAKERS', 2):
            plan = g2paudit.audio_plan(suspects, controls, found)
        self.assertEqual([(item['word'], item['control']) for item in plan],
                         [('hobbes', False), ('hobbes', False), ('cat', True), ('cat', True)])
        self.assertEqual(plan[0]['candidates'], ['HH AA1 B IY0 Z', 'HH AA1 B Z', 'HH OW1 B Z'])
        with unittest.mock.patch.object(g2paudit, 'CONTROL_WORDS', 0), unittest.mock.patch.object(g2paudit, 'MIN_SPEAKERS', 2):
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
        self.assertEqual(g2paudit.merged('M EH1 R IY0'), 'M E* R IY')
        self.assertEqual(g2paudit.merged('K EH1 R IH0 K T ER0'), 'K E* R @ K T ER')
        self.assertEqual(g2paudit.merged('M EH1'), 'M EH')

    def test_audio_and_adjudicate(self):
        labels, audio = self.write_labels()
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
        vocab = fake_vocab()

        # The recognizer hears hobbes without the vowel (and the context), cat as itself
        def posteriors(model, device, samples_list):  # pylint: disable=unused-argument
            return [spelled(['m', 'ɪ', 's', 't', 'ɚ', 'h', 'ɑː', 'b', 'z', 's', 'æ', 't'], vocab) for _ in samples_list]
        with captured(), unittest.mock.patch.object(heteronyms, 'LABELS', labels), \
                unittest.mock.patch.object(heteronyms, 'AUDIO', audio), \
                unittest.mock.patch.object(cremad, 'load_ctc', lambda device: (None, vocab, device)), \
                unittest.mock.patch.object(cremad, 'ctc_log_posteriors', posteriors), \
                unittest.mock.patch.object(g2paudit, 'MIN_SPEAKERS', 2):
            self.assertEqual(g2paudit.command_audio(argparse.Namespace(out=self.out, batch=2, threads=1, log=None)), 0)
        with open(os.path.join(self.out, 'audio.jsonl'), encoding='utf-8') as file:
            items = [json.loads(line) for line in file]
        # (sat - one speaker - no tokens)
        self.assertEqual([item['word'] for item in items], ['hobbes', 'hobbes', 'hobbes'])
        self.assertEqual(items[0]['candidates'], ['HH AA1 B IY0 Z', 'HH AA1 B Z'])
        self.assertEqual([item['vote'] for item in items], [1, 1, 1])

        # Resumed - the tokens before kept
        with captured() as (_, stderr), unittest.mock.patch.object(heteronyms, 'LABELS', labels), \
                unittest.mock.patch.object(heteronyms, 'AUDIO', audio), \
                unittest.mock.patch.object(cremad, 'load_ctc', lambda device: (None, vocab, device)), \
                unittest.mock.patch.object(cremad, 'ctc_log_posteriors', posteriors), \
                unittest.mock.patch.object(g2paudit, 'MIN_SPEAKERS', 2):
            g2paudit.command_audio(argparse.Namespace(out=self.out, batch=2, threads=1, log=None))
        self.assertIn('0 tokens to score (3 before)', stderr.getvalue())

        # The verdicts - more votes, from several speakers, for the error (the votes added here)
        votes = [{'word': 'hobbes', 'control': False, 'rel': os.path.join('s', str(speaker), 'c', f'{number}.lab'), 'start': 0.0,
                  'candidates': ['HH AA1 B IY0 Z', 'HH AA1 B Z'], 'scores': [-9.0, -1.0], 'vote': 1}
                 for speaker in range(3) for number in range(2)]
        votes += [{'word': 'hobbes', 'control': False, 'rel': os.path.join('s', '9', 'c', '0.lab'), 'start': 0.0,
                   'candidates': ['HH AA1 B IY0 Z', 'HH OW1 B Z', 'HH AA1 B Z'], 'scores': [-9.0, -5.0, -1.0], 'vote': 2}]
        votes += [{'word': 'are', 'control': False, 'rel': os.path.join('s', str(speaker), 'c', '0.lab'), 'start': 0.0,
                   'candidates': ['AA1 R', 'ER0'], 'scores': [-9.0, -1.0], 'vote': 1} for speaker in range(5)]
        votes += [{'word': 'cat', 'control': True, 'rel': os.path.join('s', str(speaker), 'c', '0.lab'), 'start': 0.0,
                   'candidates': ['K AE1 T', 'K AA1 T'], 'scores': [-1.0, -9.0], 'vote': vote}
                  for speaker, vote in [(number, 0) for number in range(8)] + [(8, 1), (9, None)]]
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
        self.assertEqual(verdicts['summary'], {'error': 1, 'excluded': 1, 'none': 3})
        self.assertEqual(next(item for item in verdicts['verdicts'] if item['word'] == 'are')['reason'], 'function word')
        self.assertEqual(verdicts['controls']['words'], 1)
        self.assertEqual(verdicts['controls']['falseErrors'], 0)
        self.assertEqual(verdicts['controls']['confirmedCorrect'], 1)
        self.assertEqual((verdicts['controls']['tokenVotes'], verdicts['controls']['tokenVotesForAlternative']), (9, 1))
        with open(os.path.join(self.out, 'fixes.json'), encoding='utf-8') as file:
            fixes = json.load(file)
        self.assertEqual(fixes['fixes'], {'hobbes': 'HH AA1 B Z', 'hobbs': 'HH AA1 B Z'})
        self.assertEqual(fixes['evidence']['hobbes']['audio']['votes'], [0, 10, 0])
        self.assertEqual(fixes['evidence']['hobbes']['audio']['candidates'], ['HH AA1 B IY0 Z', 'HH AA1 B Z', 'HH OW1 B Z'])
        self.assertEqual(fixes['evidence']['hobbes']['folds'], [['HH AA1 B Z', 8.0], ['HH AA1 B Z', 7.0]])
        self.assertEqual(fixes['evidence']['hobbs']['note'], 'hobbesian')
        self.assertIn('date', fixes['evidence']['hobbs'])
        with captured():
            g2paudit.command_adjudicate(argparse.Namespace(out=self.out, extra=None, log=None))
        with open(os.path.join(self.out, 'fixes.json'), encoding='utf-8') as file:
            self.assertEqual(list(json.load(file)['fixes']), ['hobbes'])

    def test_verdict(self):
        candidates = ['A', 'B', 'C']
        self.assertEqual(g2paudit.verdict(candidates, [(1, speaker) for speaker in 'abcd'])['verdict'], 'error')
        self.assertEqual(g2paudit.verdict(candidates, [(1, 'a')] * 3 + [(1, 'b')] * 3)['verdict'], 'inconclusive')
        result = g2paudit.verdict(candidates, [(0, speaker) for speaker in 'abcdefgh'] + [(2, 'i'), (None, 'j')])
        self.assertEqual((result['verdict'], result['best'], result['votes'], result['speakers']), ('correct', 2, [8, 0, 1], [8, 0, 1]))
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
