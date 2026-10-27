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

import numpy
import torch

from picovox_train import g2p, g2pmodel

from .helpers import captured


# The tiny dictionary - a few dozen words, some with variants, a comment, and words the G2P can't look up
DICTIONARY = '''
;;; a comment
cat K AE1 T
cats K AE1 T S
dog D AO1 G
dogs D AO1 G Z
read R IY1 D
read(2) R EH1 D
read(3) R IY1 D
lead L IY1 D
lead(2) L EH1 D
the DH AH0 # a function word
a AH0
an AE1 N
and AH0 N D
sat S AE1 T
mat M AE1 T
hat HH AE1 T
bat B AE1 T
rat R AE1 T
log L AO1 G
fog F AO1 G
bog B AO1 G
hog HH AO1 G
jog JH AA1 G
cog K AA1 G
tin T IH1 N
pin P IH1 N
win W IH1 N
bin B IH1 N
fin F IH1 N
sin S IH1 N
kin K IH1 N
'tis T IH1 Z
o'clock AH0 K L AA1 K
x-ray EH1 K S R EY2
Upper AH1 P ER0
'''

# The words file - unknown words' labels, the derived words' stems, and the book counts
WORDS = {
    'labels': {'catz': 'K AE1 T S', 'readin': 'R IY1 D IH0 N', 'leads': 'L IY1 D Z', 'lead\'d': 'L IY1 D', 'doggo': 'D AO1 G OW0',
               'cattish': 'K AE2 T IH0 SH', 'unknownwords': 'Y UW1 Z'},
    'stems': {'catz': ['cat', 'spelling'], 'readin': ['read', 'derived'], 'leads': ['lead', 'derived'],
              'lead\'d': ['lead', 'spelling'], 'doggo': ['dog', 'derived'], 'cattish': ['cat', 'derived'],
              'unknownwords': ['unknownword', 'derived']},
    'evalCounts': {'the': 100, 'cat': 20, 'read': 10, 'dog': 5, 'sat': 3},
    'weightCounts': {'a': 50, 'and': 40, 'lead': 4, 'mat': 2}
}

# The settings - a tiny model, two short stages (the second relabeled, with the boosts)
CONFIG = {
    'model': {'embed': 4, 'hidden': 4}, 'batch': 8, 'alpha': 0.5, 'uniform': 0.5, 'unknown': 0.3, 'threads': 1,
    'stages': [
        {'name': 'base', 'steps': 3, 'lr': 0.003, 'warmup': 1, 'seed': 1},
        {'name': 'focus', 'steps': 2, 'lr': 0.001, 'warmup': 1, 'seed': 2, 'topBoost': {'share': 0.3, 'count': 3},
         'relabel': 'RELABEL', 'focusBoost': {'share': 0.1}}
    ],
}


class TestG2p(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        directory = self.directory.name
        self.dictionary_path = os.path.join(directory, 'cmudict.dict')
        with open(self.dictionary_path, 'w', encoding='utf-8') as file:
            file.write(DICTIONARY)
        self.words_path = os.path.join(directory, 'g2p-words.json')
        with open(self.words_path, 'w', encoding='utf-8') as file:
            json.dump(WORDS, file)
        self.relabel_path = os.path.join(directory, 'g2p-relabel.json')
        with open(self.relabel_path, 'w', encoding='utf-8') as file:
            json.dump({'relabel': {'lead': 'L EH1 D', 'cat': 'K AA1 T', 'unknownword': 'Y UW1'}}, file)
        self.config_path = os.path.join(directory, 'train.json')
        config = json.loads(json.dumps(CONFIG).replace('RELABEL', self.relabel_path))
        with open(self.config_path, 'w', encoding='utf-8') as file:
            json.dump({'g2p': config}, file)
        self.patches = [unittest.mock.patch.object(g2p, 'WORDS_PATH', self.words_path),
                        unittest.mock.patch.object(g2p.read_pronunciations, '__defaults__', (self.dictionary_path,)),
                        unittest.mock.patch.object(g2p.load_data, '__defaults__', (None, None, None))]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.directory.cleanup()

    def test_log_line(self):
        path = os.path.join(self.directory.name, 'log.txt')
        with captured() as (_, stderr):
            g2p.log_line(path, 'one')
            g2p.log_line(None, 'two')
        self.assertTrue(stderr.getvalue().endswith(' two\n'))
        with open(path, encoding='utf-8') as file:
            self.assertTrue(file.read().endswith(' one\n'))

    def test_load_config(self):
        self.assertEqual(g2p.load_config(self.config_path)['batch'], 8)

    def test_read_pronunciations(self):
        pronunciations = g2p.read_pronunciations()
        self.assertEqual(pronunciations['read'], [['R', 'IY1', 'D'], ['R', 'EH1', 'D']])
        self.assertIn('o\'clock', pronunciations)
        self.assertNotIn('\'tis', pronunciations)
        self.assertNotIn('x-ray', pronunciations)
        self.assertNotIn('Upper', pronunciations)

    def test_split_words(self):
        words = [f'word{index}' for index in range(200)]
        split = g2p.split_words(words)
        self.assertEqual(len(split['test']), 10)
        self.assertEqual(len(split['validation']), 6)
        self.assertEqual(split, g2p.split_words(list(reversed(words))))

    def test_load_data(self):
        data = g2p.load_data()
        self.assertEqual(data['first']['read'], ['R', 'IY1', 'D'])
        self.assertEqual(data['relabeled'], set())
        self.assertEqual(data['labels']['leads'], ['L', 'IY1', 'D', 'Z'])
        self.assertTrue(data['monitor'])
        self.assertFalse(set(data['monitor']) & set(WORDS['evalCounts']))

        # Relabeled targets - and the rule-labeled words derived from them follow
        data = g2p.load_data(self.relabel_path)
        self.assertEqual(data['relabeled'], {'lead', 'cat'})
        self.assertEqual(data['first']['lead'], ['L', 'EH1', 'D'])
        self.assertEqual(data['labels']['leads'], ['L', 'EH1', 'D', 'Z'])
        self.assertEqual(data['labels']['lead\'d'], ['L', 'EH1', 'D'])
        self.assertEqual(data['labels']['catz'], ['K', 'AA1', 'T', 'S'])
        self.assertEqual(data['labels']['cattish'], ['K', 'AE2', 'T', 'IH0', 'SH'])
        self.assertEqual(data['labels']['unknownwords'], ['Y', 'UW1', 'Z'])
        self.assertEqual(data['labels']['doggo'], ['D', 'AO1', 'G', 'OW0'])
        self.assertNotIn('unknownword', data['first'])

    def test_load_data_fixes(self):
        self.assertEqual(g2p.FIXES_PATH, os.path.join(g2p.ROOT, 'data', 'nn', 'g2p-fixes.json'))
        fixes_path = os.path.join(self.directory.name, 'g2p-fixes.json')
        with open(fixes_path, 'w', encoding='utf-8') as file:
            json.dump({'fixes': {'cat': 'K AE1 T S', 'dog': 'D AA1 G', 'read': 'R EH1 D', 'lead': 'L IY1 D Z', 'zzz': 'Z'}}, file)
        data = g2p.load_data(fixes=fixes_path)

        # The corrected pronunciation first, the wrong one dropped
        self.assertEqual(data['first']['dog'], ['D', 'AA1', 'G'])
        self.assertEqual(data['all']['dog'], [['D', 'AA1', 'G']])
        self.assertEqual(data['all']['read'], [['R', 'EH1', 'D']])
        self.assertNotIn('zzz', data['first'])

        # The derived words follow - a spelling variant, and a derived form with its -s suffix chosen again
        self.assertEqual(data['labels']['catz'], ['K', 'AE1', 'T', 'S', 'IH0', 'Z'])
        self.assertEqual(data['labels']['doggo'], ['D', 'AA1', 'G', 'OW0'])
        self.assertEqual(data['labels']['leads'], ['L', 'IY1', 'D', 'Z', 'IH0', 'Z'])
        self.assertEqual(data['labels']['readin'], ['R', 'EH1', 'D', 'IH0', 'N'])

        # The relabeled targets after the fixes
        data = g2p.load_data(self.relabel_path, fixes_path)
        self.assertEqual(data['first']['lead'], ['L', 'EH1', 'D'])
        self.assertEqual(data['first']['dog'], ['D', 'AA1', 'G'])

    def test_load_data_books(self):
        self.assertEqual(g2p.BOOKS_PATH, os.path.join(g2p.ROOT, 'data', 'nn', 'g2p-books.json'))
        monitored = g2p.load_data()['monitor'][0]
        books_path = os.path.join(self.directory.name, 'g2p-books.json')
        with open(books_path, 'w', encoding='utf-8') as file:
            json.dump({'description': 'books', 'labels': {'cat': 'K AA1 T', 'doggo': 'D AA1 G OW2', 'zork': 'Z AO1 R K'},
                       'books': {'one': {'counts': {'zork': 7, 'zork\'s': 2, 'cat': 3, 'cat\'s': 1}},
                                 'two': {'counts': {'mat\'s': 1, 'dog\'s': 1, 'nope\'s': 1, monitored: 1}}}}, file)
        with open(self.dictionary_path, 'a', encoding='utf-8') as file:
            file.write('dog\'s D AO1 G Z\n')
        data = g2p.load_data(books=books_path)

        # The labels - a dictionary word's corrected (its derived words follow), and new words, a teacher's label replaced
        self.assertEqual(data['first']['cat'], ['K', 'AA1', 'T'])
        self.assertEqual(data['all']['cat'], [['K', 'AA1', 'T']])
        self.assertEqual(data['labels']['catz'], ['K', 'AA1', 'T', 'S'])
        self.assertEqual(data['first']['zork'], ['Z', 'AO1', 'R', 'K'])
        self.assertEqual(data['first']['doggo'], ['D', 'AA1', 'G', 'OW2'])
        self.assertNotIn('doggo', data['labels'])

        # The possessives the dictionary lacks - the stem's target with its -s suffix (a dictionary possessive kept)
        self.assertEqual(data['first']['zork\'s'], ['Z', 'AO1', 'R', 'K', 'S'])
        self.assertEqual(data['first']['cat\'s'], ['K', 'AA1', 'T', 'S'])
        self.assertEqual(data['first']['mat\'s'], ['M', 'AE1', 'T', 'S'])
        self.assertEqual(data['first']['dog\'s'], ['D', 'AO1', 'G', 'Z'])
        self.assertNotIn('nope\'s', data['first'])

        # The counts join the weights, and the books' words leave the monitoring set
        self.assertEqual(data['weightCounts']['zork'], 7)
        self.assertEqual(data['weightCounts']['mat'], 2)
        self.assertEqual(data['weightCounts']['mat\'s'], 1)
        self.assertNotIn(monitored, data['monitor'])

    def test_command_book(self):
        books_path = os.path.join(self.directory.name, 'g2p-books.json')
        with open(books_path, 'w', encoding='utf-8') as file:
            json.dump({'description': 'The books', 'labels': {'zork': 'Z AO1 R K', 'cattish': 'K AE1 T IH0 SH'},
                       'books': {'other': {'title': 'Other', 'url': 'u', 'tokens': 1, 'counts': {'a': 1}}}}, file)
        text_path = os.path.join(self.directory.name, 'book.txt')
        with open(text_path, 'w', encoding='utf-8') as file:
            file.write('The cat’s zork, the Zork’s dog. Señor Blorp sat.')
        with captured() as (_, stderr):
            argv = ['book', text_path, '--name', 'tale', '--title', 'A Tale', '--url', 'https://x', '--books', books_path]
            self.assertEqual(g2p.main(argv), 0)
        self.assertIn('book tale: 9 tokens, 6 of its 8 words kept', stderr.getvalue())
        with open(books_path, encoding='utf-8') as file:
            text = file.read()
        self.assertEqual(text.splitlines()[:5], ['{', ' "description": "The books",', ' "labels": {',
                                                 '  "cattish": "K AE1 T IH0 SH",', '  "zork": "Z AO1 R K"'])
        books = json.loads(text)
        self.assertEqual(books['books']['tale'], {'title': 'A Tale', 'url': 'https://x', 'tokens': 9, 'counts': {
            'cat\'s': 1, 'dog': 1, 'sat': 1, 'the': 2, 'zork': 1, 'zork\'s': 1}})
        self.assertEqual(list(books['books']), ['other', 'tale'])

    def test_s_suffix(self):
        self.assertEqual(g2p.s_suffix(['K', 'AE1', 'T']), ['S'])
        self.assertEqual(g2p.s_suffix(['B', 'AH1', 'S']), ['IH0', 'Z'])
        self.assertEqual(g2p.s_suffix(['D', 'AO1', 'G']), ['Z'])
        self.assertEqual(g2p.s_suffix([]), ['Z'])

    def test_training_items(self):
        data = g2p.load_data(self.relabel_path)
        settings = {'alpha': 0.5, 'uniform': 0.5, 'unknown': 0.0}
        items, probabilities = g2p.training_items(data, settings)
        self.assertEqual(len(items), len(data['first']) - len(data['monitor']))
        self.assertAlmostEqual(probabilities.sum(), 1.0)
        boosted = {**settings, 'unknown': 0.3, 'topBoost': {'share': 0.5, 'count': 2}, 'focusBoost': {'share': 0.2}}
        items_boosted, probabilities_boosted = g2p.training_items(data, boosted)
        self.assertEqual(len(items_boosted), len(items) + len(WORDS['labels']))
        self.assertAlmostEqual(probabilities_boosted.sum(), 1.0)
        index = [word for word, _ in items_boosted].index('the')
        self.assertGreater(probabilities_boosted[index], probabilities[index])

        # The focus boost needs relabeled words
        data['relabeled'] = set()
        _, unfocused = g2p.training_items(data, {**settings, 'focusBoost': {'share': 0.2}})
        self.assertTrue(numpy.array_equal(unfocused, probabilities))

    def test_model(self):
        self.assertEqual(g2p.model_config({'embed': '4', 'hidden': 5}), {'design': 'ar', 'embed': 4, 'hidden': 5, 'decoder': 10})
        self.assertEqual(g2p.model_config({'embed': 4, 'hidden': 5, 'decoder': 7})['decoder'], 7)
        torch.manual_seed(0)
        model = g2p.build_model({'embed': 4, 'hidden': 4})
        letters, targets = g2p.make_batch(['cat', '123', 'dogs'], [['K', 'AE1', 'T'], [], ['D', 'AO1', 'G', 'Z']])
        self.assertEqual(letters[1].tolist()[:1], [g2p.LETTER_IDS['\'']])
        self.assertEqual(targets[0].tolist(), [g2p.PHONE_IDS['K'], g2p.PHONE_IDS['AE1'], g2p.PHONE_IDS['T'], g2p.EOS, 0])
        loss = model.loss(letters, targets)
        self.assertGreater(loss.item(), 0)

        # Greedy decoding stops at <eos> or the length limit
        results = model.decode(letters)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(len(phones) <= 3 * 4 + 6 for phones in results))
        with torch.no_grad():
            model.output.bias.fill_(0)
            model.output.bias[g2p.PHONE_IDS['K']] = 100.0
        self.assertEqual([len(phones) for phones in model.decode(letters)], [3 * 3 + 6, 3 * 1 + 6, 3 * 4 + 6])
        with torch.no_grad():
            model.output.bias[g2p.EOS] = 1000.0
        self.assertEqual(model.decode(letters), [[], [], []])
        self.assertEqual(g2p.predict(model, ['cat', 'dog'], batch_size=1), [[], []])
        self.assertEqual(g2p.word_ids('ab-c' + 'd' * 40), [2, 3, 4] + [5] * 29)

    def test_learning_rate(self):
        self.assertAlmostEqual(g2p.learning_rate(0, 100, 1.0, 10), 0.1)
        self.assertAlmostEqual(g2p.learning_rate(10, 100, 1.0, 10), 1.0)
        self.assertAlmostEqual(g2p.learning_rate(100, 100, 1.0, 10), 0.02)
        self.assertAlmostEqual(g2p.learning_rate(500, 100, 1.0, 10), 0.02)

    def test_evaluate(self):
        data = g2p.load_data()
        result = g2p.evaluate(lambda words: [data['all'][word][-1] if word != 'dog' else ['D'] for word in words], data)
        self.assertEqual(result['bookWrongDistinct'], 2)
        self.assertAlmostEqual(result['book'], 15 / 138)
        self.assertAlmostEqual(result['bookAny'], 5 / 138)
        self.assertEqual(result['wrong'], [['read', 10, 'R EH1 D', 'R IY1 D'], ['dog', 5, 'D', 'D AO1 G']])
        self.assertEqual(result['top10kWrong'], 2)
        self.assertEqual(g2p.describe(result), f'book err {100 * 15 / 138:.3f}% (any {100 * 5 / 138:.3f}%, 2 distinct), '
                                               f'top-10K wrong 2, monitor WER {100 * result["monitorWer"]:.2f}%')

    def test_quantize(self):
        codes, values = g2p.bfloat16([1.0, 1.00390625, 1.01171875, -2.5, 0.0])
        self.assertEqual(codes.tolist(), [0x3F80, 0x3F80, 0x3F82, 0xC020, 0])
        self.assertEqual(values.tolist(), [1.0, 1.0, 1.015625, -2.5, 0.0])
        rng = numpy.random.default_rng(0)
        values = rng.normal(size=(3, 10))
        values[2] = 0.0
        integers, codes, dequantized = g2p.quantize_array(values, 6)
        self.assertEqual(integers.shape, (3, 10))
        self.assertEqual(len(codes), 3)
        self.assertTrue(numpy.abs(integers).max() <= 31)
        self.assertTrue((dequantized[2] == 0).all())
        self.assertLess(numpy.abs(dequantized - values).max(), 0.2)

    def test_train_evaluate(self):
        out = os.path.join(self.directory.name, 'g2p')
        with captured():
            self.assertEqual(g2p.main(['train', '--out', out, '--config', self.config_path]), 0)
        with open(os.path.join(out, 'done.json'), encoding='utf-8') as file:
            done = json.load(file)
        self.assertEqual(done['checkpoint'], os.path.join(out, 'focus.pt'))
        self.assertEqual(sorted(done['results']), ['base', 'focus'])
        self.assertEqual(set(done['results']['base']), {'book', 'bookAny', 'bookWrongDistinct', 'top10kWrong', 'monitorWer', 'wrong'})
        with open(os.path.join(out, 'train.log'), encoding='utf-8') as file:
            log = file.read()
        self.assertIn('base: step 3/3 loss', log)
        self.assertIn(f'2 steps from {os.path.join(out, "base.pt")}', log)

        # Resumed - the stages done before are kept; --steps for each stage
        os.remove(os.path.join(out, 'focus.pt'))
        with captured():
            self.assertEqual(g2p.main(['train', '--out', out, '--config', self.config_path, '--steps', '1']), 0)
        with open(os.path.join(out, 'train.log'), encoding='utf-8') as file:
            log = file.read()
        self.assertIn(f'base: done before ({os.path.join(out, "base.pt")})', log)
        self.assertIn(f'1 steps from {os.path.join(out, "base.pt")}', log)

        # The measures of a module (the committed one) and of a checkpoint
        module_path = g2pmodel.MODULE_PATH
        result_path = os.path.join(self.directory.name, 'result.json')
        with captured() as (_, stderr):
            self.assertEqual(g2p.main(['evaluate', module_path, '--out', result_path]), 0)
        self.assertIn(f'evaluate {module_path}: book err ', stderr.getvalue())
        with open(result_path, encoding='utf-8') as file:
            self.assertIn('monitorWer', json.load(file))
        with captured() as (_, stderr):
            self.assertEqual(g2p.main(['evaluate', os.path.join(out, 'focus.pt'), '--relabel', self.relabel_path, '--threads', '1']), 0)
        self.assertIn('focus.pt: book err ', stderr.getvalue())

    def test_main_module(self):
        with unittest.mock.patch.object(sys, 'argv', ['g2p']), captured() as (_, stderr), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.g2p', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('the following arguments are required', stderr.getvalue())
