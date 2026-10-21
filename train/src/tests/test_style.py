# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import io
import json
import math
import os
import runpy
import shutil
import tempfile
import unittest
import unittest.mock
import warnings

from picovox_train import style

from .helpers import captured


# Two tiny books - shouted, frightened, and plain dialogue, with attributions and context
BOOKS = {
    'pg1': ('The fire spread. “Help! Fire!” he screamed. “Get out! Get out of the house!”\n\n'
            '“No,” she said calmly, “I will stay here.”\n\n'
            '“Run! They are coming!” she cried in terror.\n\nThe house burned. “It is over,” he said.\n'),
    'pg2': ('“Charge!” roared the captain. “Forward, men!”\n\n'
            '“What a fine day,” said the farmer. “Indeed it is,” said his wife.\n\n'
            '“Help me! Please!” she gasped, afraid. “The wolves!”\n\n“Good night,” he said.\n')
}

# The tiny settings - data/nn/train.json's "style" shape
SETTINGS = {
    'teacher': 'tiny', 'enrichedCount': 3, 'exclaimedCount': 3, 'plainCount': 3, 'minChars': 1, 'maxChars': 600, 'seed': 1,
    'network': {'buckets': 64, 'dim': 4, 'hidden': 8}, 'pretrainEpochs': 2, 'pretrainBatch': 4, 'pretrainTop': 10, 'epochs': 2,
    'batch': 4, 'lr': 0.01, 'weightDecay': 0.0, 'dropout': 0.0, 'weightPower': 0, 'folds': 2, 'threads': 1, 'precision': 0.5
}


# An Ollama /api/chat that answers each quotation by its words - shouting for a scream or roar, fearful for terror,
# neutral otherwise - with its answer's logprobs (none for a "No"), recording the requests
class FakeTeacher:

    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout):
        assert timeout == 600
        body = json.loads(request.data)
        self.requests.append(body)
        user = body['messages'][1]['content']
        label = 'shouting' if 'scream' in user or 'roar' in user else ('fearful' if 'terror' in user or 'gasp' in user else 'neutral')
        content = json.dumps({'label': label})
        tokens = ['{"', 'label', '":"', label[:3], label[3:], '"}']
        alternatives = [{'bytes': list(label[:3].encode()), 'logprob': math.log(0.8)},
                        {'bytes': list(b'neu' if label != 'neutral' else b'ang'), 'logprob': math.log(0.2)}]
        logprobs = [{'bytes': list(token.encode()), 'top_logprobs': alternatives} for token in tokens]
        answer = {'message': {'content': content}, 'logprobs': None if 'it):  “No,' in user else logprobs}
        return io.BytesIO(json.dumps(answer).encode())


class StyleCase(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.books = os.path.join(self.directory, 'books')
        os.makedirs(self.books)
        for name, text in BOOKS.items():
            with open(os.path.join(self.books, f'{name}.txt'), 'w', encoding='utf-8') as file:
                file.write(text)
        self.config = os.path.join(self.directory, 'train.json')
        with open(self.config, 'w', encoding='utf-8') as file:
            json.dump({'context': {'labels': {'texts': sorted(BOOKS)}}, 'style': SETTINGS}, file)
        self.test_path = os.path.join(self.directory, 'style-test.json')
        self.labels = os.path.join(self.directory, 'style-labels.json')

    def tearDown(self):
        shutil.rmtree(self.directory)

    def path(self, name):
        return os.path.join(self.directory, name)

    # The books' quotations (nn/js/quoteLines.js), written to <directory>/quotations.json
    def quotations(self):
        with captured():
            style.main(['quotations', '--out', self.directory, '--config', self.config])
        with open(self.path('quotations.json'), encoding='utf-8') as file:
            return json.load(file)

    # The hand-labeled test file - the given quotations ([(book, quotation, label, stratum)])
    def write_test(self, items):
        with open(self.test_path, 'w', encoding='utf-8') as file:
            json.dump({'quotations': [{'id': f'{book}:{quotation["quotation"]}', 'label': label, 'stratum': stratum,
                                       **{key: quotation[key] for key in ('text', 'before', 'after', 'previous', 'next')}}
                                      for book, quotation, label, stratum in items]}, file)

    # Label the books' sample with the fake teacher
    def label(self, *extra):
        teacher = FakeTeacher()
        with unittest.mock.patch('urllib.request.urlopen', teacher), captured() as (_, stderr):
            style.main(['label', '--out', self.directory, '--config', self.config, '--labels', self.labels, '--test', self.test_path,
                        '--log', self.path('label.log'), *extra])
        return teacher, stderr.getvalue()

    def setup_quotations(self):
        original = style.book_quotations
        patched = unittest.mock.patch.object(style, 'book_quotations', lambda names: original(names, self.books))
        patched.start()
        self.addCleanup(patched.stop)
        return self.quotations()



class TestFeatures(unittest.TestCase):

    def test_style_words(self):
        # lib/style.js styleWords's
        self.assertEqual(style.style_words('Señor, it’s DONE!'), ['senor', 'it\'s', 'done'])
        self.assertEqual(style.style_words('... --'), [])

    def test_fnv1a(self):
        # lib/style.js fnv1a's - of UTF-16 code units
        self.assertEqual(style.fnv1a(''), 2166136261)
        self.assertEqual(style.fnv1a('a'), 3826002220)
        self.assertEqual(style.fnv1a('hello'), 0x4f9f2cab)
        self.assertEqual(style.word_buckets('Hello, a!', 4096), [0x4f9f2cab % 4096, 3826002220 % 4096])

    def test_style_cues(self):
        # lib/style.js styleCues's
        self.assertEqual(style.style_cues({'text': '“STOP! Stop—now!”', 'before': 'he cried!'}),
                         [2 / 3, 0.0, 5 / 11, 0.0, math.log1p(3) / 4, 1.0, 1.0, 1.0])
        self.assertEqual(style.style_cues({'text': '"Why? _Why?_ Tell me why, why, why!!!!"', 'after': 'she said.'}),
                         [1.0, 1.0, 3 / 21, 1.0, math.log1p(7) / 4, 0.0, 0.0, 0.0])
        self.assertEqual(style.style_cues({'text': '...'}), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_kind_texts(self):
        self.assertEqual(style.kind_texts({'text': 'Go!'}), ['Go!', ' ', ' '])
        self.assertEqual(style.kind_texts({'text': 'Go!', 'before': 'a', 'after': 'b', 'previous': 'c', 'next': 'd'}),
                         ['Go!', 'a b', 'c d'])

    def test_digest(self):
        self.assertEqual(style.digest('Go!'), 'e95c3961')


class TestSampling(unittest.TestCase):

    def test_partition(self):
        quotation = {'text': 'Run!', 'before': '', 'after': '', 'previous': '', 'next': ''}
        self.assertEqual(style.partition(quotation), 'exclaimed')
        self.assertEqual(style.partition({**quotation, 'after': 'she screamed.'}), 'enriched')
        self.assertEqual(style.partition({**quotation, 'text': 'Run.', 'after': 'she screamed.'}), 'plain')
        self.assertFalse(style.is_enriched({**quotation, 'text': 'Run.', 'next': 'In terror.'}))

    def test_sample_quotations(self):
        # Each part's count of its quotations (the length limits and the excluded ones left out), the parts' sizes
        quotations = {'b': [{'quotation': index, 'text': text, 'before': '', 'after': after, 'previous': '', 'next': ''}
                            for index, (text, after) in enumerate([('Run!', 'he shouted.'), ('Hi!', ''), ('Hey!', ''), ('Hello.', ''),
                                                                   ('A very long quotation.', ''), ('Fine.', '')])]}
        settings = {**SETTINGS, 'maxChars': 10, 'exclaimedCount': 1}
        sample, sizes = style.sample_quotations(quotations, settings, {'b:5'})
        self.assertEqual(sizes, {'enriched': 1, 'exclaimed': 2, 'plain': 1})
        self.assertEqual([(book, quotation['quotation'], part) for book, quotation, part in sample][0], ('b', 0, 'enriched'))
        self.assertEqual([part for _, _, part in sample], ['enriched', 'exclaimed', 'plain'])
        self.assertEqual(sample, style.sample_quotations(quotations, settings, {'b:5'})[0])


class TestTeacher(unittest.TestCase):

    def test_teacher_label(self):
        # The label and its probabilities - the alternatives at the label's first token, each a style's first letters
        teacher = FakeTeacher()
        quotation = {'text': '“Charge!”', 'before': '', 'after': 'roared the captain.', 'previous': 'Dawn.', 'next': ''}
        with unittest.mock.patch('urllib.request.urlopen', teacher):
            self.assertEqual(style.teacher_label(quotation, 'tiny', 'http://teacher'), ('shouting', [0.2, 0.0, 0.8, 0.0]))
        body = teacher.requests[0]
        self.assertEqual((body['model'], body['think'], body['logprobs']), ('tiny', False, True))
        self.assertEqual(body['format']['properties']['label']['enum'], style.CLASSES)
        self.assertIn('roared the captain.', body['messages'][1]['content'])

        # No logprobs - no probabilities
        with unittest.mock.patch('urllib.request.urlopen', teacher):
            self.assertEqual(style.teacher_label({**quotation, 'text': '“No, sir!”'}, 'tiny')[1], None)

    def test_label_probabilities(self):
        content = '{"label":"fearful"}'
        def token(text, alternatives):
            return {'bytes': list(text.encode()), 'top_logprobs': [
                {'bytes': list(alt.encode()), 'logprob': logprob} for alt, logprob in alternatives]}
        # A space or quote before the letters, an improbable alternative, and an empty one are skipped
        logprobs = [token('{"label":"', []), token('fear', [(' fear', math.log(0.6)), ('"shout', math.log(0.3)), ('x', -1000.0),
                                                            ('"', math.log(0.1))]), token('ful"}', [])]
        self.assertEqual(style.label_probabilities(content, 'fearful', logprobs), [0.0, 0.0, 0.3333, 0.6667])
        # No style among the alternatives, and the answer's tokens ending before the label
        self.assertIsNone(style.label_probabilities(content, 'fearful', [token('{"label":"', []), token('fear', [('zz', 0.0)])]))
        self.assertIsNone(style.label_probabilities(content, 'fearful', [token('{"', [])]))


class TestLabels(StyleCase):

    def test_read_write_labels(self):
        self.assertIsNone(style.read_labels(self.labels))
        labels = {'description': 'd', 'teacher': 't',
                  'labels': {'pg2': [[0, 'neutral', 'plain', 'abc']], 'pg1': [[1, 'angry', 'enriched', 'def']]}}
        style.write_labels(self.labels, labels)
        self.assertEqual(style.read_labels(self.labels), labels)
        with open(self.labels, encoding='utf-8') as file:
            lines = file.read().splitlines()
        self.assertEqual(lines[4:6], ['  "pg1": [[1, "angry", "enriched", "def"]],', '  "pg2": [[0, "neutral", "plain", "abc"]]'])

    def test_quotations(self):
        quotations = self.setup_quotations()
        self.assertEqual(sorted(quotations), ['pg1', 'pg2'])
        self.assertEqual(quotations['pg1'][0]['text'], '“Help! Fire!”')
        with open(self.path('quotations.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file), quotations)

    def test_label(self):
        # Every sampled quotation labeled, the hand-labeled ones left out, each with its digest and probabilities (if any)
        quotations = self.setup_quotations()
        self.write_test([('pg2', quotations['pg2'][0], 'shouting', 'enriched')])
        teacher, log = self.label()
        self.assertIn('label: the parts', log)
        labels = style.read_labels(self.labels)
        rows = {(book, row[0]): row for book, items in labels['labels'].items() for row in items}
        self.assertNotIn(('pg2', 0), rows)
        self.assertEqual(len(rows), len(teacher.requests))
        scream = rows[('pg1', 0)]
        self.assertEqual(scream[1:4], ['shouting', 'enriched', style.digest(quotations['pg1'][0]['text'])])
        self.assertEqual(scream[4], [0.2, 0.0, 0.8, 0.0])
        self.assertEqual([len(row) for row in rows.values()].count(4), 1)
        with open(self.path('label.log'), encoding='utf-8') as file:
            self.assertIn(f'label: {len(rows)}/{len(rows)}', file.read())

        # Resumed - nothing left to label
        teacher, _ = self.label()
        self.assertEqual(teacher.requests, [])

        # A relabel of a quotation already labeled replaces its row
        labels['labels']['pg1'] = [row for row in labels['labels']['pg1'] if row[0] != 0]
        style.write_labels(self.labels, labels)
        teacher, _ = self.label('--jobs', '2')
        self.assertEqual(len(teacher.requests), 1)
        self.assertEqual(style.read_labels(self.labels)['labels']['pg1'][0][0], 0)

    def test_labeled_examples(self):
        quotations = {'b': [{'quotation': 0, 'text': 'Go!'}, {'quotation': 1, 'text': 'Stay.'}]}
        labels = {'labels': {'b': [[0, 'angry', 'exclaimed', style.digest('Go!'), [0.1, 0.7, 0.1, 0.1]],
                                   [1, 'neutral', 'plain', style.digest('Stay.')]]}}
        examples = style.labeled_examples(quotations, labels)
        self.assertEqual([(quotation['quotation'], label, target) for quotation, label, target in examples],
                         [(0, 1, [0.1, 0.7, 0.1, 0.1]), (1, 0, [1.0, 0.0, 0.0, 0.0])])

        # A changed text, or a quotation gone - relabel
        for rows in ([[0, 'angry', 'exclaimed', 'changed']], [[2, 'angry', 'exclaimed', 'abc']]):
            with self.assertRaises(ValueError) as raised:
                style.labeled_examples(quotations, {'labels': {'b': rows}})
            self.assertIn('its text changed - relabel', str(raised.exception))

    def test_sample_weights(self):
        # The inverse of each part's labeled share, normalized to a mean of one; power tempers it
        quotations = {'b': [{'quotation': index, 'text': text, 'before': '', 'after': '', 'previous': '', 'next': ''}
                            for index, text in enumerate(['A!', 'B!', 'C.', 'D.', 'E.', 'F.'])]}
        examples = [(quotations['b'][0], 0, None), (quotations['b'][2], 0, None)]
        self.assertEqual(style.sample_weights(examples, quotations, SETTINGS), [2 / 3, 4 / 3])
        self.assertEqual(style.sample_weights(examples, quotations, SETTINGS, 0), [1.0, 1.0])


class TestThresholds(unittest.TestCase):

    def test_precision_bound(self):
        # The Wilson lower bound - two of two isn't evidence; fifty of fifty nearly 95%; weights shrink the count
        self.assertAlmostEqual(style.precision_bound(2, 2, 2), 0.425, places=3)
        self.assertAlmostEqual(style.precision_bound(50, 50, 50), 0.949, places=3)
        self.assertLess(style.precision_bound(50, 50, 50 * 50), style.precision_bound(50, 50, 50))
        self.assertAlmostEqual(style.precision_bound(0, 10, 10), 0.0)

    def test_choose_thresholds(self):
        # The least probability whose picks' bound reaches the target - a tie's picks together; a style that never
        # does, 1.01
        probabilities = [[0.1, 0.9, 0.0, 0.0]] * 40 + [[0.2, 0.8, 0.0, 0.0]] * 20 + [[0.3, 0.7, 0.0, 0.0]] * 20
        labels = [1] * 40 + [1] * 19 + [0] + [0] * 20
        thresholds = style.choose_thresholds(probabilities, labels, [1.0] * 80, 0.85)
        self.assertEqual(thresholds, {'angry': 0.8, 'shouting': 1.01, 'fearful': 1.01})
        self.assertEqual(style.choose_thresholds(probabilities, labels, [1.0] * 80, 0.99)['angry'], 1.01)

    def test_choose_styles(self):
        # Each quotation's most probable style at its threshold, else neutral
        thresholds = {'angry': 0.5, 'shouting': 0.3, 'fearful': 0.3}
        rows = [[0.1, 0.6, 0.3, 0.0], [0.6, 0.1, 0.2, 0.1], [0.2, 0.0, 0.35, 0.45], [0.5, 0.45, 0.05, 0.0]]
        self.assertEqual(style.choose_styles(rows, thresholds), [1, 0, 3, 0])

    def test_measures(self):
        result = style.measures([1, 0, 2, 3, 0], [1, 0, 3, 0, 2])
        self.assertEqual(result, {
            'accuracy': 0.4, 'styled': 3, 'precision': 1 / 3, 'styledPrecision': 2 / 3, 'recall': 1 / 3, 'neutralStyled': 1,
            'confusion': {'angry>angry': 1, 'fearful>shouting': 1, 'neutral>fearful': 1, 'neutral>neutral': 1, 'shouting>neutral': 1}
        })
        self.assertEqual(style.measures([], [])['accuracy'], 0)
        self.assertEqual(style.describe(result), 'accuracy 0.400, 3 styled - precision 0.333 (styled rightly 0.667), recall 0.333, '
                                                 '1 neutral styled')


class TestNetwork(StyleCase):

    def test_train_export_evaluate(self):
        # Label, train (pretrained, the word table fixed, each fold held out), export, and evaluate - the package's
        # inference (lib/style.js under Node) matches the network's, to the int8 weights' error
        quotations = self.setup_quotations()
        self.write_test([('pg2', quotations['pg2'][0], 'shouting', 'enriched'), ('pg2', quotations['pg2'][2], 'neutral', 'random')])
        self.label()
        with captured() as (_, stderr):
            style.main(['train', '--out', self.directory, '--config', self.config, '--labels', self.labels, '--test', self.test_path,
                        '--steps', '3'])
        self.assertIn('pretraining step', stderr.getvalue())
        self.assertIn('examples in 2 folds - thresholds', stderr.getvalue())
        import torch  # pylint: disable=import-outside-toplevel
        checkpoint = torch.load(self.path('style.pt'), weights_only=False)
        self.assertEqual(sorted(checkpoint['state']), ['embedding.weight', 'hidden.bias', 'hidden.weight', 'output.bias', 'output.weight'])
        self.assertEqual(sorted(checkpoint['thresholds']), ['angry', 'fearful', 'shouting'])
        with open(self.path('heldout.json'), encoding='utf-8') as file:
            heldout = json.load(file)
        self.assertEqual(len(heldout), len([row for rows in style.read_labels(self.labels)['labels'].values() for row in rows]))
        self.assertTrue(all(len(row['probabilities']) == 4 and row['part'] in ('enriched', 'exclaimed', 'plain') for row in heldout))

        # The export - the module, line by line
        module = self.path('styleModel.js')
        with captured():
            style.main(['export', self.path('style.pt'), '--out', module])
        with open(module, encoding='utf-8') as file:
            text = file.read()
        self.assertTrue(text.startswith('// Generated by picovox_train/style.py export - do not edit\n'))
        self.assertIn("    'version': 2,\n    'config': {'buckets': 64, 'dim': 4, 'hidden': 8},", text)
        self.assertTrue(all(len(line) <= style.MODULE_LINE_LENGTH for line in text.splitlines()[5:]))

        # Parity - the module's probabilities (lib/style.js) and the network's
        model = style.build_model(checkpoint['config'])
        model.load_state_dict(checkpoint['state'])
        items = [quotation for rows in quotations.values() for quotation in rows]
        expected = style.predict(model, [style.quotation_features(item, 64) for item in items], batch_size=3)
        for actual, wanted in zip(style.module_probabilities(module, items), expected):
            for value, other in zip(actual, wanted):
                self.assertAlmostEqual(value, other, delta=0.02)

        # The evaluation - all, and the random stratum
        with captured() as (_, stderr):
            style.main(['evaluate', module, '--test', self.test_path])
        self.assertIn('evaluate all (2): accuracy', stderr.getvalue())
        self.assertIn('evaluate random (1): accuracy', stderr.getvalue())

    def test_batch_tensors(self):
        # The kinds' bags in order, the attribution's (and its cue) hidden for the pretraining
        features = [([[1, 2], [3], [4, 5]], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), ([[], [6], []], [1.0] * 8)]
        indexes, offsets, cues = style.batch_tensors(features)
        self.assertEqual((indexes.tolist(), offsets.tolist()), ([1, 2, 3, 4, 5, 6], [0, 2, 3, 5, 5, 6]))
        self.assertEqual(cues[0, 6].item(), 1.0)
        indexes, offsets, cues = style.batch_tensors(features, True)
        self.assertEqual((indexes.tolist(), offsets.tolist()), ([1, 2, 4, 5], [0, 2, 2, 4, 4, 4]))
        self.assertEqual((cues[0, 6].item(), cues[1, 6].item(), cues[1, 5].item()), (0.0, 0.0, 1.0))

    def test_attribution_examples(self):
        # The attributed quotations, each its attribution's most common words' indexes
        quotations = [{'before': '', 'after': 'he shouted.'}, {'before': 'She', 'after': 'shouted'}, {'before': '', 'after': ''}]
        examples = style.attribution_examples(quotations, ['f0', 'f1', 'f2'], 2)
        self.assertEqual([feature for feature, _ in examples], ['f0', 'f1'])
        self.assertEqual(sorted(len(words) for _, words in examples), [1, 2])

    def test_pretrain(self):
        # The attribution's words from the rest - logged every tenth of its steps
        model = style.build_model(SETTINGS['network'], heads=3)
        features = [([[1, 2], [3], [4]], [0.0] * 8), ([[5], [6, 7], []], [1.0] * 8)]
        examples = [(features[0], [0, 2]), (features[1], [1])]
        with captured() as (_, stderr):
            style.pretrain(model, examples, {**SETTINGS, 'pretrainEpochs': 40, 'pretrainBatch': 4})
        self.assertEqual(stderr.getvalue().count('pretraining step'), 10)
        self.assertIn('pretraining step 20/20 loss', stderr.getvalue())

    def test_fold_of(self):
        self.assertEqual([style.fold_of('pg1', index, 5) for index in range(6)], [style.fold_of('pg1', index, 5) for index in range(6)])
        self.assertTrue(all(0 <= style.fold_of('pg2', index, 3) < 3 for index in range(20)))

    def test_quantize(self):
        data, scale = style.quantize([0.5, -1.0, 0.25])
        self.assertEqual((list(data), scale), ([64, 129, 32], 1.0 / 127))
        self.assertEqual(style.quantize([0.0, 0.0]), (bytes(2), 1.0))


class TestMain(StyleCase):

    def test_main(self):
        # The command line - python -m picovox_train.style, a command required
        with unittest.mock.patch('sys.argv', ['style']), captured() as (_, stderr), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.style', run_name='__main__')
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('the following arguments are required', stderr.getvalue())

    def test_log_line(self):
        with captured() as (_, stderr):
            style.log_line(None, 'hello')
            style.log_line(self.path('log.txt'), 'there')
        self.assertTrue(stderr.getvalue().endswith(' there\n'))
        with open(self.path('log.txt'), encoding='utf-8') as file:
            self.assertTrue(file.read().endswith(' there\n'))
        self.assertEqual(style.load_config(self.config), SETTINGS)
