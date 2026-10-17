# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import numpy
import torch

from picovox_train import ROOT, context, g2p, g2pmodel

from .helpers import captured


# The tiny context network's and reading stage's settings
CONTEXT_TINY = {'steps': 3, 'quantSteps': 2, 'batch': 8, 'embed': 4, 'hidden': 4, 'layer': 4, 'threads': 1, 'warmup': 1}
G2P_TINY = {'steps': 3, 'batch': 8, 'threads': 1, 'warmup': 1, 'readingShare': 0.3, 'invariantShare': 0.3}


# A tiny G2P model (g2p.py's), seeded
def tiny_g2p(seed=0):
    torch.manual_seed(seed)
    return g2p.build_model({'embed': 4, 'hidden': 4})


# A context window around a word
def window(*items):
    padded = list(items) + [''] * (2 * context.WINDOW + 1 - len(items))
    return padded


# Tiny labels (contextlabels.py's format) - read, lives, and the S/Z-only use; aligned examples over many speakers (both
# folds' sides), silver and seed examples, train and held out
def tiny_labels():
    words = {
        'read': {'readings': [['R IY1 D'], ['R EH1 D']], 'order': [0, 1], 'family': 'own', 'counts': [6, 6]},
        'lives': {'readings': [['L IH1 V Z'], ['L AY1 V Z']], 'order': [0, 1], 'family': 'own', 'counts': [4, 4]},
        'use': {'readings': [['Y UW1 Z'], ['Y UW1 S']], 'order': [0, 1], 'family': 'consonant', 'counts': [4, 4]}
    }
    examples = []
    for number in range(24):
        word = ('read', 'lives', 'use')[number % 3]
        slot = (number // 3) % 2
        before = 'will' if slot == 0 else 'had'
        examples.append({'window': window('', 'i', before, word, 'it', '.', ''), 'word': word, 'slot': slot,
                         'source': 'aligned', 'split': 'heldout' if number % 4 == 0 else 'train', 'group': f'speaker-{number}'})
    examples.append({'window': window('', '', 'the', 'read', '.', '', ''), 'word': 'read', 'slot': 1, 'source': 'silver',
                     'split': 'train', 'group': 'pg1'})
    examples.append({'window': window('', '', 'a', 'read', '.', '', ''), 'word': 'read', 'slot': 1, 'source': 'silver',
                     'split': 'heldout', 'group': 'pg2'})
    examples.append({'window': window('', 'she', 'will', 'read', '.', '', ''), 'word': 'read', 'slot': 0, 'source': 'seed',
                     'split': 'train', 'group': 'seed'})
    return {'words': words, 'examples': examples}


class TestContext(unittest.TestCase):

    def setUp(self):
        self.threads = torch.get_num_threads()
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        torch.set_num_threads(self.threads)
        shutil.rmtree(self.directory)

    def path(self, name):
        return os.path.join(self.directory, name)

    def write_labels(self, labels=None):
        path = self.path('labels.json')
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(labels or tiny_labels(), file)
        return path

    def test_settings(self):
        settings = context.load_settings()
        self.assertEqual(settings['g2p']['readings'], context.READINGS)
        self.assertNotIn('window', context.CONTEXT_SETTINGS)
        self.assertNotIn('readings', context.G2P_SETTINGS)
        self.assertEqual(context.CONTEXT_CHAR_IDS['\''], 1)

    def test_log_line(self):
        log_path = self.path('train.log')
        with captured() as (_, stderr):
            context.log_line(None, 'first')
            context.log_line(log_path, 'second')
        self.assertIn('first', stderr.getvalue())
        with open(log_path, encoding='utf-8') as file:
            self.assertTrue(file.read().endswith(' second\n'))

    def test_add_reading_heads_and_load(self):
        model = context.add_reading_heads(tiny_g2p())
        self.assertEqual(model.config['readings'], context.READINGS)
        self.assertEqual(tuple(model.reading.weight.shape), (context.READINGS - 1, 8))
        self.assertEqual(float(model.ambiguity.bias.detach()), -4.0)

        # A checkpoint with the heads, and one without
        torch.save({'config': model.config, 'state': model.state_dict()}, self.path('heads.pt'))
        loaded = context.load_g2p(self.path('heads.pt'))
        self.assertTrue(hasattr(loaded, 'reading'))
        self.assertTrue(torch.equal(loaded.reading.weight, model.reading.weight))
        plain = tiny_g2p()
        torch.save({'config': plain.config, 'state': plain.state_dict()}, self.path('plain.pt'))
        self.assertFalse(hasattr(context.load_g2p(self.path('plain.pt')), 'reading'))

    def test_reading_loss(self):
        model = context.add_reading_heads(tiny_g2p())
        letters, targets = g2p.make_batch(['read', 'cat'], [['R', 'EH1', 'D'], ['K', 'AE1', 'T']])
        readings = torch.tensor([1, 0])
        loss, phone_loss, ambiguity_loss = context.reading_loss(model, letters, targets, readings, torch.tensor([1.0, 0.0]), 0.5)
        self.assertAlmostEqual(loss.item(), phone_loss.item() + 0.5 * ambiguity_loss.item(), places=5)

        # The reading input changes the decoder's start only for an alternate reading
        _, _, state_default, _ = context.encode_reading(model, letters, torch.tensor([0, 0]))
        _, _, state_alternate, _ = context.encode_reading(model, letters, torch.tensor([2, 0]))
        self.assertFalse(torch.equal(state_default[0, 0], state_alternate[0, 0]))
        self.assertTrue(torch.equal(state_default[0, 1], state_alternate[0, 1]))

    def test_decode_readings(self):
        model = context.add_reading_heads(tiny_g2p())
        with torch.no_grad():
            # Every step's best phone <eos> - every word done at once
            model.output.bias.zero_()
            model.output.bias[g2p.EOS] = 100.0
            self.assertEqual(context.decode_readings(model, ['a', 'read'], [0, 1]), [[], []])

            # Never <eos> - each word ends at its limit (3 phones a letter plus 6), the shorter first
            model.output.bias[g2p.EOS] = -100.0
            model.output.bias[g2p.PHONE_IDS['K']] = 100.0
            phones = context.decode_readings(model, ['a', 'abc'], [0, 0], batch_size=1)
        self.assertEqual(phones, [['K'] * 9, ['K'] * 15])
        phones = context.decode_readings(model, ['a', 'abc'], [0, 0])
        self.assertEqual(phones, [['K'] * 9, ['K'] * 15])

    def test_ambiguity_logits(self):
        model = context.add_reading_heads(tiny_g2p())
        logits = context.ambiguity_logits(model, ['read', 'cat', 'lives'], batch_size=2)
        self.assertEqual(len(logits), 3)
        self.assertTrue(all(logit < 0 for logit in logits))

    def test_reading_targets(self):
        labels = tiny_labels()
        alternates = context.reading_targets(labels)
        self.assertIn(('read', ['R', 'EH1', 'D'], 1), alternates)
        self.assertFalse(any(reading == 0 for _, _, reading in alternates))
        with_defaults = context.reading_targets(labels, {'read': ['R', 'IY2', 'D']})
        self.assertIn(('read', ['R', 'IY2', 'D'], 0), with_defaults)
        self.assertIn(('lives', ['L', 'IH1', 'V', 'Z'], 0), with_defaults)

    def tiny_items(self):
        items = [('read', ['R', 'IY1', 'D']), ('lives', ['L', 'IH1', 'V', 'Z']), ('cat', ['K', 'AE1', 'T']), ('dog', ['D', 'AO1', 'G'])]
        return items, numpy.array([0.25, 0.25, 0.25, 0.25])

    def test_train_g2p(self):
        items, probabilities = self.tiny_items()
        lines = []
        model = context.train_g2p(tiny_g2p(), items, probabilities, tiny_labels(), dict(G2P_TINY, train='all'), lines.append)
        self.assertTrue(hasattr(model, 'reading'))
        self.assertFalse(model.training)
        self.assertIn('parameters trained (all)', lines[0])
        self.assertIn('train-g2p: step 3/3', lines[-1])

        # Only some tensors trained (the recipe's), from a model with the heads
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        context.train_g2p(model, items, probabilities, tiny_labels(), G2P_TINY, lines.append)
        self.assertTrue(torch.equal(before['encoder.weight_ih_l0'], model.state_dict()['encoder.weight_ih_l0']))
        self.assertFalse(torch.equal(before['reading.weight'], model.state_dict()['reading.weight']))

    def test_evaluate_g2p(self):
        model = context.add_reading_heads(tiny_g2p())
        labels = tiny_labels()

        def decode(_model, words, readings):
            return [labels['words'][word]['readings'][reading][0].split() if word == 'read' else ['X']
                    for word, reading in zip(words, readings)]
        with unittest.mock.patch.object(context, 'decode_readings', decode), \
                unittest.mock.patch.object(context, 'ambiguity_logits',
                                           lambda _model, words: [1.0 if word in ('read', 'cat') else -1.0 for word in words]):
            result = context.evaluate_g2p(model, labels, ['cat', 'dog'])
        self.assertEqual(result['readingsRight'], 2)
        self.assertEqual(result['readings'], 6)
        self.assertEqual(result['readingsWrong'], ['lives/0 X', 'lives/1 X', 'use/0 X', 'use/1 X'])
        self.assertEqual(result['ambiguousMissed'], ['lives', 'use'])
        self.assertEqual(result['falsePositives'], 1)
        self.assertEqual(result['falsePositiveWords'], ['cat'])

    def test_context_network(self):
        self.assertEqual(context.item_chars('read'), [context.CONTEXT_CHAR_IDS[char] for char in 'read'])
        self.assertEqual(len(context.item_chars('x' * 20 + 'ing')), context.MAX_CHARS)
        self.assertEqual(context.item_chars('a-b'), [context.CONTEXT_CHAR_IDS['a'], context.CONTEXT_CHAR_IDS['b']])
        chars, lengths = context.window_arrays([window('', 'i', 'will', 'read', '.', '', '')])
        self.assertEqual(chars.shape, (1, 2 * context.WINDOW + 1, context.MAX_CHARS))
        self.assertEqual(lengths.tolist(), [[0, 1, 4, 4, 1, 0, 0]])
        torch.manual_seed(0)
        model = context.build_context({'embed': 4, 'hidden': 4, 'layer': 4})
        self.assertEqual(model.config, {'embed': 4, 'hidden': 4, 'layer': 4, 'slots': context.READINGS})
        logits = model(torch.from_numpy(chars), torch.from_numpy(lengths))
        self.assertEqual(tuple(logits.shape), (1, context.READINGS))
        slots = context.context_predict(model, [window('', 'i', 'will', 'read'), window('read')] * 3, batch_size=4)
        self.assertEqual(len(slots), 6)
        self.assertTrue(all(0 <= slot < context.READINGS for slot in slots))

        # Items with no characters have zero vectors
        vectors = model.item_vectors(torch.zeros(2, context.MAX_CHARS, dtype=torch.long), torch.zeros(2, dtype=torch.long))
        self.assertTrue(torch.equal(vectors, torch.zeros(2, 4)))

    def test_weight_quantizer(self):
        torch.manual_seed(0)
        model = context.build_context({'embed': 4, 'hidden': 4, 'layer': 4})
        weight = model.layer.weight.detach().clone()
        bias = model.layer.bias.detach().clone()
        quantizer = context.WeightQuantizer(model, context.EXPORT)
        quantizer.quantize()
        _, _, dequantized = g2p.quantize_array(weight.double().numpy(), context.EXPORT['bits'], context.EXPORT['groupSize'])
        self.assertTrue(torch.equal(model.layer.weight.detach(), torch.from_numpy(dequantized).float()))
        self.assertTrue(torch.equal(model.layer.bias.detach(), bias))
        quantizer.restore()
        self.assertTrue(torch.equal(model.layer.weight.detach(), weight))

    def test_fold_labels(self):
        labels = tiny_labels()
        folded = context.fold_labels(labels, 0)
        aligned = [example for example in folded['examples'] if example['source'] == 'aligned']
        self.assertEqual({example['split'] for example in aligned}, {'train', 'heldout'})
        others = [example for example in folded['examples'] if example['source'] != 'aligned']
        self.assertEqual([example['split'] for example in others], [example['split'] for example in labels['examples']
                                                                     if example['source'] != 'aligned'])

    def test_train_context(self):
        lines = []
        model = context.train_context(tiny_labels(), CONTEXT_TINY, lines.append)
        self.assertFalse(model.training)
        self.assertIn('train-context: ', lines[0])
        self.assertIn('step 5/5 (quantized)', lines[-1])

        # With the reading prior
        model = context.train_context(tiny_labels(), dict(CONTEXT_TINY, prior=0.5, quantSteps=0), lines.append)
        self.assertIn('step 3/3 loss', lines[-1])

    def test_evaluate_context(self):
        labels = tiny_labels()
        labels['examples'].append({'window': window('', '', 'x', 'unseen', '.', '', ''), 'word': 'unseen', 'slot': 0, 'source': 'seed',
                                   'split': 'heldout', 'group': 'seed'})
        result = context.evaluate_context(lambda windows: [0] * len(windows), labels)
        heldout = [example for example in labels['examples'] if example['split'] == 'heldout']
        self.assertEqual(sum(counts['total'] for counts in result['sources'].values()), len(heldout))
        self.assertEqual(result['words']['seed/unseen'], {'right': 1, 'total': 1, 'majority': 1})
        self.assertEqual(result['sources']['silver'], {'right': 0, 'total': 1, 'majority': 1})

    def test_describe_and_gate(self):
        result = {'words': {'aligned/read': {'right': 3, 'total': 4, 'majority': 2}, 'seed/read': {'right': 1, 'total': 1, 'majority': 0},
                            'aligned/lives': {'right': 1, 'total': 2, 'majority': 1}}}
        self.assertEqual(context.describe_heldout(result), 'held-out 5/7 (71.4%), word mean 65.0% (2 words)')
        labels = tiny_labels()
        self.assertEqual(context.gate_words(labels), ['read', 'lives'])
        passed, line = context.gate_result(result, labels)
        self.assertFalse(passed)
        self.assertTrue(line.startswith('held-out aligned 66.7% vs majority 50.0% (+20 needed): read 3/4 (majority 2), lives 1/2'))
        self.assertTrue(line.endswith(' - FAILED'))
        result['words']['aligned/lives']['right'] = 2
        passed, line = context.gate_result(result, labels)
        self.assertTrue(passed)
        self.assertFalse(line.endswith('FAILED'))
        passed, _ = context.gate_result({'words': {}}, labels)
        self.assertFalse(passed)

        # The per-word margins - the S/Z-only pairs marked, the words with few tokens or no labels left out
        margins = {'words': {'aligned/read': {'right': 30, 'total': 40, 'majority': 20},
                             'aligned/use': {'right': 20, 'total': 20, 'majority': 10},
                             'aligned/lives': {'right': 1, 'total': 2, 'majority': 1},
                             'seed/read': {'right': 30, 'total': 30, 'majority': 0},
                             'aligned/other': {'right': 30, 'total': 30, 'majority': 0}}}
        self.assertEqual(context.word_margins(margins, labels), 'per word (vs majority): read 30/40 (+25), use S/Z 20/20 (+50)')

    def save_models(self):
        g2p_model = context.add_reading_heads(tiny_g2p())
        torch.save({'config': g2p_model.config, 'state': g2p_model.state_dict()}, self.path('g2p.pt'))
        torch.manual_seed(1)
        context_model = context.build_context({'embed': 4, 'hidden': 4, 'layer': 4})
        torch.save({'config': context_model.config, 'state': context_model.state_dict()}, self.path('context.pt'))
        return g2p_model, context_model

    def test_export(self):
        self.save_models()
        args = ['export', '--g2p', self.path('g2p.pt'), '--context', self.path('context.pt'), '--out', self.path('g2pModel.js'),
                '--comment', 'A test.']
        with captured() as (_, stderr):
            self.assertEqual(context.main(args), 0)
        self.assertIn('export: wrote', stderr.getvalue())
        with open(self.path('g2pModel.js'), encoding='utf-8') as file:
            text = file.read()
        self.assertTrue(text.startswith('// Generated by nn/context.py export - do not edit\n//\n// The neural grapheme-to-phoneme model'))
        self.assertIn('// A test.\n', text)
        self.assertIn('6-bit weights', text)
        module = g2pmodel.parse_module(text)
        self.assertEqual(module['readings'], context.READINGS)
        self.assertEqual(module['context']['window'], context.WINDOW)
        self.assertEqual(module['context']['slots'], context.READINGS)
        self.assertIn('reading.weight', g2pmodel.unpack_tensors(module))
        self.assertIn('window_gru.weight_ih_l0', g2pmodel.unpack_tensors(module['context']))
        self.assertTrue(all(len(line) <= g2p.MODULE_LINE_LENGTH for line in text.split('\n') if not line.startswith('//')))

        # The module as the app runs it (g2pmodel.py) pronounces with it
        self.assertIsInstance(g2pmodel.G2p(module).pronounce('read'), list)

        # Other bits; a module whose weights don't round-trip isn't written
        with captured():
            self.assertEqual(context.main(args[:-2] + ['--bits', '8']), 0)
        with open(self.path('g2pModel.js'), encoding='utf-8') as file:
            self.assertIn('8-bit weights', file.read())
        os.remove(self.path('g2pModel.js'))
        with unittest.mock.patch.object(g2pmodel, 'unpack_tensors',
                                        lambda model: {item['name']: numpy.full(item['shape'], 1e9) for item in model['tensors']}), \
                captured() as (_, stderr):
            self.assertEqual(context.main(args), 1)
        self.assertIn('the module\'s weights differ', stderr.getvalue())
        self.assertFalse(os.path.exists(self.path('g2pModel.js')))

    def test_encode_weights(self):
        # The smallest of the layouts, deflated or not
        quantized = g2p.quantize_state({'weight': torch.zeros(8, 8)}, context.EXPORT)
        encoding, text = context.encode_weights(quantized)
        self.assertEqual(encoding, {'layout': 'bytes', 'deflate': True})
        self.assertTrue(text)
        quantized = g2p.quantize_state({'weight': torch.randn(2, 3, generator=torch.Generator().manual_seed(1))}, context.EXPORT)
        encoding, _ = context.encode_weights(quantized)
        self.assertEqual(encoding, {'layout': 'packed', 'deflate': False})
        self.assertEqual(context.base64_lines('a' * 250), ['a' * 120, 'a' * 120, 'a' * 10])

    def test_gate(self):
        labels = tiny_labels()
        module_g2p = g2pmodel.G2p(g2pmodel.read_module())
        heldout = [example for example in labels['examples'] if example['split'] == 'heldout']
        chosen = module_g2p.choose_slots([example['window'] for example in heldout])

        # The examples' slots the module's choices - better than the majority (every training slot 1)
        for example in labels['examples']:
            if example['split'] == 'train':
                example['slot'] = 1
        for example, slot in zip(heldout, chosen):
            example['slot'] = slot
        for example in heldout:
            if example['word'] in ('read', 'lives') and example['source'] == 'aligned':
                example['slot'] = 0
        path = self.write_labels(labels)
        with unittest.mock.patch.object(module_g2p, 'choose_slots', lambda windows: [0] * len(windows)), \
                unittest.mock.patch.object(g2pmodel, 'G2p', lambda model: module_g2p), captured() as (stdout, _):
            self.assertEqual(context.main(['gate', '--labels', path, '--out', self.path('gate.json')]), 0)
        self.assertIn('; all held-out: aligned ', stdout.getvalue())
        with open(self.path('gate.json'), encoding='utf-8') as file:
            self.assertIn('aligned/read', json.load(file)['words'])

        # The real module's choices; and below the gate
        with captured() as (stdout, _):
            context.main(['gate', '--labels', path])
        self.assertIn('held-out aligned', stdout.getvalue())
        with unittest.mock.patch.object(module_g2p, 'choose_slots', lambda windows: [2] * len(windows)), \
                unittest.mock.patch.object(g2pmodel, 'G2p', lambda model: module_g2p), captured() as (stdout, _):
            self.assertEqual(context.main(['gate', '--labels', path]), 1)
        self.assertIn('FAILED', stdout.getvalue())

    def test_js_pronouncer_and_heteronym_scores(self):
        module_path = g2pmodel.MODULE_PATH
        pronounce = context.js_pronouncer(module_path)
        self.assertEqual(pronounce(['lead', 'hello']), [['L', 'IY1', 'D'], ['HH', 'AH0', 'L', 'OW1']])
        python = g2pmodel.G2p(g2pmodel.read_module())
        self.assertEqual(pronounce(['cat']), [python.pronounce('cat')])
        kinds, status = context.heteronym_scores(module_path)
        self.assertEqual(status, 0)
        self.assertEqual(set(kinds), {'default', 'context', 'missing'})
        self.assertTrue(all(right <= total for right, total in kinds.values()))

    def test_check(self):
        # Identical below the comment header
        copy = self.path('g2pModel.js')
        with open(g2pmodel.MODULE_PATH, encoding='utf-8') as file:
            text = file.read()
        with open(copy, 'w', encoding='utf-8') as file:
            file.write('// Another provenance\n' + text)
        with captured() as (stdout, _):
            self.assertEqual(context.main(['check', copy]), 0)
        self.assertIn('is identical to', stdout.getvalue())

        # A differing module - the regression set and the book measures within the tolerances, or not
        with open(copy, 'w', encoding='utf-8') as file:
            file.write(text.replace("'version': 1", "'version': 1 "))
        measures = {'book': 0.001, 'bookWrongDistinct': 10, 'top10kWrong': 5, 'monitorWer': 0.1}

        def run(new_kinds, status, new_measures):
            scores = {copy: (new_kinds, status), g2pmodel.MODULE_PATH: ({'default': (48, 52), 'context': (44, 50)}, 0)}
            evaluated = iter([new_measures, measures])
            with unittest.mock.patch.object(context, 'heteronym_scores', lambda path: scores[path]), \
                    unittest.mock.patch.object(context, 'js_pronouncer', lambda path: path), \
                    unittest.mock.patch.object(g2p, 'load_data', lambda relabel: {'relabel': relabel}), \
                    unittest.mock.patch.object(g2p, 'evaluate', lambda pronounce, data: next(evaluated)), \
                    unittest.mock.patch.object(g2p, 'describe', lambda result: f'book {result["book"]}'), captured() as (stdout, _):
                return context.main(['check', copy]), stdout.getvalue()
        status, output = run({'default': (47, 52), 'context': (44, 50)}, 0, dict(measures))
        self.assertEqual(status, 0)
        self.assertIn('within the tolerances', output)
        self.assertIn('regression set default 47/52 (committed 48), context 44/50 (committed 44)', output)
        status, output = run({'default': (40, 52), 'missing': (1, 3)}, 1, dict(measures, book=0.01, monitorWer=0.2))
        self.assertEqual(status, 1)
        self.assertIn('FAILED: the regression set is below its minimums; default 40/52 (committed 48); book 0.01', output)
        self.assertIn('monitorWer 0.2', output)
        self.assertIn('missing 1/3 (committed 0)', output)

        # With no relabeled stage
        config = g2p.load_config()
        config['stages'][-1].pop('relabel', None)
        with unittest.mock.patch.object(g2p, 'load_config', lambda: config):
            status, _ = run({'default': (48, 52)}, 0, dict(measures))
        self.assertEqual(status, 0)

    def test_train_g2p_command(self):
        items, _ = self.tiny_items()
        data = {'first': dict(items), 'monitor': [], 'weightCounts': {'read': 5, 'cat': 3}, 'evalCounts': {'dog': 2}, 'labels': {},
                'stems': {}, 'relabeled': set()}
        plain = tiny_g2p()
        torch.save({'config': plain.config, 'state': plain.state_dict()}, self.path('init.pt'))
        config = {'alpha': 0.5, 'uniform': 0.1, 'unknown': 0.0, 'model': {'embed': 4, 'hidden': 4},
                  'stages': [{'name': 'one'}, {'name': 'focus', 'relabel': 'data/nn/g2p-relabel.json'}]}
        config_path = self.path('train.json')
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump({'g2p': config}, file)
        relabels = []

        def load_data(relabel):
            relabels.append(relabel)
            return data
        log_path = self.path('train.log')
        with unittest.mock.patch.object(g2p, 'load_data', load_data), captured():
            self.assertEqual(context.main(['train-g2p', '--labels', self.write_labels(), '--init', self.path('init.pt'),
                                           '--out', self.path('g2p.pt'), '--config', config_path, '--settings', json.dumps(G2P_TINY),
                                           '--log', log_path]), 0)
        self.assertEqual(relabels, [os.path.join(ROOT, 'data/nn/g2p-relabel.json')])
        checkpoint = torch.load(self.path('g2p.pt'), weights_only=False)
        self.assertEqual(checkpoint['config']['readings'], context.READINGS)
        self.assertEqual(checkpoint['settings']['steps'], 3)
        with open(log_path, encoding='utf-8') as file:
            self.assertIn('train-g2p: readings ', file.read())

        # The recipe's settings (the default config), no relabeled stage
        config['stages'][-1].pop('relabel')
        with unittest.mock.patch.object(g2p, 'load_data', load_data), unittest.mock.patch.object(g2p, 'load_config', lambda path: config), \
                unittest.mock.patch.dict(context.G2P_SETTINGS, G2P_TINY), captured():
            self.assertEqual(context.main(['train-g2p', '--labels', self.write_labels(), '--init', self.path('g2p.pt'),
                                           '--out', self.path('g2p2.pt')]), 0)
        self.assertIsNone(relabels[-1])

    def test_train_context_command(self):
        path = self.write_labels()
        log_path = self.path('train.log')
        with captured():
            self.assertEqual(context.main(['train-context', '--labels', path, '--out', self.path('context.pt'),
                                           '--settings', json.dumps(CONTEXT_TINY), '--log', log_path]), 0)
        checkpoint = torch.load(self.path('context.pt'), weights_only=False)
        self.assertEqual(checkpoint['config']['slots'], context.READINGS)
        with open(self.path('context.json'), encoding='utf-8') as file:
            self.assertIn('sources', json.load(file))
        with open(log_path, encoding='utf-8') as file:
            self.assertIn('train-context: held-out aligned', file.read())

        # A fold held out, the recipe's settings
        with unittest.mock.patch.dict(context.CONTEXT_SETTINGS, CONTEXT_TINY), captured() as (_, stderr):
            self.assertEqual(context.main(['train-context', '--labels', path, '--out', self.path('fold.pt'), '--fold', '1']), 0)
        self.assertIn('train-context fold 1:', stderr.getvalue())

    def test_cross_validate_and_folds(self):
        path = self.write_labels()
        out = self.path('folds')
        os.makedirs(out)
        done = {'words': {'aligned/read': {'right': 2, 'total': 2, 'majority': 1}}, 'sources': {}}
        with open(os.path.join(out, 'fold-0.json'), 'w', encoding='utf-8') as file:
            json.dump(done, file)
        with unittest.mock.patch.dict(context.CONTEXT_SETTINGS, CONTEXT_TINY), unittest.mock.patch.object(context, 'FOLDS', 2), \
                captured() as (stdout, stderr):
            status = context.main(['cross-validate', '--labels', path, '--out', out])
        self.assertIn(status, (0, 1))
        self.assertIn('fold 1: train-context:', stderr.getvalue())
        self.assertNotIn('fold 0:', stderr.getvalue())
        self.assertTrue(os.path.exists(os.path.join(out, 'fold-1.json')))
        self.assertTrue(stdout.getvalue().startswith('2 folds: held-out aligned'))

        # The folds' gate - aligned results only, pooled
        results = [os.path.join(out, 'fold-0.json'), self.path('more.json')]
        with open(results[1], 'w', encoding='utf-8') as file:
            json.dump({'words': {'aligned/read': {'right': 1, 'total': 1, 'majority': 0},
                                 'aligned/lives': {'right': 3, 'total': 3, 'majority': 0},
                                 'seed/read': {'right': 0, 'total': 9, 'majority': 9}}}, file)
        with captured() as (stdout, _):
            self.assertEqual(context.main(['folds', *results, '--labels', path]), 0)
        self.assertEqual(stdout.getvalue(), '2 folds: held-out aligned 100.0% vs majority 16.7% (+20 needed): read 3/3 (majority 1), '
                                            'lives 3/3 (majority 0)\nper word (vs majority): \n')
        with open(results[1], 'w', encoding='utf-8') as file:
            json.dump({'words': {'aligned/read': {'right': 0, 'total': 40, 'majority': 40}}}, file)
        with captured() as (stdout, _):
            self.assertEqual(context.command_folds(argparse.Namespace(results=results, labels=path)), 1)
        self.assertIn('per word (vs majority): read 2/42 (-93)', stdout.getvalue())

    def test_main_module(self):
        # The module run as a script (python -m)
        with unittest.mock.patch.dict(sys.modules), unittest.mock.patch.object(sys, 'argv', ['context', '--help']), \
                captured() as (stdout, _):
            sys.modules.pop('picovox_train.context')
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.context', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('The heteronym models', stdout.getvalue())
