# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import base64
import json
import os
import runpy
import tempfile
import unittest
import unittest.mock
import warnings

import numpy
import torch

from picovox_train import ROOT, export, params
from picovox_train.synthesize import Voice
from tests import helpers, model_fixtures, parity


class TestExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        cls.data_dir = cls.patches.__enter__()
        model_fixtures.write_data(cls.data_dir)
        cls.checkpoint = model_fixtures.write_checkpoint(os.path.join(cls.root, 'teacher.pt'), args={
            'recipe_file': os.path.join(ROOT, 'data', 'nn', 'train.json'), 'init': 'build/train/runs/a/best.pt',
            'init_critic': os.path.join(cls.root, 'critic.pt'), 'steps': 3, 'config': '{}', 'files': ['/a/b', 'c']
        })

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_load_model(self):
        model, state = export.load_model(self.checkpoint)
        self.assertFalse(model.training)
        self.assertEqual(state['speakers'], model_fixtures.SPEAKERS)
        self.assertEqual(model.config['dim'], 16)
        # The multi-speaker model only
        state['config'] = {**state['config'], 'kind': 'single'}
        path = os.path.join(self.root, 'single.pt')
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'only the multi-speaker model'):
            export.load_model(path)

    def test_exported_tensors(self):
        model, _ = export.load_model(self.checkpoint)
        with torch.no_grad():
            model.eq_free.fill_(10.0)
        tensors = export.exported_tensors(model)
        self.assertNotIn('eq_free', tensors)
        expected = torch.full_like(tensors['eq_table'], params.EQ_RANGE_DB * float(numpy.tanh(10.0)))
        self.assertTrue(torch.allclose(tensors['eq_table'], expected))
        self.assertTrue(all(tensor.dtype == torch.float32 for tensor in tensors.values()))

    def test_quantized_model(self):
        model, _ = export.load_model(self.checkpoint)
        quantized = export.quantized_model(model)
        state, original = quantized.state_dict(), model.state_dict()
        name = 'embedding.weight'
        self.assertFalse(torch.equal(state[name], original[name]))
        self.assertLess(float((state[name] - original[name]).abs().max()), float(original[name].abs().max()) / 64)
        self.assertTrue(torch.equal(state['head.bias'], original['head.bias']))
        self.assertTrue(torch.equal(state['eq_free'], original['eq_free']))

    def test_token_remap(self):
        vocab = ['<pad>', '<unk>', '<question>', 'AA0', 'AA1', 'AA2', 'B', 'CH', 'EH0', 'EH1', 'EH2', '<period>']
        counts = {vocab.index('AA1'): 50, vocab.index('AA0'): 3, vocab.index('B'): 20, vocab.index('<period>'): 30,
                  vocab.index('EH1'): 5}
        self.assertEqual(export.token_remap(vocab, counts), {
            '<unk>': '_', '<question>': '<period>', 'AA0': 'AA1', 'AA2': 'AA1'
        })

    def test_training_token_counts(self):
        counts, vocab = export.training_token_counts({'args': {'cremad': 0.2}})
        counts_readers, _ = export.training_token_counts({'args': {'cremad': 0}})
        counts_none, _ = export.training_token_counts({})
        self.assertEqual(vocab[2], '<bos>')
        self.assertEqual(counts_readers, counts_none)
        self.assertGreater(counts[2], counts_readers[2])
        # (the readers' train split - two utterances per speaker, two speakers)
        self.assertEqual(counts_readers[2], 4)

    def test_speaker_list(self):
        speakers = export.speaker_list({'speakers': model_fixtures.SPEAKERS})
        self.assertEqual([speaker['role'] for speaker in speakers], ['reader', 'reader', 'style', 'style'])
        self.assertEqual(speakers[0]['key'], 'arctic-slt')
        self.assertEqual(speakers[0]['sex'], 'female')
        self.assertEqual(speakers[2], {'id': 2, 'key': 'cremad-1001', 'sex': 'male', 'corpus': 'CREMA-D', 'role': 'style'})
        self.assertEqual(export.speaker_list({'speakers': ['arctic-slt', 'unknown']})[1]['sex'], None)
        with self.assertRaisesRegex(ValueError, 'is not its data/nn/speakers.json id'):
            export.speaker_list({'speakers': ['arctic-bdl']})

    def test_relative_paths(self):
        self.assertEqual(export.relative_paths({
            'under': os.path.join(ROOT, 'data', 'nn', 'train.json'), 'outside': os.path.join(os.path.dirname(ROOT), 'x.pt'),
            'relative': 'build/train/runs/a.pt', 'number': 3, 'none': None, 'list': [os.path.join(ROOT, 'a'), 'b', 1],
            'nested': {'path': os.path.join(ROOT, 'lib')}
        }), {
            'under': 'data/nn/train.json', 'outside': '../x.pt', 'relative': 'build/train/runs/a.pt', 'number': 3, 'none': None,
            'list': ['a', 'b', 1], 'nested': {'path': 'lib'}
        })
        self.assertIsNone(export.relative_paths(None))

    def test_export(self):
        model, state = export.load_model(self.checkpoint)
        with tempfile.TemporaryDirectory() as out_dir:
            manifest = export.export(model, state, 'voice', out_dir, extra={'note': 'x'}, speaker=1)
            with open(os.path.join(out_dir, 'voice.json'), encoding='utf-8') as file:
                self.assertEqual(json.load(file), json.loads(json.dumps(manifest)))
            binary_size = os.path.getsize(os.path.join(out_dir, 'voice.bin'))
        self.assertEqual(manifest['bytes'], binary_size)
        self.assertEqual(manifest['note'], 'x')
        self.assertEqual(manifest['defaultSpeaker'], 1)
        self.assertEqual(manifest['inference'], {'pauseThreshold': 0.5, 'bandwidthScale': 1.0})
        self.assertEqual(len(manifest['eq']), params.EQ_BAND_COUNT)
        encodings = {tensor['name']: tensor['encoding'] for tensor in manifest['tensors']}
        self.assertEqual(encodings['embedding.weight'], 'int8')
        self.assertEqual(encodings['eq_table'], 'float32')
        self.assertEqual(encodings['head.bias'], 'float32')
        self.assertEqual(manifest['parameterCountCore'], manifest['parameterCount'] - 4 * 4 - 4 * params.EQ_BAND_COUNT)
        # The training record - its paths relative to the repository root, never absolute
        training = manifest['training']
        self.assertEqual(training['step'], 3)
        self.assertEqual(training['args']['recipe_file'], 'data/nn/train.json')
        self.assertEqual(training['args']['init'], 'build/train/runs/a/best.pt')
        self.assertEqual(training['args']['init_critic'], os.path.relpath(os.path.join(self.root, 'critic.pt'), ROOT))
        self.assertFalse(os.path.isabs(training['args']['files'][0]))
        self.assertNotIn(ROOT, json.dumps(manifest))

    def test_main(self):
        with tempfile.TemporaryDirectory() as out_dir, helpers.captured() as (stdout, _):
            self.assertEqual(export.main([self.checkpoint, '--voice', 'model', '--out', out_dir]), 0)
            self.assertEqual(export.main([self.checkpoint, '--voice', 'small', '--out', out_dir, '--speaker', 'arctic-slt']), 0)
            with open(os.path.join(out_dir, 'small.json'), encoding='utf-8') as file:
                small = json.load(file)
            with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as file:
                default = json.load(file)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([line['voice'] for line in lines], ['model', 'small'])
        self.assertEqual(lines[0]['quantized'] + lines[0]['float32'], len(default['tensors']))
        self.assertEqual(default['defaultSpeaker'], 1)
        self.assertEqual(small['defaultSpeaker'], 0)

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['export', '--help']), helpers.captured() as (stdout, _), \
                self.assertRaises(SystemExit) as raised, warnings.catch_warnings():
            # (runpy's warning that the module is imported already - the tests imported it)
            warnings.filterwarnings('ignore', "'picovox_train.export' found in sys.modules", RuntimeWarning)
            runpy.run_module('picovox_train.export', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('usage:', stdout.getvalue())


class TestExportParity(unittest.TestCase):
    """lib/model.js against PyTorch with the export's dequantized weights - the browser path's inference is the exported
    model's (the parity check - tools/parity.js: the exported models are its committed inputs)"""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        cls.data_dir = cls.patches.__enter__()
        model_fixtures.write_data(cls.data_dir)
        cls.checkpoint = model_fixtures.write_checkpoint(os.path.join(cls.root, 'teacher.pt'), seed=3)

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    def check_parity(self, name, speaker, style=None):
        texts = ['Call me Ishmael.', 'It was the best of times, it was the worst of times.', 'Is it cold? Wait here!']
        model, state = export.load_model(self.checkpoint)
        out_dir = os.path.join(self.root, name)
        manifest = export.export(model, state, 'model', out_dir)
        with open(os.path.join(out_dir, 'model.bin'), 'rb') as file:
            binary = base64.b64encode(file.read()).decode('ascii')
        parity.inputs(self, name, {'manifest': manifest, 'bin': binary, 'speaker': speaker, 'style': style, 'intensity': 0.7,
                                   'texts': texts})
        voice = Voice(self.checkpoint, speaker=speaker, emotion=style, intensity=0.7)
        for text, expected in zip(texts, parity.outputs(name), strict=True):
            values, parts = voice.frames(text)
            self.assertEqual(expected['durations'], [duration for part in parts for duration in part['durations']])
            js_values = numpy.array(expected['frames'], dtype=numpy.float64)
            self.assertEqual(js_values.shape, values[:, :params.COUNT].shape)
            self.assertGreater(len(values), 10)
            difference = numpy.abs(params.normalize(js_values) - params.normalize(values[:, :params.COUNT]))
            self.assertLess(float(difference.max()), 1e-4)
            self.assertTrue(numpy.allclose(expected['eq'], voice.eq, atol=1e-3))

    def test_parity(self):
        self.check_parity('export', 1)

    def test_parity_style(self):
        self.check_parity('export-style', 3, style='happy')
