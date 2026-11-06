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

from picovox_train import ROOT, export, params, quant
from tests import helpers, model_fixtures, parity
from tests.synthesize import Voice


class TestExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        cls.data_dir = cls.patches.__enter__()
        model_fixtures.write_data(cls.data_dir)
        cls.checkpoint = model_fixtures.write_checkpoint(os.path.join(cls.root, 'teacher.pt'), args={
            'recipe_file': os.path.join(ROOT, 'train', 'data', 'train.json'), 'init': 'build/train/runs/a/best.pt',
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
        data = export.training_data()
        quantized = export.quantized_model(model, data)
        state, original = quantized.state_dict(), model.state_dict()
        name = 'embedding.weight'
        self.assertFalse(torch.equal(state[name], original[name]))
        self.assertLess(float((state[name] - original[name]).abs().max()), float(original[name].abs().max()) / 2 / quant.LIMIT * 1.001)
        # (the matrices and kernels - GPTQ's, on the grid of their scales; the tiny model's kernels quantized too)
        with unittest.mock.patch.object(quant, 'MIN_QUANTIZED', 1):
            state = export.quantized_model(model, data).state_dict()
            tensors = export.quantized_tensors(model, data)
        for name in ('head.weight', 'decoder.0.conv.conv.weight'):
            values, scales = tensors[name]
            self.assertTrue(torch.equal(state[name], torch.tensor(quant.dequantize(values, scales, original[name].shape))), name)
        self.assertTrue(torch.equal(state['head.bias'], original['head.bias']))
        self.assertTrue(torch.equal(state['eq_free'], original['eq_free']))

    def test_calibration(self):
        # The training split's utterances - a fixed selection of CALIBRATION when there are more
        data = export.training_data()
        items = export.calibration_items(data)
        self.assertTrue(items)
        self.assertTrue(all(tokens.shape == durations.shape and tokens.shape[0] == 1 and speaker.shape == (1,)
                            for tokens, durations, speaker in items))
        with unittest.mock.patch.object(export, 'CALIBRATION', 2):
            fewer = export.calibration_items(data)
            self.assertEqual(len(fewer), 2)
            self.assertEqual([item[0].tolist() for item in export.calibration_items(data)], [item[0].tolist() for item in fewer])

        # The covariances - every quantized matrix and kernel, not the embedding table; a depthwise kernel's per channel
        model, _ = export.load_model(self.checkpoint)
        with unittest.mock.patch.object(quant, 'MIN_QUANTIZED', 1):
            covariances = export.input_covariances(model, data)
        self.assertNotIn('embedding.weight', covariances)
        self.assertEqual(covariances['head.weight'].shape, (model.head.weight.shape[1],) * 2)
        conv = model.decoder[0].conv.conv
        self.assertEqual(covariances['decoder.0.conv.conv.weight'].shape, (conv.out_channels, conv.kernel_size[0], conv.kernel_size[0]))

    def test_token_remap(self):
        vocab = ['<pad>', '<unk>', '<question>', 'AA0', 'AA1', 'AA2', 'B', 'CH', 'EH0', 'EH1', 'EH2', '<period>']
        counts = {vocab.index('AA1'): 50, vocab.index('AA0'): 3, vocab.index('B'): 20, vocab.index('<period>'): 30,
                  vocab.index('EH1'): 5}
        self.assertEqual(export.token_remap(vocab, counts), {
            '<unk>': '_', '<question>': '<period>', 'AA0': 'AA1', 'AA2': 'AA1'
        })

    def test_training_token_counts(self):
        counts, vocab = export.training_token_counts(export.training_data())
        self.assertEqual(vocab[2], '<bos>')
        # (the train split - two utterances per speaker, two speakers)
        self.assertEqual(counts[2], 4)

    def test_speaker_list(self):
        data = export.training_data()
        speakers = export.speaker_list({'speakers': model_fixtures.SPEAKERS}, data)
        self.assertEqual(speakers[1], {'id': 1, 'key': 'female', 'sex': 'female', 'source': 'phonon Female (0123456789ab)',
                                       'medianF0': 150.0})
        self.assertEqual(speakers[0]['sex'], 'male')
        with self.assertRaisesRegex(ValueError, 'speaker female: table row 0 is not its index 1'):
            export.speaker_list({'speakers': ['female']}, data)

    def test_relative_paths(self):
        self.assertEqual(export.relative_paths({
            'under': os.path.join(ROOT, 'train', 'data', 'train.json'), 'outside': os.path.join(os.path.dirname(ROOT), 'x.pt'),
            'relative': 'build/train/runs/a.pt', 'number': 3, 'none': None, 'list': [os.path.join(ROOT, 'a'), 'b', 1],
            'nested': {'path': os.path.join(ROOT, 'lib')}
        }), {
            'under': 'train/data/train.json', 'outside': '../x.pt', 'relative': 'build/train/runs/a.pt', 'number': 3, 'none': None,
            'list': ['a', 'b', 1], 'nested': {'path': 'lib'}
        })
        self.assertIsNone(export.relative_paths(None))

    def test_export(self):
        model, state = export.load_model(self.checkpoint)
        with tempfile.TemporaryDirectory() as out_dir:
            manifest = export.export(model, state, 'voice', out_dir)
            with open(os.path.join(out_dir, 'voice.json'), encoding='utf-8') as file:
                self.assertEqual(json.load(file), json.loads(json.dumps(manifest)))
            binary_size = os.path.getsize(os.path.join(out_dir, 'voice.bin'))
        self.assertEqual(manifest['bytes'], binary_size)
        self.assertEqual((manifest['kind'], manifest['defaultSpeaker']), ('multi', 0))
        with torch.no_grad():
            self.assertEqual(manifest['eq'], [round(float(value), 3) for value in model.eq(torch.tensor([0]))[0]])
        self.assertEqual(len(manifest['eq']), params.EQ_BAND_COUNT)
        encodings = {tensor['name']: tensor['encoding'] for tensor in manifest['tensors']}
        self.assertEqual(encodings['embedding.weight'], 'packed')
        self.assertTrue(all(tensor['bits'] == quant.BITS for tensor in manifest['tensors'] if tensor['encoding'] == 'packed'))
        self.assertEqual(encodings['eq_table'], 'float32')
        self.assertEqual(encodings['head.bias'], 'float32')
        self.assertEqual(manifest['parameterCountCore'], manifest['parameterCount'] - 2 * 4 - 2 * params.EQ_BAND_COUNT)
        self.assertEqual(manifest['license'], 'Trained on speech generated by Phonon (Apache 2.0)')
        self.assertNotIn('styles', manifest)
        # The training record - its paths relative to the repository root, never absolute
        training = manifest['training']
        self.assertEqual(training['step'], 3)
        self.assertEqual(training['args']['recipe_file'], 'train/data/train.json')
        self.assertEqual(training['args']['init'], 'build/train/runs/a/best.pt')
        self.assertEqual(training['args']['init_critic'], os.path.relpath(os.path.join(self.root, 'critic.pt'), ROOT))
        self.assertFalse(os.path.isabs(training['args']['files'][0]))
        self.assertNotIn(ROOT, json.dumps(manifest))

    def test_main(self):
        with tempfile.TemporaryDirectory() as out_dir, helpers.captured() as (stdout, _):
            self.assertEqual(export.main([self.checkpoint, '--voice', 'model', '--out', out_dir]), 0)
            with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as file:
                default = json.load(file)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([line['voice'] for line in lines], ['model'])
        self.assertEqual(lines[0]['quantized'] + lines[0]['float32'], len(default['tensors']))
        self.assertEqual(default['defaultSpeaker'], 0)

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

    def check_parity(self, name, speaker):
        texts = ['Call me Ishmael.', 'It was the best of times, it was the worst of times.', 'Is it cold? Wait here!']
        model, state = export.load_model(self.checkpoint)
        out_dir = os.path.join(self.root, name)
        manifest = export.export(model, state, 'model', out_dir)
        with open(os.path.join(out_dir, 'model.bin'), 'rb') as file:
            binary = base64.b64encode(file.read()).decode('ascii')
        parity.inputs(self, name, {'manifest': manifest, 'bin': binary, 'speaker': speaker, 'texts': texts})
        voice = Voice(self.checkpoint, speaker=speaker)
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
