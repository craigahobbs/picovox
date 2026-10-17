# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import tempfile
import unittest
import unittest.mock
import warnings

import numpy
import soundfile
import torch

from picovox_train import export, frontend, params, synthesize

from . import helpers
from .test_trainmulti import TrainData


DATA = None


def setUpModule():  # pylint: disable=invalid-name
    global DATA  # pylint: disable=global-statement
    DATA = TrainData()
    DATA.train('voice', '--steps', '1')


def tearDownModule():  # pylint: disable=invalid-name
    DATA.cleanup()


class TestSynthesize(unittest.TestCase):

    def setUp(self):
        patcher = unittest.mock.patch.object(export, 'DATA_DIR', DATA.data_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.checkpoint = DATA.run_path('voice', 'last.pt')

    def test_sentences(self):
        vocab = frontend.vocabulary()
        ids = frontend.tokenize('Hello there. Who are you? Fine')['ids']
        parts = synthesize.sentences(ids, vocab)
        self.assertEqual(len(parts), 3)
        self.assertEqual(synthesize.sentences([2, 20, 21], vocab), [[2, 20, 21]])
        self.assertTrue(all(part[0] == 2 for part in parts))
        self.assertEqual([vocab[part[-1]] for part in parts[:2]], ['<period>', '<wh>'])
        self.assertEqual(sum(len(part) - 1 for part in parts), len(ids) - 1)
        # (without a leading <bos>; nothing after the last sentence's end)
        self.assertEqual(synthesize.sentences(ids[1:5] + [vocab.index('<period>')], vocab), [[2] + ids[1:5] + [vocab.index('<period>')]])
        self.assertEqual(synthesize.sentences([2], vocab), [])

    def test_speaker_id(self):
        state = {'speakers': ['arctic-slt', 'arctic-bdl']}
        self.assertEqual(synthesize.speaker_id(state, 1), 1)
        self.assertEqual(synthesize.speaker_id(state, '1'), 1)
        self.assertEqual(synthesize.speaker_id(state, 'arctic-bdl'), 1)

    def test_voice(self):
        voice = synthesize.Voice(self.checkpoint, speaker='arctic-bdl')
        self.assertEqual(voice.speaker, 1)
        self.assertFalse(voice.model.training)
        self.assertEqual(voice.speaker_input.tolist(), [1])
        self.assertEqual(voice.emotion.tolist(), [-1])
        self.assertEqual(voice.eq.shape, (params.EQ_BAND_COUNT,))
        # (the tiny training data has no <question> - remapped to <period>)
        self.assertEqual(voice.remap[voice.vocab.index('<question>')], voice.vocab.index('<period>'))
        values, parts = voice.frames('Hello there. Is it?')
        self.assertEqual(values.shape[1], params.COUNT)
        self.assertEqual(len(parts), 2)
        self.assertEqual(values.shape[0], 2 * sum(sum(part['durations']) for part in parts) + 1)
        self.assertTrue(numpy.array_equal(values[-1], values[-2]))
        empty, parts = voice.frames('')
        self.assertEqual((empty.shape, parts), ((0, params.COUNT), []))

    def test_voice_blend_style(self):
        voice = synthesize.Voice(self.checkpoint, quantized=False, blend={'0': 1, 'arctic-bdl': 3}, emotion='happy', intensity=0.5,
                                 bandwidth_scale=0.5)
        self.assertEqual(voice.speaker_input.tolist(), [[0.25, 0.75, 0.0, 0.0]])
        self.assertEqual(voice.emotion.tolist(), [3])
        self.assertEqual(voice.intensity.tolist(), [0.5])
        self.assertEqual(voice.condition(), {'speaker': voice.speaker_input, 'emotion': voice.emotion,
                                             'intensity': voice.intensity})
        values, _ = voice.frames('Hello.')
        plain = synthesize.Voice(self.checkpoint, quantized=False, blend={'0': 1, 'arctic-bdl': 3}, emotion='happy',
                                 intensity=0.5)
        unscaled, _ = plain.frames('Hello.')
        self.assertTrue(numpy.allclose(values[:, params.INDEX['b1']], 0.5 * unscaled[:, params.INDEX['b1']]))
        self.assertTrue(numpy.array_equal(values[:, params.INDEX['f1']], unscaled[:, params.INDEX['f1']]))
        # The 4-bit export's weights
        quantized = synthesize.Voice(self.checkpoint, bits=4, group=8)
        self.assertFalse(torch.equal(quantized.model.head.weight, plain.model.head.weight))

    def test_render(self):
        voice = synthesize.Voice(self.checkpoint)
        values, _ = voice.frames('Hello.')
        audio = synthesize.render(values, voice.eq, 16000)
        self.assertEqual(len(audio), (len(values) - 1) * 80)
        self.assertAlmostEqual(float(numpy.abs(audio).max()), 0.89, places=6)
        self.assertTrue(numpy.array_equal(audio, synthesize.render(values, voice.eq, 16000)))
        silent = numpy.tile([params.DEFAULTS[name] for name in params.NAMES], (5, 1))
        quiet = synthesize.render(silent, numpy.zeros(params.EQ_BAND_COUNT), 16000)
        self.assertEqual(len(quiet), 320)
        self.assertTrue(numpy.all(numpy.isfinite(quiet)))

    def test_main(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = os.path.join(directory, 'hello.wav')
            frames = os.path.join(directory, 'hello.json')
            with helpers.captured() as (stdout, _):
                self.assertEqual(synthesize.main([self.checkpoint, 'Hello.', '-o', wav, '--rate', '16000', '--params', frames,
                                                  '--speaker', 'arctic-slt', '--style', 'sad', '--intensity', '0.5']), 0)
            audio, rate = soundfile.read(wav)
            with open(frames, encoding='utf-8') as file:
                written = json.load(file)
            self.assertRegex(stdout.getvalue(), r'^hello\.wav: [0-9.]+ s\n$')
            self.assertEqual(rate, 16000)
            self.assertEqual(len(audio), (len(written['frames']) - 1) * 80)
            self.assertEqual(len(written['eq']), params.EQ_BAND_COUNT)
            with helpers.captured():
                self.assertEqual(synthesize.main([self.checkpoint, 'Hello.', '-o', wav, '--rate', '16000', '--float', '--blend',
                                                  '{"0": 1, "1": 1}', '--bandwidth-scale', '0.9']), 0)
            self.assertEqual(soundfile.read(wav)[1], 16000)

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['synthesize']), helpers.captured() as (_, stderr), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.synthesize', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('required', stderr.getvalue())
