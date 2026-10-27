# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest
import unittest.mock

import numpy
import torch

from picovox_train import export, params

from . import synthesize
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
        # The export's weights
        model, _ = export.load_model(self.checkpoint)
        self.assertTrue(torch.equal(voice.model.head.weight, export.quantized_model(model).head.weight))
        self.assertFalse(torch.equal(voice.model.head.weight, model.head.weight))
        # (the tiny training data has no <question> - remapped to <period>)
        self.assertEqual(voice.remap[voice.vocab.index('<question>')], voice.vocab.index('<period>'))
        values, parts = voice.frames('Hello there. Is it?')
        self.assertEqual(values.shape[1], params.COUNT)
        self.assertEqual(len(parts), 2)
        self.assertEqual(values.shape[0], 2 * sum(sum(part['durations']) for part in parts) + 1)
        self.assertTrue(numpy.array_equal(values[-1], values[-2]))
        empty, parts = voice.frames('')
        self.assertEqual((empty.shape, parts), ((0, params.COUNT), []))

    def test_voice_style(self):
        voice = synthesize.Voice(self.checkpoint, emotion='happy', intensity=0.5)
        self.assertEqual(voice.speaker, 0)
        self.assertEqual(voice.emotion.tolist(), [3])
        self.assertEqual(voice.intensity.tolist(), [0.5])
        values, _ = voice.frames('Hello.')
        neutral, _ = synthesize.Voice(self.checkpoint).frames('Hello.')
        self.assertFalse(numpy.array_equal(values, neutral))
