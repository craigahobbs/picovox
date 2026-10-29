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
        state = {'speakers': ['male', 'female']}
        self.assertEqual(synthesize.speaker_id(state, 1), 1)
        self.assertEqual(synthesize.speaker_id(state, '1'), 1)
        self.assertEqual(synthesize.speaker_id(state, 'female'), 1)

    def test_voice(self):
        voice = synthesize.Voice(self.checkpoint, speaker='female')
        self.assertEqual(voice.speaker, 1)
        self.assertFalse(voice.model.training)
        self.assertEqual(voice.speaker_input.tolist(), [1])
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

    def test_voice_default(self):
        voice = synthesize.Voice(self.checkpoint)
        self.assertEqual(voice.speaker, 0)
        values, _ = voice.frames('Hello.')
        other, _ = synthesize.Voice(self.checkpoint, speaker=1).frames('Hello.')
        self.assertFalse(numpy.array_equal(values, other))
