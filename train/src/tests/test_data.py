# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import os
import tempfile
import unittest
import unittest.mock

import numpy

from picovox_train import data, frontend
from tests import data_fixtures, helpers


class TestProcessing(unittest.TestCase):

    # The fake acoustic analysis
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS))

    def test_silent_runs(self):
        rate = 16000
        samples = numpy.concatenate([data_fixtures.buzz(0.3, rate), numpy.zeros(int(0.2 * rate)), data_fixtures.buzz(0.3, rate),
                                     numpy.zeros(int(0.03 * rate)), data_fixtures.buzz(0.3, rate)])
        runs = data.silent_runs(samples, rate)
        self.assertEqual(len(runs), 1)
        start, end = runs[0]
        self.assertAlmostEqual(start, 0.3, delta=0.02)
        self.assertAlmostEqual(end, 0.5, delta=0.02)

    def test_sample_track(self):
        values, valid = data.sample_track(numpy.zeros(0), numpy.zeros(0), 3)
        self.assertEqual(values.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(valid.tolist(), [False, False, False])
        values, valid = data.sample_track(numpy.array([0.005]), numpy.array([100.0]), 2)
        self.assertEqual(values.tolist(), [100.0, 0.0])
        self.assertEqual(valid.tolist(), [True, False])
        times = (numpy.arange(4) + 0.5) * 0.01
        values, valid = data.sample_track(times, numpy.array([1.0, 2.0, 3.0, 4.0]), 3)
        self.assertEqual(values.tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(valid.all())

    def test_frame_tracks(self):
        samples = numpy.concatenate([numpy.zeros(4800), data_fixtures.buzz(0.4, data.SAMPLE_RATE)])
        tracks = data.frame_tracks(samples, 60, 75, 400, 5000)
        self.assertEqual(tracks.shape, (60, len(data.CHANNELS)))
        voiced = tracks[:, data.CHANNELS.index('voiced')]
        self.assertEqual(voiced[:15].sum(), 0)
        self.assertGreater(voiced[25:55].mean(), 0.9)
        f0 = tracks[25:55, data.CHANNELS.index('f0')]
        self.assertAlmostEqual(float(numpy.median(f0[f0 > 0])), 150.0, delta=3)
        self.assertEqual(float(tracks[5, data.CHANNELS.index('energy')]), -100.0)
        self.assertGreater(tracks[40, data.CHANNELS.index('f1')], 0)

    def test_raw_to_npy(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, npy = os.path.join(directory, 'frames.raw'), os.path.join(directory, 'frames.npy')
            values = numpy.arange(12, dtype=numpy.float32).reshape(4, 3)
            with open(raw, 'wb') as file:
                file.write(values.tobytes())
            data.raw_to_npy(raw, npy, numpy.float32, (4, 3))
            self.assertFalse(os.path.exists(raw))
            self.assertTrue(numpy.array_equal(numpy.load(npy), values))


class TestLoading(unittest.TestCase):

    def test_corpus_data(self):
        with tempfile.TemporaryDirectory() as directory:
            index = helpers.write_corpus(directory)
            corpus = data.CorpusData(directory)
            self.assertEqual(corpus.directory, directory)
            self.assertEqual(corpus.index, index)
            self.assertEqual(len(corpus.entries()), 6)
            self.assertEqual(len(corpus.entries(include_excluded=True)), 8)
            self.assertEqual([entry['id'] for entry in corpus.entries('val')], ['male/0002', 'female/0002'])
            self.assertEqual([entry['id'] for entry in corpus.entries(speakers=['female'], include_excluded=True)],
                             ['female/0000', 'female/0001', 'female/0002', 'female/0003'])
            entry = corpus.entries()[1]
            arrays = corpus.arrays(entry)
            self.assertEqual(arrays['tokens'].tolist(), frontend.tokenize(entry['text'])['ids'])
            self.assertEqual(arrays['tokens'].dtype, numpy.int64)
            self.assertEqual(int(arrays['durations'].sum()), entry['frameCount'])
            self.assertEqual(arrays['seconds'].dtype, numpy.float32)
            self.assertEqual(len(arrays['audio']), entry['sampleCount'])
            self.assertLessEqual(float(numpy.abs(arrays['audio']).max()), 1.0)
            self.assertEqual(arrays['frames'].shape, (entry['frameCount'], len(data.CHANNELS)))
