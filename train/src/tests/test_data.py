# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import os
import tempfile
import unittest
import unittest.mock

import numpy

from picovox_train import data, frontend
from picovox_train.wav import read_wav
from tests import data_fixtures, helpers
from tests.helpers import write_wav


class TestReading(unittest.TestCase):

    def test_read_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'txt.done.data')
            with open(path, 'w', encoding='utf-8') as file:
                file.write('( arctic_a0001 "Author of the danger trail." )\n( arctic_a0002 "Not at this." )\nnoise\n')
            self.assertEqual(data.read_prompts(path), {'arctic_a0001': 'Author of the danger trail.', 'arctic_a0002': 'Not at this.'})

    def test_read_xwaves(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.lab')
            with open(path, 'w', encoding='utf-8') as file:
                file.write('signal a\nnfields 1\n#\n0.10 125 pau\n0.25 125 k\n\n0.40 125 ao\n')
            self.assertEqual(data.read_xwaves(path), [('pau', 0.0, 0.1), ('k', 0.1, 0.25), ('ao', 0.25, 0.4)])
            self.assertEqual(data.read_labels({'lab': path}), data.read_xwaves(path))

    def test_read_mfa(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.lab')
            with open(path, 'w', encoding='utf-8') as file:
                file.write('0.1\t0.2\tK\n0.2\t0.3\n0.5\t0.6\tAO1\nbad\n')
            expected = [('sil', 0.0, 0.1), ('K', 0.1, 0.2), ('', 0.2, 0.3), ('sil', 0.3, 0.5), ('AO1', 0.5, 0.6)]
            self.assertEqual(data.read_mfa(path), expected)
            self.assertEqual(data.read_labels({'lab': path, 'labelFormat': 'mfa'}), expected)

    def test_arctic_utterances(self):
        with tempfile.TemporaryDirectory() as directory:
            data_fixtures.make_arctic(directory, 'slt', {'arctic_a0001': 'Call me Ishmael.', 'arctic_a0002': None})
            with open(os.path.join(directory, 'cmu_us_slt_arctic', 'wav', 'arctic_a0001.json'), 'w', encoding='utf-8') as file:
                file.write('{}')
            with unittest.mock.patch.object(data, 'ARCTIC_DIR', directory):
                utterances = data.arctic_utterances('slt')
            base = os.path.join(directory, 'cmu_us_slt_arctic')
            self.assertEqual(utterances, [
                {'speaker': 'slt', 'sentence': 'arctic_a0001', 'wav': os.path.join(base, 'wav', 'arctic_a0001.wav'),
                 'lab': os.path.join(base, 'lab', 'arctic_a0001.lab'), 'text': 'Call me Ishmael.'},
                {'speaker': 'slt', 'sentence': 'arctic_a0002', 'wav': os.path.join(base, 'wav', 'arctic_a0002.wav'),
                 'lab': os.path.join(base, 'lab', 'arctic_a0002.lab'), 'text': None}
            ])

    def test_sentence_split(self):
        splits = {data.sentence_split(f'arctic_a{number:04d}') for number in range(200)}
        self.assertEqual(splits, {'train', 'val', 'test'})
        self.assertEqual(data.sentence_split('arctic_a0001'), data.sentence_split('arctic_a0001'))

    def test_mean_label_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.lab')
            data_fixtures.write_xwaves(path, [('pau', 0.0, 0.2), ('k', 0.2, 0.3), ('k', 0.3, 0.5), ('ax', 0.5, 0.6)])
            means = data.mean_label_seconds([{'lab': path}, {'lab': os.path.join(directory, 'missing.lab')}])
        self.assertEqual(set(means), {'PAU', 'K', 'AX'})
        self.assertAlmostEqual(means['K'], 0.15)
        self.assertAlmostEqual(means['PAU'], 0.2)


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

    def test_align_utterance(self):
        with tempfile.TemporaryDirectory() as directory:
            text = 'It was the best of times.'
            # A substituted phone in "best" (match 0.75) and a deleted one in "of" (0.5 - the worst), a pause inside "was"
            labels, seconds = data_fixtures.text_labels(text, drop=(12,), replace={8: 'k'}, pause_after=(4,))
            path = os.path.join(directory, 'a.lab')
            data_fixtures.write_xwaves(path, labels)
            rate = 16000
            samples = data_fixtures.labels_audio(labels, seconds, rate)
            tokens, token_seconds, cut_start, cut_end, report = data.align_utterance({'text': text, 'lab': path}, {}, samples, rate)
        self.assertEqual(tokens, frontend.tokenize(text)['tokens'])
        self.assertEqual(report['ids'], frontend.tokenize(text)['ids'])
        self.assertAlmostEqual(sum(token_seconds), cut_end - cut_start, places=6)
        self.assertAlmostEqual(cut_start, data_fixtures.EDGE_SECONDS - data.MAX_EDGE_SILENCE)
        self.assertAlmostEqual(token_seconds[0], data.MAX_EDGE_SILENCE)
        self.assertEqual(report['words'], 6)
        self.assertEqual(report['worstWord'], 'of')
        self.assertEqual(report['worstWordMatch'], 0.5)
        self.assertEqual(report['deleted'], 1)
        self.assertEqual(report['substituted'], 1)

    def test_align_utterance_short_labels(self):
        # MFA labels end before the recording - a pause is appended to the audio's end
        with tempfile.TemporaryDirectory() as directory:
            text = 'Call me Ishmael.'
            labels, seconds = data_fixtures.text_labels(text, 'mfa')
            path = os.path.join(directory, 'a.lab')
            data_fixtures.write_mfa(path, labels)
            rate = 24000
            samples = data_fixtures.labels_audio(labels, seconds, rate)
            tokens, token_seconds, _, _, report = data.align_utterance({'text': text, 'lab': path, 'labelFormat': 'mfa'}, {},
                                                                       samples, rate)
        self.assertEqual(len(tokens), len(token_seconds))
        self.assertEqual(report['exact'], report['phones'])
        self.assertIsNone(report['worstWord'])
        self.assertEqual(report['worstWordMatch'], 1.0)

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

    def _utterance(self, directory, rate, channels=1):
        text = 'Call me Ishmael.'
        labels, seconds = data_fixtures.text_labels(text)
        samples = data_fixtures.labels_audio(labels, seconds, rate)
        wav = os.path.join(directory, 'a.wav')
        write_wav(wav, numpy.stack([samples] * channels, axis=1) if channels > 1 else samples, rate)
        lab = os.path.join(directory, 'a.lab')
        data_fixtures.write_xwaves(lab, labels)
        return {'text': text, 'wav': wav, 'lab': lab}, seconds

    def test_process_utterance(self):
        with tempfile.TemporaryDirectory() as directory:
            utterance, seconds = self._utterance(directory, 16000, channels=2)
            result = data.process_utterance((utterance, {}, (75, 400, 5500)))
        frame_count = sum(result['frames'])
        self.assertEqual(len(result['frames']), len(result['tokens']))
        self.assertEqual(result['audio'].dtype, numpy.int16)
        self.assertEqual(len(result['audio']), frame_count * data.HOP)
        self.assertEqual(result['tracks'].shape, (frame_count, len(data.CHANNELS)))
        self.assertAlmostEqual(result['cutEnd'] - result['cutStart'], seconds - 2 * (data_fixtures.EDGE_SECONDS - data.MAX_EDGE_SILENCE),
                               places=3)
        self.assertIs(result['utterance'], utterance)

    def test_process_utterance_native_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            utterance, _ = self._utterance(directory, data.SAMPLE_RATE)
            samples = read_wav(utterance['wav'])[0][:, 0]
            result = data.process_utterance((utterance, {}, (75, 400, 5500)))
        # (no resampling - the audio is the recording's samples, cut)
        start = int(round(result['cutStart'] * data.SAMPLE_RATE))
        expected = numpy.clip(numpy.round(samples[start:start + len(result['audio'])] * 32768), -32768, 32767)
        self.assertTrue(numpy.array_equal(result['audio'], expected.astype(numpy.int16)))

    def test_match_rate(self):
        self.assertEqual(data.match_rate({'exact': 3, 'near': 1, 'phones': 8}), 0.5)
        self.assertEqual(data.match_rate({'exact': 0, 'near': 0, 'phones': 0}), 0.0)


class TestLoading(unittest.TestCase):

    def test_arctic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            index = helpers.write_corpus(directory)
            corpus = data.ArcticData(directory)
            self.assertEqual(corpus.directory, directory)
            self.assertEqual(corpus.index, index)
            self.assertEqual(len(corpus.entries()), 6)
            self.assertEqual(len(corpus.entries(include_excluded=True)), 8)
            self.assertEqual([entry['id'] for entry in corpus.entries('val')], ['arctic-slt/0002', 'arctic-bdl/0002'])
            self.assertEqual([entry['id'] for entry in corpus.entries(speakers=['arctic-bdl'], include_excluded=True)],
                             ['arctic-bdl/0000', 'arctic-bdl/0001', 'arctic-bdl/0002', 'arctic-bdl/0003'])
            entry = corpus.entries()[1]
            arrays = corpus.arrays(entry)
            self.assertEqual(arrays['tokens'].tolist(), frontend.tokenize(entry['text'])['ids'])
            self.assertEqual(arrays['tokens'].dtype, numpy.int64)
            self.assertEqual(int(arrays['durations'].sum()), entry['frameCount'])
            self.assertEqual(arrays['seconds'].dtype, numpy.float32)
            self.assertEqual(len(arrays['audio']), entry['sampleCount'])
            self.assertLessEqual(float(numpy.abs(arrays['audio']).max()), 1.0)
            self.assertEqual(arrays['frames'].shape, (entry['frameCount'], len(data.CHANNELS)))
