# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import math
import os
import tempfile
import unittest

import numpy

from picovox_train import ctc, frontend
from tests import data_fixtures
from tests.ctc_fixtures import VOCAB, FakeCtc, clip_labels, fake_recognizer, greedy_of
from tests.helpers import write_wav


# Log posteriors that spell recognizer tokens (each for frames frames, a blank frame after it) - [frames, vocab]
def spelled(tokens, frames=3):
    rows = []
    for token in tokens:
        rows.extend([VOCAB[token]] * frames + [VOCAB['<pad>']])
    posteriors = numpy.full((len(rows), len(VOCAB)), 0.001)
    posteriors[numpy.arange(len(rows)), rows] = 1.0
    return numpy.log(posteriors / posteriors.sum(1, keepdims=True))


# Log posteriors [frames, vocab] with each phone of tokens in its run of frames (blank elsewhere)
def run_posteriors(tokens, runs, frames):
    values = numpy.full((frames, len(VOCAB)), -20.0)
    values[:, 0] = 0.0
    for (_, key), (first, last) in zip(ctc.phone_positions(tokens), runs):
        values[first:last + 1, 0] = -20.0
        values[first:last + 1, VOCAB[ctc.IPA_TOKENS[key][0]]] = 0.0
    return values


class TestRecognizer(unittest.TestCase):

    def test_unstressed(self):
        self.assertEqual(ctc.unstressed('HH AA1 B IY0 Z'), 'HH AA B IY Z')

    def test_load_ctc(self):
        with fake_recognizer() as (model, hub):
            self.assertEqual(ctc.load_ctc(), (model, VOCAB, 'cpu'))
            self.assertEqual(model.device, 'cpu')
            self.assertEqual(os.environ['HF_HOME'], ctc.HF_HOME)
            self.assertEqual(ctc.load_ctc('mps')[2], 'mps')
        self.assertEqual(hub.snapshot_download.call_args.args, (ctc.CTC_REPO,))

    def test_ctc_log_posteriors(self):
        # Sorted by length, batched to max_batch_seconds - each clip's frames, normalized posteriors
        model = FakeCtc()
        samples = [data_fixtures.buzz(seconds, 16000) for seconds in (0.5, 0.2, 0.8)]
        results = ctc.ctc_log_posteriors(model, 'cpu', samples, max_batch_seconds=1.0)
        self.assertEqual([len(result) for result in results], [25, 10, 40])
        self.assertTrue(numpy.allclose(numpy.exp(results[0]).sum(axis=1), 1, atol=1e-5))
        self.assertEqual(model.calls, [(2, 8000), (1, 12800)])
        self.assertEqual(ctc.ctc_log_posteriors(model, 'cpu', []), [])

    def test_read_16k(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            samples = data_fixtures.buzz(0.5, 16000)
            write_wav(path, samples, 16000)
            self.assertTrue(numpy.allclose(ctc.read_16k(path), samples, rtol=0, atol=1 / 32768))
            # Stereo at another rate - mixed down and resampled
            write_wav(path, numpy.stack([samples[::2], samples[::2]], axis=1), 8000)
            resampled = ctc.read_16k(path)
        self.assertEqual(len(resampled), 8000)
        self.assertTrue(resampled.flags['C_CONTIGUOUS'])
        self.assertEqual(resampled.dtype, numpy.float64)

    def test_first_pass_f0(self):
        with tempfile.TemporaryDirectory() as directory, data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS):
            path = os.path.join(directory, 'a.wav')
            samples = numpy.concatenate([numpy.zeros(4000), data_fixtures.buzz(0.5, 16000), numpy.zeros(4000)])
            write_wav(path, samples, 16000)
            values = ctc.first_pass_f0(path)
        self.assertEqual(values.dtype, numpy.float32)
        self.assertTrue(len(values) > 20)
        self.assertTrue(numpy.all(values > 0))
        self.assertAlmostEqual(float(numpy.median(values)), 150.0, delta=5.0)


class TestAlignment(unittest.TestCase):

    def test_ctc_viterbi(self):
        self.assertIsNone(ctc.ctc_viterbi(numpy.zeros((2, 3)), numpy.zeros(2)))
        # Two positions: blank, A, A, blank, B, blank - and ending on a phone
        low, high = numpy.log(0.05), numpy.log(0.9)
        positions = numpy.full((6, 2), low)
        positions[1:3, 0] = high
        positions[4, 1] = high
        blank = numpy.full(6, low)
        blank[[0, 3, 5]] = high
        self.assertEqual(ctc.ctc_viterbi(positions, blank), [(1, 2), (4, 4)])
        positions[5, 1] = numpy.log(0.99)
        blank[5] = low
        self.assertEqual(ctc.ctc_viterbi(positions, blank), [(1, 2), (4, 5)])

    def test_phone_positions(self):
        tokens = frontend.tokenize('Call me.')['tokens']
        self.assertEqual(ctc.phone_positions(tokens), [(1, 'K'), (2, 'AO'), (3, 'L'), (5, 'M'), (6, 'IY')])

    def test_ctc_labels(self):
        tokens = frontend.tokenize('Call me.')['tokens']
        rate = ctc.CTC_SAMPLE_RATE
        # Speech from 0.1 s to 0.9 s of 1 s; the phones' runs from 0.2 s - the edges extend to the speech
        samples = numpy.zeros(rate)
        samples[int(0.1 * rate):int(0.9 * rate)] = data_fixtures.buzz(0.8, rate)
        runs = [(10, 13), (15, 20), (22, 25), (28, 31), (33, 36)]
        labels, confidences, score = ctc.ctc_labels(tokens, run_posteriors(tokens, runs, 50), VOCAB, samples)
        self.assertEqual([label for label, _, _ in labels], ['pau', 'K', 'AO1', 'L', 'M', 'IY0', 'pau'])
        self.assertAlmostEqual(labels[1][1], 0.1, delta=0.011)
        self.assertAlmostEqual(labels[-2][2], 0.9, delta=0.02)
        self.assertAlmostEqual(labels[2][1], (14 + 15) / 2 * 0.02 - 0.005)
        for previous, following in zip(labels, labels[1:]):
            self.assertAlmostEqual(previous[2], following[1])
        self.assertTrue(numpy.allclose(confidences, 1.0))
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_ctc_labels_edges(self):
        # Speech to the end; the first run at frame 0; a phone's start pushed past its predecessor's (5 ms apart)
        tokens = frontend.tokenize('Call me.')['tokens']
        rate = ctc.CTC_SAMPLE_RATE
        samples = data_fixtures.buzz(0.5, rate)
        runs = [(0, 3), (4, 4), (5, 9), (10, 15), (16, 24)]
        labels, _, _ = ctc.ctc_labels(tokens, run_posteriors(tokens, runs, 25), VOCAB, samples)
        self.assertEqual([label for label, _, _ in labels], ['K', 'AO1', 'L', 'M', 'IY0'])
        self.assertEqual(labels[0][1], 0.0)
        self.assertEqual(labels[-1][2], 0.5)
        # Speech ending before the last run (its end stays), a trailing pause
        samples[int(0.3 * rate):] = 0.0
        labels, _, _ = ctc.ctc_labels(tokens, run_posteriors(tokens, runs, 25), VOCAB, samples)
        self.assertEqual(labels[-1][0], 'pau')
        self.assertAlmostEqual(labels[-2][2], 25 * 0.02 - 0.005)
        # Too few frames
        self.assertIsNone(ctc.ctc_labels(tokens, run_posteriors(tokens, [], 3), VOCAB, samples))

    def test_token_keys(self):
        self.assertEqual(ctc.TOKEN_KEYS['s'], {'S', 'Z'})
        self.assertEqual(ctc.TOKEN_KEYS['ɪɹ'], {'IH', 'IY', 'R'})

    def test_greedy_phone_error(self):
        text = 'Don\'t forget a jacket.'
        tokens = frontend.tokenize(text)['tokens']
        count = len(ctc.phone_positions(tokens))
        self.assertEqual(ctc.greedy_phone_error(greedy_of(text), tokens), (0.0, 0))
        self.assertEqual(ctc.greedy_phone_error(greedy_of(text, tail=4), tokens), (4 / count, 4))
        # A misheard token (zz) and the tail cut short
        error, tail = ctc.greedy_phone_error('d oʊ n t zz ɡ ɛ t', tokens)
        self.assertGreater(error, 0)
        self.assertEqual((count, tail), (15, 6))
        self.assertEqual(ctc.greedy_phone_error('', tokens), (1.0, count))

    def test_greedy_decode(self):
        posteriors = numpy.full((7, len(VOCAB)), -10.0)
        for frame, token in enumerate(['<pad>', 'k', 'k', '<pad>', 'k', '<s>', 'ɔː']):
            posteriors[frame, VOCAB[token]] = 0.0
        self.assertEqual(ctc.greedy_decode(posteriors, VOCAB), 'k k ɔː')

    def test_label_durations(self):
        tokens = frontend.tokenize('Call me.')['tokens']
        labels, seconds = clip_labels('Call me.')
        samples = data_fixtures.labels_audio([tuple(label) for label in labels], seconds, 16000)
        durations, report = ctc.label_durations(tokens, [tuple(label) for label in labels], samples, 16000, {})
        self.assertAlmostEqual(sum(durations), seconds)
        self.assertEqual(report['exact'], 5)
        self.assertIn('pausesRefined', report)
        self.assertIn('pauseSecondsMoved', report)


class TestVote(unittest.TestCase):

    def test_phone_token_ids(self):
        self.assertEqual(ctc.phone_token_ids('IY1', VOCAB), [VOCAB[token] for token in ctc.STRICT_TOKENS['IY']])
        # An unstressed vowel may be reduced (its reductions once each)
        unstressed = ctc.phone_token_ids('IY0', VOCAB)
        self.assertIn(VOCAB['ə'], unstressed)
        self.assertEqual(len(unstressed), len(set(unstressed)))
        self.assertEqual(ctc.phone_token_ids('Z', {'z': 3}), [3])
        # Voicing - S and Z only their own; other phones unchanged
        self.assertEqual(ctc.phone_token_ids('Z', VOCAB), [VOCAB['z'], VOCAB['s']])
        self.assertEqual(ctc.phone_token_ids('Z', VOCAB, voicing=True), [VOCAB['z']])
        self.assertEqual(ctc.phone_token_ids('S', VOCAB, voicing=True), [VOCAB['s']])
        self.assertEqual(ctc.phone_token_ids('T', VOCAB, voicing=True), ctc.phone_token_ids('T', VOCAB))

    def test_ctc_log_likelihood(self):
        # Hobbes spoken without the vowel - the label (hobbies) less likely
        posteriors = spelled(['h', 'ɑː', 'b', 'z'])
        label = ctc.ctc_log_likelihood(posteriors, 'HH AA1 B IY0 Z'.split(), VOCAB)
        alternative = ctc.ctc_log_likelihood(posteriors, 'HH AA1 B Z'.split(), VOCAB)
        self.assertGreater(alternative, label + ctc.VOTE_MARGIN)
        self.assertEqual(ctc.ctc_log_likelihood(posteriors[:2], 'HH AA1 B Z'.split(), VOCAB), -math.inf)

        # Voicing - S and Z alike unless strict
        voiced = ctc.ctc_log_likelihood(posteriors, 'HH AA1 B Z'.split(), VOCAB, {3})
        unvoiced = ctc.ctc_log_likelihood(posteriors, 'HH AA1 B S'.split(), VOCAB, {3})
        self.assertAlmostEqual(alternative, ctc.ctc_log_likelihood(posteriors, 'HH AA1 B S'.split(), VOCAB))
        self.assertGreater(voiced, unvoiced + ctc.VOTE_MARGIN)
        self.assertLess(voiced, alternative)

    def test_candidate_log_likelihood(self):
        # An r-colored token of one frame - the vowel and its R as one
        posteriors = spelled(['s', 't', 'ɑːɹ', 't', 's'], frames=1)
        full = ctc.ctc_log_likelihood(posteriors, 'S T AA1 R T S'.split(), VOCAB)
        either = ctc.candidate_log_likelihood(posteriors, ['S'], 'T AA1 R T'.split(), ['S'], VOCAB)
        self.assertGreater(either, full + ctc.VOTE_MARGIN)
        self.assertGreater(either, ctc.candidate_log_likelihood(posteriors, ['S'], 'T ER1 T'.split(), ['S'], VOCAB))
        # No vowel and R - the phones alone
        self.assertEqual(ctc.candidate_log_likelihood(posteriors, [], 'S T ER1 T S'.split(), [], VOCAB),
                         ctc.ctc_log_likelihood(posteriors, 'S T ER1 T S'.split(), VOCAB))
        # An R at the start isn't merged
        self.assertEqual(ctc.candidate_log_likelihood(posteriors, [], 'R T S'.split(), [], VOCAB),
                         ctc.ctc_log_likelihood(posteriors, 'R T S'.split(), VOCAB))

        # Voicing - the candidate's positions (after its context), through the R merge too
        posteriors = spelled(['k', 'ɑːɹ', 'z', 't'], frames=1)
        cars = ctc.candidate_log_likelihood(posteriors, ['K'], 'AA1 R Z'.split(), ['T'], VOCAB, voicing=[2])
        carse = ctc.candidate_log_likelihood(posteriors, ['K'], 'AA1 R S'.split(), ['T'], VOCAB, voicing=[2])
        self.assertGreater(cars, carse + ctc.VOTE_MARGIN)
        self.assertAlmostEqual(cars, ctc.ctc_log_likelihood(posteriors, 'K AA1 Z T'.split(), VOCAB, {2}))
        self.assertAlmostEqual(ctc.candidate_log_likelihood(posteriors, ['K'], 'AA1 R Z'.split(), ['T'], VOCAB),
                               ctc.candidate_log_likelihood(posteriors, ['K'], 'AA1 R S'.split(), ['T'], VOCAB))

    def test_vote(self):
        self.assertEqual(ctc.vote([-10.0, -5.0]), 1)
        self.assertEqual(ctc.vote([-1.0, -9.0, -4.0]), 0)
        self.assertIsNone(ctc.vote([-1.0, -1.5]))

    def test_read_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'one.wav')
            write_wav(path, numpy.zeros((24000, 2)), 24000)
            self.assertEqual(len(ctc.read_segment(path, 0.25, 0.75)), 8000)

    def test_merged(self):
        self.assertEqual(ctc.merged('K AA1 T'), ctc.merged('K AO1 T'))
        self.assertEqual(ctc.merged('K AA1 T'), 'K A* T')
        # AE, EH, and EY before R
        self.assertEqual(ctc.merged('M EH1 R IY0'), 'M E* R IY')
        self.assertEqual(ctc.merged('M AE1 R IY0'), ctc.merged('M EY1 R IY0'))
        self.assertEqual(ctc.merged('B EH1 T'), 'B EH T')
        # IH and IY before R (the recognizer's ɪɹ)
        self.assertEqual(ctc.merged('HH IY1 R'), 'HH I* R')
        self.assertEqual(ctc.merged('HH IH1 R'), ctc.merged('HH IY1 R'))
        self.assertEqual(ctc.merged('B IY1 T'), 'B IY T')
        # The reduced vowels one schwa
        self.assertEqual(ctc.merged('AH0 B IH0 T AH1'), '@ B @ T AH')
