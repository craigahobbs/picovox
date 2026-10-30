# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import math
import unittest

import numpy
import torch

from picovox_train import losses, model, params

from .helpers import tiny_config


class TestLosses(unittest.TestCase):

    def test_priors(self):
        self.assertEqual(set(losses.PRIORS['female']), set(losses.PRIORS['male']))
        self.assertEqual(losses.F_INDEX, [params.INDEX[f'f{number}'] for number in range(1, 7)])

    def test_f0_track(self):
        f0 = numpy.array([0.0, 100.0, 0.0, 200.0, 0.0])
        voiced = numpy.array([False, True, False, True, False])
        track = losses.f0_track(f0, voiced, 5, 50.0, 400.0)
        self.assertEqual(track.dtype, numpy.float32)
        self.assertEqual(len(track), 11)
        # Held at the ends, log-interpolated between
        self.assertAlmostEqual(float(track[0]), 100.0, places=3)
        self.assertAlmostEqual(float(track[-1]), 200.0, places=3)
        self.assertAlmostEqual(float(track[5]), math.sqrt(100 * 200), places=2)
        # Within the speaker's range
        self.assertEqual(float(losses.f0_track(f0, voiced, 5, 150.0, 180.0).min()), 150.0)
        # Too few voiced frames - the range's geometric middle
        flat = losses.f0_track(f0, numpy.zeros(5, dtype=bool), 5, 100.0, 400.0)
        self.assertTrue(numpy.allclose(flat, 200.0))

    def test_spectral_loss(self):
        generator = torch.Generator().manual_seed(0)
        target = torch.randn(2, 4096, generator=generator)
        predicted = target + 0.1 * torch.randn(2, 4096, generator=generator)
        same = losses.spectral_loss(target, target, 16000, 7000.0)
        self.assertLess(float(same), 1e-3)
        loss = losses.spectral_loss(predicted, target, 16000, 7000.0)
        self.assertGreater(float(loss), float(same))
        without_mel = losses.spectral_loss(predicted, target, 16000, 7000.0, fft_sizes=(512,), mel_weight=0.0)
        self.assertLess(float(without_mel), float(loss))
        # The mel banks are cached per rate, size, band, and device
        self.assertIs(losses.mel_bank(16000, 1024, 7000.0, 'cpu'), losses.mel_bank(16000, 1024, 7000.0, 'cpu'))
        self.assertEqual(losses.mel_bank(16000, 256, 7000.0, 'cpu').shape, (32, 129))

    def test_duration_loss(self):
        tokens = torch.tensor([[model.BOS, 20, model.WORD, 21, 5, model.PAD]])
        durations = torch.tensor([[2, 3, 0, 4, 5, 0]])
        target = torch.log1p(durations.float())
        outputs = torch.stack([target, torch.where(durations > 0, 20.0, -20.0), target], dim=-1)
        phone, pause, paused = losses.duration_loss(outputs, tokens, durations)
        self.assertLess(float(phone), 1e-9)
        self.assertLess(float(pause), 1e-6)
        self.assertLess(float(paused), 1e-9)
        wrong = torch.zeros_like(outputs)
        phone, pause, paused = losses.duration_loss(wrong, tokens, durations, edge_weight=1.0)
        self.assertAlmostEqual(float(phone), float((target[0, [1, 3]] ** 2).mean()), places=6)
        self.assertAlmostEqual(float(pause), math.log(2), places=5)
        self.assertAlmostEqual(float(paused), float((target[0, [0, 4]] ** 2).mean()), places=5)

    def test_silence(self):
        silence, mask = losses._silence('cpu')  # pylint: disable=protected-access
        self.assertEqual(float(silence[params.INDEX['av']]), params.FLOOR_DB)
        self.assertIs(losses._silence('cpu')[0], silence)  # pylint: disable=protected-access
        self.assertEqual(float(mask.sum()), 3)

    def test_render_values(self):
        torch.manual_seed(0)
        network = model.MultiModel(tiny_config())
        unit = torch.rand(2, 6, params.COUNT)
        f0 = torch.full((2, 7), 123.0)
        batch = {'frameCounts': [3, 2], 'f0': f0}
        values = losses.render_values(network, unit, batch)
        self.assertEqual(values.shape, (2, 7, params.COUNT))
        # The end frame, silence past each utterance, the recorded F0
        self.assertTrue(torch.equal(values[0, 6, params.INDEX['f1']], values[0, 5, params.INDEX['f1']]))
        self.assertEqual(float(values[1, 5, params.INDEX['av']]), params.FLOOR_DB)
        self.assertNotEqual(float(values[1, 4, params.INDEX['av']]), params.FLOOR_DB)
        self.assertTrue(torch.all(values[..., params.INDEX['f0']] == 123.0))
        # The batch's render counts on the device, and the model's F0
        own = losses.render_values(network, unit, {**batch, 'renderCounts': torch.tensor([6, 4])}, teacher_f0=False)
        self.assertTrue(torch.allclose(own[..., params.INDEX['f1']], values[..., params.INDEX['f1']]))
        self.assertFalse(torch.all(own[..., params.INDEX['f0']] == 123.0))
