# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import math
import unittest

import numpy
import torch

from picovox_train import spectral


class TestSpectral(unittest.TestCase):

    def test_stft(self):
        rate = 16000
        audio = torch.sin(2 * math.pi * 1000 * torch.arange(4096) / rate)[None].double()
        spectrum = spectral.stft(audio, 512, 128)
        self.assertEqual(spectrum.shape, (1, 257, 33))
        self.assertTrue(spectrum.is_complex())
        magnitude = spectral.stft_magnitude(audio, 512, 128)
        self.assertEqual(magnitude.shape, (1, 33, 257))
        # The peak is at 1 kHz (bin 32)
        self.assertEqual(int(magnitude[0, 16].argmax()), 32)
        self.assertTrue(torch.allclose(spectral.spectrum_magnitude(spectrum), magnitude))
        # Without the epsilon, silence is exactly zero
        silence = spectral.stft_magnitude(torch.zeros(1, 1024), 256, 64, eps=0)
        self.assertEqual(float(silence.abs().max()), 0.0)

    def test_mel_filterbank(self):
        bank = spectral.mel_filterbank(16000, 512)
        self.assertEqual(bank.shape, (80, 257))
        self.assertTrue((bank >= 0).all())
        self.assertIs(spectral.mel_filterbank(16000, 512), bank)
        # The bands are increasingly wide, and stop below the high edge
        limited = spectral.mel_filterbank(16000, 512, mels=32, low=40.0, high=4000.0)
        self.assertEqual(limited.shape, (32, 257))
        self.assertEqual(float(limited[:, 130:].sum()), 0.0)
        centers = numpy.argmax(limited, axis=1)
        self.assertTrue((numpy.diff(centers) >= 0).all())
