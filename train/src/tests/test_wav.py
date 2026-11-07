# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import math
import os
import tempfile
import unittest
import wave

import numpy
import torch

from picovox_train import wav
from picovox_train.wav import read_mono, read_wav
from tests import data_fixtures
from tests.helpers import write_wav


class TestWav(unittest.TestCase):

    def test_write_read(self):
        # Written as libsndfile writes (round(x * 2^31) >> 16 - so 774.75 is 774, -2.0000033 is -2), clipped; read as an
        # integer over 32768
        samples = numpy.array([0.0, 0.5, -0.5, 774.75 / 32768, -2.0000033 / 32768, 32767.5 / 32768, 1.0, -1.0, 1.5, -1.5])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            write_wav(path, samples, 24000)
            with wave.open(path) as file:
                self.assertEqual((file.getnchannels(), file.getsampwidth(), file.getframerate()), (1, 2, 24000))
                integers = numpy.frombuffer(file.readframes(file.getnframes()), dtype='<i2')
            read, rate = read_wav(path)
            read32, _ = read_wav(path, dtype='float32')
        self.assertEqual(integers.tolist(), [0, 16384, -16384, 774, -2, 32767, 32767, -32768, 32767, -32768])
        self.assertEqual((read.shape, read.dtype, rate, read32.dtype), ((10, 1), numpy.float64, 24000, numpy.float32))
        self.assertTrue(numpy.array_equal(read[:, 0], integers / 32768))
        self.assertTrue(numpy.array_equal(read32[:, 0], (integers / 32768).astype(numpy.float32)))

    def test_stereo(self):
        samples = numpy.stack([numpy.linspace(-0.5, 0.5, 100), numpy.zeros(100)], axis=1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            write_wav(path, samples, 8000)
            read, rate = read_wav(path)
        self.assertEqual((read.shape, rate), ((100, 2), 8000))
        self.assertTrue(numpy.allclose(read, samples, rtol=0, atol=1 / 32768))
        self.assertTrue(numpy.array_equal(read[:, 1], numpy.zeros(100)))

    def test_read_mono(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            samples = data_fixtures.buzz(0.5, 16000)
            write_wav(path, samples, 16000)
            self.assertTrue(numpy.allclose(read_mono(path, 16000), samples, rtol=0, atol=1 / 32768))
            # Stereo at another rate - mixed down and resampled; float32; a segment
            write_wav(path, numpy.stack([samples[::2], samples[::2]], axis=1), 8000)
            resampled = read_mono(path, 16000)
            single = read_mono(path, 16000, dtype='float32')
            segment = read_mono(path, 16000, start=0.125, end=0.375)
        self.assertEqual(len(resampled), 8000)
        self.assertTrue(resampled.flags['C_CONTIGUOUS'])
        self.assertEqual(resampled.dtype, numpy.float64)
        self.assertEqual((len(single), single.dtype), (8000, numpy.float32))
        self.assertEqual(len(segment), 4000)

    def test_not_16_bit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            with wave.Wave_write(path) as file:
                file.setnchannels(1)
                file.setsampwidth(1)
                file.setframerate(8000)
                file.writeframes(bytes(10))
            with self.assertRaisesRegex(ValueError, '8-bit samples - only 16-bit PCM is read'):
                read_wav(path)


class TestResample(unittest.TestCase):

    def test_resample(self):
        # A tone at 24 kHz resampled to 16 kHz is the tone sampled at 16 kHz (the filter's edges aside); its length is
        # the ceiling of the rate ratio's; the kernels are made once per rates, zeros, device, and dtype
        times = torch.arange(24000, dtype=torch.float32) / 24000
        audio = torch.sin(2 * math.pi * 440 * times)[None].repeat(2, 1)
        resampled = wav.resample(audio, 24000, 16000)
        self.assertEqual(resampled.shape, (2, 16000))
        expected = torch.sin(2 * math.pi * 440 * torch.arange(16000, dtype=torch.float32) / 16000)
        self.assertTrue(torch.allclose(resampled[0, 100:-100], expected[100:-100], atol=2e-3))
        self.assertTrue(torch.equal(resampled[0], resampled[1]))
        self.assertEqual(wav.resample(torch.zeros(1, 24001), 24000, 16000).shape, (1, 16001))
        kernels, width, orig, new = wav.resample_kernel(24000, 16000, torch.device('cpu'), torch.float32)
        self.assertEqual((orig, new, width), (3, 2, 10))
        self.assertEqual(kernels.shape, (2, 1, 2 * width + orig))
        self.assertIs(kernels, wav.resample_kernel(24000, 16000, torch.device('cpu'), torch.float32)[0])
        # (the long filter - the corpora's - closer to the tone, in float64 too)
        long = wav.resample(audio.double(), 24000, 16000, wav.READ_ZEROS)
        self.assertEqual(wav.resample_kernel(24000, 16000, torch.device('cpu'), torch.float64, wav.READ_ZEROS)[1], 97)
        self.assertLess(float((long[0, 500:-500] - expected[500:-500].double()).abs().max()),
                        float((resampled[0, 500:-500] - expected[500:-500]).abs().max()))
        # (a constant stays a constant - the kernels sum to the rate change's gain)
        constant = wav.resample(torch.ones(1, 4800), 24000, 16000)
        self.assertTrue(torch.allclose(constant[0, 50:-50], torch.ones(3100), atol=1e-3))
        # (differentiable)
        audio.requires_grad_(True)
        wav.resample(audio, 24000, 16000).square().sum().backward()
        self.assertGreater(float(audio.grad.abs().sum()), 0.0)
