# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import os
import tempfile
import unittest
import wave

import numpy

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
            single = read_mono(path, 16000, dtype='float32', quality='HQ')
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
