# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest

import numpy
import torch

from picovox_train import quant


class TestQuant(unittest.TestCase):

    def test_is_quantized(self):
        self.assertTrue(quant.is_quantized(torch.zeros(32, 32)))
        self.assertFalse(quant.is_quantized(torch.zeros(1024)))
        self.assertFalse(quant.is_quantized(torch.zeros(4, 4)))
        self.assertFalse(quant.is_quantized(torch.zeros(32, 32), 'eq_free'))
        self.assertFalse(quant.is_quantized(torch.zeros(32, 32), 'eq_table'))

    def test_quantize(self):
        tensor = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]])
        values, scales = quant.quantize(tensor)
        self.assertEqual(values.dtype, numpy.int8)
        self.assertEqual(values.tolist(), [[127, -64, 32], [0, 0, 0]])
        self.assertEqual(scales.dtype, numpy.float32)
        self.assertAlmostEqual(float(scales[0]), 1 / 127, places=7)
        # A row of zeros has scale 1
        self.assertEqual(float(scales[1]), 1.0)

    def test_dequantize(self):
        weights = torch.tensor(numpy.random.default_rng(1).normal(size=(4, 3, 5)), dtype=torch.float32)
        values, scales = quant.quantize(weights)
        self.assertEqual(values.shape, (4, 15))
        restored = quant.dequantize(values, scales, weights.shape)
        self.assertEqual(restored.dtype, numpy.float32)
        self.assertEqual(restored.shape, (4, 3, 5))
        self.assertLess(numpy.abs(restored - weights.numpy()).max(), float(weights.abs().max()) / 127)
