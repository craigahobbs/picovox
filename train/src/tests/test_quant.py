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
        self.assertEqual((quant.BITS, quant.LIMIT), (7, 63))
        tensor = torch.tensor([[1.0, -0.5, 0.25], [0.0, 0.0, 0.0]])
        values, scales = quant.quantize(tensor)
        self.assertEqual(values.dtype, numpy.int8)
        self.assertEqual(values.tolist(), [[63, -32, 16], [0, 0, 0]])
        self.assertEqual(scales.dtype, numpy.float32)
        self.assertAlmostEqual(float(scales[0]), 1 / 63, places=7)
        # A row of zeros has scale 1
        self.assertEqual(float(scales[1]), 1.0)

    def test_quantize_gptq(self):
        # With the layer's input covariance, GPTQ's integers - the layer's output error (on its inputs) below the
        # nearest rounding's; the scales the same
        rng = numpy.random.default_rng(2)
        weights = torch.tensor(rng.normal(size=(16, 24)), dtype=torch.float32)
        inputs = rng.normal(size=(500, 24)) @ rng.normal(size=(24, 24))
        covariance = inputs.T @ inputs / len(inputs)
        nearest, nearest_scales = quant.quantize(weights)
        rounded, scales = quant.quantize(weights, covariance)
        self.assertTrue(numpy.array_equal(scales, nearest_scales))
        self.assertLessEqual(numpy.abs(rounded).max(), quant.LIMIT)
        def error(values):
            difference = quant.dequantize(values, scales, weights.shape) - weights.numpy()
            return numpy.trace(difference @ covariance @ difference.T)
        self.assertLess(error(rounded), 0.9 * error(nearest))

        # A covariance per row (a depthwise convolution's channels), and a dead input (never nonzero - its weights zero)
        kernels = torch.tensor(rng.normal(size=(3, 1, 4)), dtype=torch.float32)
        per_row = numpy.stack([numpy.eye(4)] * 3)
        per_row[:, 3, 3] = 0.0
        values, _ = quant.quantize(kernels, per_row)
        self.assertEqual(values.shape, (3, 4))
        self.assertEqual(values[:, 3].tolist(), [0, 0, 0])

    def test_gptq(self):
        # An identity covariance - the nearest rounding; a scale of 0 - a row of zeros
        rows = numpy.array([[0.26, -0.74, 1.0], [0.5, 0.5, 0.5]])
        integers = quant.gptq(rows, numpy.array([0.25, 0.0]), numpy.eye(3), 3)
        self.assertEqual(integers.tolist(), [[1, -3, 3], [0, 0, 0]])

    def test_pack(self):
        # Each value + 63 as 7 bits, little-endian bit by bit
        packed = quant.pack(numpy.array([[-63, 63], [0, 1]], dtype=numpy.int8))
        bits = numpy.unpackbits(numpy.frombuffer(packed, numpy.uint8), bitorder='little')[:28].reshape(4, 7)
        self.assertEqual(len(packed), 4)
        self.assertEqual(((bits * (1 << numpy.arange(7))).sum(1) - 63).tolist(), [-63, 63, 0, 1])

    def test_dequantize(self):
        weights = torch.tensor(numpy.random.default_rng(1).normal(size=(4, 3, 5)), dtype=torch.float32)
        values, scales = quant.quantize(weights)
        self.assertEqual(values.shape, (4, 15))
        restored = quant.dequantize(values, scales, weights.shape)
        self.assertEqual(restored.dtype, numpy.float32)
        self.assertEqual(restored.shape, (4, 3, 5))
        self.assertLess(numpy.abs(restored - weights.numpy()).max(), float(weights.abs().max()) / quant.LIMIT)
