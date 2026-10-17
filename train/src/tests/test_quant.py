# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest

import numpy
import torch

from picovox_train import quant


class TestQuant(unittest.TestCase):

    def test_qmax(self):
        self.assertEqual(quant.qmax(8), 127)
        self.assertEqual(quant.qmax(4), 7)

    def test_is_quantized(self):
        self.assertTrue(quant.is_quantized(torch.zeros(32, 32)))
        self.assertFalse(quant.is_quantized(torch.zeros(1024)))
        self.assertFalse(quant.is_quantized(torch.zeros(4, 4)))
        self.assertFalse(quant.is_quantized(torch.zeros(32, 32), 'eq_free'))
        self.assertTrue(quant.is_int8(8, 0))
        self.assertFalse(quant.is_int8(8, 32))
        self.assertFalse(quant.is_int8(6, 0))

    def test_group_size(self):
        self.assertEqual(quant.group_size(100, 0), 100)
        self.assertEqual(quant.group_size(100, 32), 32)
        self.assertEqual(quant.group_size(100, 200), 100)

    def test_quantize_int8(self):
        weights = numpy.random.default_rng(1).normal(size=(4, 3, 5))
        weights[2] = 0
        values, scales = quant.quantize_array(weights)
        self.assertEqual(values.shape, (4, 15))
        self.assertEqual(scales.shape, (4, 1))
        self.assertEqual(int(numpy.abs(values).max()), 127)
        # A row of zeros has scale 1; the scales are float32's
        self.assertEqual(float(scales[2, 0]), 1.0)
        self.assertTrue((scales == scales.astype(numpy.float32)).all())
        restored = quant.dequantize_array(values, scales, weights.shape)
        self.assertEqual(restored.dtype, numpy.float32)
        self.assertLess(numpy.abs(restored - weights).max(), numpy.abs(weights).max() / 127)

    def test_quantize_packed(self):
        weights = numpy.random.default_rng(2).normal(size=(3, 10))
        values, scales = quant.quantize_array(weights, bits=4, group=4)
        self.assertEqual(values.shape, (3, 10))
        self.assertEqual(scales.shape, (3, 3))
        self.assertLessEqual(int(numpy.abs(values).max()), 7)
        # The scales are float16's
        self.assertTrue((scales == scales.astype(numpy.float16)).all())
        restored = quant.dequantize_array(values, scales, weights.shape, group=4)
        self.assertLess(numpy.abs(restored - weights).max(), numpy.abs(weights).max() / 7 * 0.51 + 1e-3)
        # Tiny weights whose float16 scale rounds to zero get scale 1
        _, tiny_scales = quant.quantize_array(numpy.full((1, 4), 1e-12), bits=6)
        self.assertEqual(float(tiny_scales[0, 0]), 1.0)

    def test_pack(self):
        for bits in (3, 4, 5, 6, 7, 8):
            values = numpy.random.default_rng(bits).integers(-quant.qmax(bits), quant.qmax(bits) + 1, size=37)
            data = quant.pack(values, bits)
            self.assertEqual(len(data), (37 * bits + 7) // 8)
            self.assertEqual(quant.unpack(data, bits, 37).tolist(), values.tolist())
        # Little-endian, LSB first: 0 at 4 bits is 8 (0b1000)
        self.assertEqual(quant.pack([0, 1], 4), bytes([0x98]))

    def test_fake_quantize(self):
        weights = torch.randn(6, 40, generator=torch.Generator().manual_seed(3), requires_grad=True)
        for bits, group in ((8, 0), (4, 16), (6, 0)):
            fake = quant.fake_quantize(weights, bits, group)
            values, scales = quant.quantize_array(weights.detach(), bits, group)
            expected = quant.dequantize_array(values, scales, weights.shape, group)
            self.assertTrue(numpy.allclose(fake.detach().numpy(), expected, atol=1e-6))
            # The straight-through estimator
            fake.sum().backward()
            self.assertTrue(torch.equal(weights.grad, torch.ones_like(weights)))
            weights.grad = None
        # A row of zeros
        zeros = quant.fake_quantize(torch.zeros(2, 8), 4, 0)
        self.assertEqual(float(zeros.abs().sum()), 0.0)

    def test_fake_quantized(self):
        model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.LayerNorm(32))
        weight = model[0].weight
        with quant.fake_quantized(model, bits=4):
            swapped = model[0].weight
            self.assertIsNot(swapped, weight)
            self.assertIs(model[0].bias, model[0].bias)
            self.assertEqual(int(torch.unique(swapped.detach()[0]).numel() <= 15), 1)
            output = model(torch.ones(1, 64))
        self.assertIs(model[0].weight, weight)
        output.sum().backward()
        self.assertIsNotNone(weight.grad)
        # A top-level parameter (no module name), and a parameter set to None
        holder = torch.nn.Module()
        holder.table = torch.nn.Parameter(torch.randn(40, 40))
        holder.register_parameter('missing', None)
        with quant.fake_quantized(holder):
            self.assertIsNot(holder.table, holder._parameters['missing'])  # pylint: disable=protected-access
        self.assertIsInstance(holder.table, torch.nn.Parameter)
