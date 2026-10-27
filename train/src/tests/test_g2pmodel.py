# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring, protected-access

import base64
import math
import os
import tempfile
import unittest
import unittest.mock
import zlib

import numpy as np
import torch

from picovox_train import context, g2p, g2pmodel

from . import parity


# Uniform values in [low, high) - Generator.uniform's low + scale * value is fused (FMA) on some platforms, and the
# values are the parity check's inputs
def uniform(rng, low, high, count):
    return low + (high - low) * rng.random(count)


# Test values for exp and tanh - every range of fdlibm's reductions and special cases
def sample_values():
    rng = np.random.default_rng(7)
    special = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.3, -0.3, 2.0 ** -30, -(2.0 ** -30), 2.0 ** -60, 0.34657359,
               0.35, -0.35, 1.03972077, 1.04, -1.04, 7.09782712893383973096e+02, 709.79, -745.13321910194110842, -745.2,
               -708.4, -720.0, 710.0, 22.0, -22.0, 23.0, 40.0, -40.0, 56.0 * math.log(2), -2.0, 1e-300, -1e-300, 5e-324,
               np.inf, -np.inf, np.nan, 0.6931471805599453, -0.6931471805599453, 1.5, 3.0, 20.0, -0.9, 0.9]
    ranges = [uniform(rng, -1, 1, 400), uniform(rng, -10, 10, 400), uniform(rng, -40, 40, 400), uniform(rng, -750, 720, 400),
              uniform(rng, -1e-8, 1e-8, 100), rng.normal(0, 3, 400)]
    return np.concatenate([np.array(special)] + ranges)


# A tiny G2P model with its reading heads and context network, as context.py exports it - (the module text, the G2P's
# quantized tensors)
def tiny_module():
    torch.manual_seed(3)
    g2p_model = context.add_reading_heads(g2p.build_model({'embed': 4, 'hidden': 4, 'decoder': 6}))
    context_model = context.build_context({'embed': 4, 'hidden': 4, 'layer': 4})
    quantized = g2p.quantize_state(g2p_model.state_dict(), context.EXPORT)
    context_quantized = g2p.quantize_state(context_model.state_dict(), context.EXPORT)
    return context.module_text(g2p_model.config, quantized, context_model.config, context_quantized, ['test']), quantized


class TestG2pModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g2p = g2pmodel.G2p(g2pmodel.read_module())

    def test_parse_module(self):
        model = g2pmodel.read_module()
        self.assertEqual(model['design'], 'ar')
        self.assertTrue(model['deflate'])
        self.assertEqual(model['layout'], 'bytes')
        self.assertIn('weights', model['context'])
        self.assertFalse('`' in model['weights'] or '`' in model['context']['weights'])
        model = g2pmodel.parse_module(tiny_module()[0])
        self.assertEqual((model['hidden'], model['decoder'], model['readings']), (4, 6, 3))
        self.assertFalse('`' in model['weights'] or '`' in model['context']['weights'])

    def test_unpack_tensors(self):
        # Every layout, with and without deflate - the weights are the quantized weights
        _, quantized = tiny_module()
        manifest = [{'name': item['name'], 'shape': item['shape'], 'bits': item['bits']} for item in quantized]
        for layout in ('packed', 'bytes'):
            for deflate in (False, True):
                data = g2p.weight_bytes(quantized, layout)
                if deflate:
                    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
                    data = compressor.compress(data) + compressor.flush()
                model = {'weights': base64.b64encode(data).decode('ascii'), 'layout': layout, 'deflate': deflate, 'tensors': manifest}
                tensors = g2pmodel.unpack_tensors(model)
                for item in quantized:
                    self.assertTrue(np.array_equal(tensors[item['name']], item['dequantized']), (layout, deflate, item['name']))

    # (math.fma is Python 3.13's)
    @unittest.skipUnless(hasattr(math, 'fma'), 'math.fma is Python 3.13+')
    def test_fma(self):
        rng = np.random.default_rng(1)
        first, second, addend = (rng.normal(size=2000) * 10.0 ** rng.integers(-5, 5, 2000) for _ in range(3))
        expected = np.array([math.fma(a, b, c) for a, b, c in zip(first, second, addend)])
        self.assertTrue(np.array_equal(g2pmodel._fma(first, second, addend), expected))
        with unittest.mock.patch.object(g2pmodel, 'FUSED', True):
            self.assertTrue(np.array_equal(g2pmodel._mul_add(first, second, addend), expected))
        with unittest.mock.patch.object(g2pmodel, 'FUSED', False):
            self.assertTrue(np.array_equal(g2pmodel._mul_add(first, second, addend), first * second + addend))

    def test_exp_tanh_javascript(self):
        # Math.exp and Math.tanh bit for bit (the parity check - tools/parity.js) - V8 (fdlibm fused on arm64) as this
        # machine's FUSED path
        values = sample_values()
        parity.inputs(self, 'expTanh', {'values': parity.to_hex(values)})
        outputs = parity.outputs('expTanh')
        for name, function in (('exp', g2pmodel.js_exp), ('tanh', g2pmodel.js_tanh)):
            expected = parity.from_hex(outputs[name])
            result = function(values)
            same = (result.view(np.uint64) == expected.view(np.uint64)) | (np.isnan(result) & np.isnan(expected))
            self.assertTrue(same.all(), (name, values[~same][:5], result[~same][:5], expected[~same][:5]))

    def test_exp_tanh_unfused(self):
        # The unfused path (other machines) - within an ulp of the C library's
        values = sample_values()
        with unittest.mock.patch.object(g2pmodel, 'FUSED', False):
            for function, reference in ((g2pmodel.js_exp, np.exp), (g2pmodel.js_tanh, np.tanh)):
                # (the infinities' reductions are invalid unfused - the network's values are finite)
                with np.errstate(over='ignore', invalid='ignore'):
                    result = function(values)
                    expected = reference(values)
                finite = np.isfinite(expected) & (expected != 0)
                self.assertTrue(np.all(np.abs(result[finite] - expected[finite]) <= 2 * np.spacing(np.abs(expected[finite]))))
                self.assertTrue(np.array_equal(np.isnan(result), np.isnan(expected)))
                self.assertTrue(np.array_equal(result[np.isinf(expected)], expected[np.isinf(expected)]))

    def test_pronounce_javascript(self):
        # The committed module pronounces as lib/g2p.js does (the parity check - tools/parity.js)
        words = ['hello', 'world', 'ishmael', 'read', 'lead', 'zxqv', 'antidisestablishmentarianism', 'a', 'colonel',
                 'gutenberg', 'wouldn\'t', 'xylophonists', 'q', 'abcdefghijklmnopqrstuvwxyzabcdefghij']
        parity.inputs(self, 'pronounce', {'words': words})
        self.assertEqual(g2pmodel.G2p(g2pmodel.read_module()).pronounce_many(words), parity.outputs('pronounce'))

    def test_g2p(self):
        g2p_model = self.g2p
        self.assertEqual(g2p_model.word_ids('ab-c'), [2, 3, 4])
        self.assertEqual(len(g2p_model.word_ids('a' * 40)), g2p_model.model['maxLetters'])

        # Words without letters are pronounced empty; the cache answers again
        self.assertEqual(g2p_model.pronounce_many(['--', 'cat']), [[], ['K', 'AE1', 'T']])
        self.assertFalse(g2p_model.ambiguous_words['--'])
        self.assertEqual(g2p_model.pronounce('cat'), ['K', 'AE1', 'T'])

        # Readings - another reading of a heteronym, and a reading the model lacks (the default)
        self.assertTrue(g2p_model.ambiguous('lead'))
        self.assertTrue(g2p_model.ambiguous('lead'))
        readings = {tuple(g2p_model.pronounce('lead', reading)) for reading in range(3)}
        self.assertGreater(len(readings), 1)
        self.assertEqual(g2p_model.pronounce('cat', 7), ['K', 'AE1', 'T'])
        self.assertFalse(g2p_model.ambiguous('dog'))

        # The context network's readings
        items = ['he', 'will', 'lead', 'the', 'way', '.']
        slot = g2p_model.choose_reading(items, 2)
        self.assertIn(slot, range(3))
        window = g2p_model.context.window_of(items, 2)
        self.assertEqual(len(window), 2 * g2p_model.context.window + 1)
        self.assertEqual(g2p_model.choose_slots([list(window)]), [slot])
        self.assertEqual(g2p_model.context.window_of(['lead'], 0), ('',) * g2p_model.context.window + ('lead',) +
                         ('',) * g2p_model.context.window)
        g2p_model.context.item_vectors(['###', 'lead'])
        self.assertFalse(g2p_model.context.vectors['###'].any())
        vector = g2p_model.context.vectors['lead']
        g2p_model.context.item_vectors(['lead'])
        self.assertIs(g2p_model.context.vectors['lead'], vector)

        # The state round-trips
        state = g2p_model.state()
        other = g2pmodel.G2p(g2pmodel.read_module())
        other.set_state(state)
        self.assertEqual(other.cache['cat'], ['K', 'AE1', 'T'])
        self.assertEqual(other.context.choices, g2p_model.context.choices)

    def test_read_module(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'g2pModel.js')
            with open(path, 'w', encoding='utf-8') as file:
                file.write(tiny_module()[0])
            self.assertEqual(g2pmodel.read_module(path)['decoder'], 6)
