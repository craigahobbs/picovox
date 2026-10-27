# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring, protected-access

import math
import os
import tempfile
import unittest
import unittest.mock

import numpy as np
import torch

from picovox_train import g2p, g2pmodel

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


# The module text of a tiny G2P model (g2p.py's export) in a layout, with or without deflate, group size, and the
# reading and ambiguity tensors
def tiny_module(layout='packed', deflate=False, group_size=0, readings=False):
    torch.manual_seed(3)
    model = g2p.build_model({'embed': 4, 'hidden': 4, 'decoder': 6})
    state = model.state_dict()
    if readings:
        state = {**state, 'reading.weight': torch.randn(2, 6), 'ambiguity_layer.weight': torch.randn(5, 8),
                 'ambiguity_layer.bias': torch.randn(5), 'ambiguity.weight': torch.randn(1, 5), 'ambiguity.bias': torch.tensor([0.5])}
    spec = {'bits': 6, 'biasBits': 8, 'groupSize': group_size}
    quantized = g2p.quantize_state(state, spec)
    text = g2p.module_text(model.config, quantized, spec, layout, deflate, b'', ['test'])
    data = g2p.weight_bytes(quantized, layout)
    if deflate:
        compressor = g2p.zlib.compressobj(9, g2p.zlib.DEFLATED, -15)
        data = compressor.compress(data) + compressor.flush()
    text = g2p.module_text(model.config, quantized, spec, layout, deflate, data, ['test'])
    if readings:
        text = text.replace("    'groupSize'", "    'readings': 3,\n    'groupSize'")
    return text, quantized


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
        self.assertFalse(model['deflate'])
        self.assertNotIn('context', model)

    def test_unpack_tensors(self):
        # Every layout, with and without deflate and groups - the module's weights are the quantized weights
        for layout, deflate, group_size in (('packed', False, 0), ('packed', True, 3), ('bytes', True, 0), ('bytes', False, 5)):
            text, quantized = tiny_module(layout, deflate, group_size)
            tensors = g2pmodel.unpack_tensors(g2pmodel.parse_module(text))
            for item in quantized:
                self.assertTrue(np.array_equal(tensors[item['name']], item['dequantized']), (layout, deflate, group_size, item['name']))

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

    def test_g2p_tiny(self):
        # A model without a context network or readings (g2p.py's export)
        g2p_model = g2pmodel.G2p(g2pmodel.parse_module(tiny_module()[0]))
        self.assertIsNone(g2p_model.context)
        self.assertFalse(g2p_model.ambiguous('cat'))
        phones = g2p_model.pronounce_many(['cat', 'dog'])
        self.assertEqual(len(phones), 2)
        self.assertTrue(all(len(item) <= 3 * 3 + 6 for item in phones))
        self.assertEqual(g2p_model.state()['choices'], {})
        g2p_model.set_state({'readings': [{'x': ['K']}], 'ambiguous': {'x': False}, 'choices': {}})
        self.assertEqual(g2p_model.pronounce('x'), ['K'])

        # With readings and the ambiguity head but no context network
        g2p_model = g2pmodel.G2p(g2pmodel.parse_module(tiny_module(readings=True)[0]))
        self.assertEqual(g2p_model.readings, 3)
        self.assertEqual(len(g2p_model.pronounce_many(['cat'], 2)), 1)
        self.assertFalse(g2p_model.ambiguous('cat'))

        # Another design
        model = g2pmodel.parse_module(tiny_module()[0])
        model['design'] = 'ctc'
        with self.assertRaisesRegex(ValueError, 'Unsupported G2P model design ctc'):
            g2pmodel.G2p(model)

    def test_read_module(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'g2pModel.js')
            with open(path, 'w', encoding='utf-8') as file:
                file.write(tiny_module()[0])
            self.assertEqual(g2pmodel.read_module(path)['hidden'], 4)
