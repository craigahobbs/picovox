# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock

import numpy
import torch

from picovox_train import params

from .helpers import captured


class TestParams(unittest.TestCase):

    def test_parameters(self):
        self.assertEqual(params.COUNT, 32)
        self.assertEqual(params.NAMES[params.INDEX['f1']], 'f1')
        self.assertEqual(params.DEFAULTS['av'], params.FLOOR_DB)
        self.assertNotIn('shelf', params.DB_NAMES)
        self.assertIn('av', params.DB_NAMES)

    def test_normalize_numpy(self):
        defaults = numpy.array([[params.DEFAULTS[name] for name in params.NAMES]])
        unit = params.normalize(defaults)
        self.assertIsInstance(unit, numpy.ndarray)
        self.assertEqual(unit.shape, (1, params.COUNT))
        self.assertTrue(((unit >= 0) & (unit <= 1)).all())
        self.assertTrue(numpy.allclose(params.denormalize(unit), defaults))
        # The range's ends map to 0 and 1
        lows = numpy.array([parameter[4] for parameter in params.PARAMETERS])
        highs = numpy.array([parameter[5] for parameter in params.PARAMETERS])
        self.assertTrue(numpy.allclose(params.normalize(lows), 0))
        self.assertTrue(numpy.allclose(params.normalize(highs), 1))

    def test_normalize_torch(self):
        unit = torch.rand(2, 3, params.COUNT, dtype=torch.float64)
        values = params.denormalize(unit)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(values.shape, (2, 3, params.COUNT))
        self.assertTrue(torch.allclose(params.normalize(values), unit))

    def test_bandwidth_scale(self):
        values = numpy.array([[params.DEFAULTS[name] for name in params.NAMES]])
        scaled = params.bandwidth_scale(values, 0.5)
        self.assertEqual(scaled[0, params.INDEX['b1']], 40.0)
        self.assertEqual(scaled[0, params.INDEX['b6']], values[0, params.INDEX['b6']])
        self.assertEqual(values[0, params.INDEX['b1']], 80.0)
        tensor = torch.tensor(values)
        scaled_tensor = params.bandwidth_scale(tensor, 2.0, names=('b6',))
        self.assertEqual(float(scaled_tensor[0, params.INDEX['b6']]), 2000.0)
        self.assertEqual(float(tensor[0, params.INDEX['b6']]), 1000.0)

    def test_write_params(self):
        frames = [[params.DEFAULTS[name] + 0.123456 for name in params.NAMES] + [99.0]] * 2
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'frames.json')
            params.write_params(path, frames)
            with open(path, encoding='utf-8') as file:
                written = json.load(file)
            self.assertEqual(written['frameRate'], 200)
            self.assertEqual(written['names'], params.NAMES)
            self.assertEqual(len(written['frames'][0]), params.COUNT)
            self.assertEqual(written['frames'][0][0], round(100.123456, 4))
            self.assertNotIn('eq', written)
            params.write_params(path, frames, eq=[1.23456, -2.0])
            with open(path, encoding='utf-8') as file:
                self.assertEqual(json.load(file)['eq'], [1.235, -2.0])

    def test_javascript_current(self):
        # lib/params.js is generated from the parameter set - and current
        with open(params.JS_PATH, encoding='utf-8') as file:
            self.assertEqual(file.read(), params.javascript())

    def test_javascript(self):
        text = params.javascript()
        self.assertIn("export const FRAME_RATE = 200;", text)
        self.assertIn("export const DB_NAMES = ['av', 'ah', 'af', 'ap1', 'ap2', 'ab'];", text)
        self.assertIn("'doc': 'The fundamental frequency'},", text)
        self.assertTrue(text.endswith('\n'))
        # The literal quoting
        self.assertEqual(params._js("it's \\ok"), "'it\\'s \\\\ok'")  # pylint: disable=protected-access
        self.assertEqual(params._js((1, 'a')), "[1, 'a']")  # pylint: disable=protected-access

    def test_main(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'params.js')
            with unittest.mock.patch.object(params, 'JS_PATH', path):
                self.assertEqual(params.main(['--write-js']), 0)
            with open(path, encoding='utf-8') as file:
                self.assertEqual(file.read(), params.javascript())

    def test_main_module(self):
        # python -m picovox_train.params (without --write-js - the usage error)
        with unittest.mock.patch.dict(sys.modules), unittest.mock.patch('sys.argv', ['params']), captured(), \
                self.assertRaises(SystemExit) as context:
            del sys.modules['picovox_train.params']
            runpy.run_module('picovox_train.params', run_name='__main__')
        self.assertEqual(context.exception.code, 2)

    def test_main_usage(self):
        with captured() as (_, stderr), self.assertRaises(SystemExit) as context:
            params.main([])
        self.assertEqual(context.exception.code, 2)
        self.assertIn('--write-js', stderr.getvalue())
