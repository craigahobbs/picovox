# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import os
import tempfile
import unittest
import unittest.mock

from . import parity


class TestParity(unittest.TestCase):

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())  # pylint: disable=consider-using-with
        self.inputs = os.path.join(directory, 'parity-inputs.json')
        self.outputs = os.path.join(directory, 'parity-outputs.json')
        with open(self.inputs, 'w', encoding='utf-8') as file:
            file.write('{"b": 1, "c": {"values": [1.0, 2.0]}}')
        self.enterContext(unittest.mock.patch.object(parity, 'INPUTS', self.inputs))
        self.enterContext(unittest.mock.patch.object(parity, 'OUTPUTS', self.outputs))

    @unittest.mock.patch.dict(os.environ, {'PARITY_WRITE': '1'})
    def test_inputs_written(self):
        # Each key on its own line, sorted
        self.assertEqual(parity.inputs(self, 'a', {'values': (1.0, 2.0)}), {'values': [1.0, 2.0]})
        with open(self.inputs, encoding='utf-8') as file:
            self.assertEqual(file.read(), '{\n"a": {"values": [1.0, 2.0]},\n"b": 1,\n"c": {"values": [1.0, 2.0]}\n}\n')

    @unittest.mock.patch.dict(os.environ, {'PARITY_WRITE': ''})
    def test_inputs_checked(self):
        self.assertEqual(parity.inputs(self, 'c', {'values': (1.0, 2.0)}), {'values': [1.0, 2.0]})
        self.assertRaises(AssertionError, parity.inputs, self, 'c', {'values': [1.0]})
        self.assertRaises(AssertionError, parity.inputs, self, 'd', 1)

    @unittest.mock.patch.dict(os.environ, {'PARITY_WRITE': ''})
    def test_outputs(self):
        self.assertRaisesRegex(AssertionError, 'make parity builds it', parity.outputs, 'a')
        with open(self.outputs, 'w', encoding='utf-8') as file:
            file.write('{"a": [1, 2]}')
        self.assertEqual(parity.outputs('a'), [1, 2])

    @unittest.mock.patch.dict(os.environ, {'PARITY_WRITE': '1'})
    def test_outputs_written(self):
        # (the inputs just written - their outputs are make parity's next)
        self.assertRaises(unittest.SkipTest, parity.outputs, 'a')

    def test_hex(self):
        self.assertEqual(parity.to_hex([1.0, float('-inf')]), ['3ff0000000000000', 'fff0000000000000'])
        self.assertEqual(parity.from_hex(['3ff0000000000000', 'fff0000000000000']).tolist(), [1.0, float('-inf')])
