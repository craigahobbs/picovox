# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest

import numpy

from picovox_train import rng

from . import parity


class TestRng(unittest.TestCase):

    def test_stream_seed(self):
        # lib/rng.js's golden values (test/testRng.js)
        self.assertEqual(rng.stream_seed(1, 1), 1278455945)
        self.assertEqual(rng.stream_seed(1000, 2), 2545871315)
        self.assertEqual(rng.stream_seed(5, 3), 2274654832)
        self.assertEqual(rng.stream_seed(2 ** 32 + 5, 3), rng.stream_seed(5, 3))
        streams = {rng.stream_seed(1, stream) for stream in (rng.STREAM_ASPIRATION, rng.STREAM_FRICATION, rng.STREAM_JITTER,
                                                              rng.STREAM_SHIMMER)}
        self.assertEqual(len(streams), 4)

    def test_uniform(self):
        self.assertEqual(rng.uniform(7, numpy.arange(6)).tolist(), [
            0.011704753153026104, 0.06195825757458806, 0.97690763277933, 0.6990287057124078, 0.5214452685322613, 0.4055216880515218
        ])
        self.assertEqual(float(rng.uniform(7, numpy.array([1000000]))[0]), 0.17535980022512376)
        values = rng.uniform(99, numpy.arange(10000))
        self.assertTrue(((values >= 0) & (values < 1)).all())
        self.assertLess(abs(values.mean() - 0.5), 0.01)

    def test_gauss(self):
        self.assertEqual(float(rng.gauss(7, 0, 1)[0]), -0.4574043168616386)
        self.assertEqual(float(rng.gauss(123, 45, 1)[0]), -1.1194866375645351)
        self.assertEqual(rng.gauss(42, 10, 3).tolist(), [-0.7347769063645522, -0.1463601602974729, 1.3256268005328773])
        values = rng.gauss(5, 0, 10000)
        self.assertLess(abs(values.mean()), 0.05)
        self.assertLess(abs((values ** 2).mean() - 1), 0.05)

    def test_javascript(self):
        # lib/rng.js computes the same draws, exactly (the parity check - tools/parity.js)
        uniform = [0, 1, 17, 99999]
        gauss = [0, 5, 1234]
        parity.inputs(self, 'rng', {'seed': 31337, 'stream': 2, 'uniform': uniform, 'gauss': gauss})
        expected = parity.outputs('rng')
        seed = rng.stream_seed(31337, 2)
        self.assertEqual(seed, expected['streamSeed'])
        self.assertEqual(rng.uniform(seed, numpy.array(uniform)).tolist(), expected['uniform'])
        self.assertEqual([float(rng.gauss(seed, index, 1)[0]) for index in gauss], expected['gauss'])
