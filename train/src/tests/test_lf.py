# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest
import unittest.mock

import numpy

from picovox_train import lf, params

from . import parity


class TestLf(unittest.TestCase):

    def test_lf_timing(self):
        # lib/lf.js's golden values (test/testLf.js)
        self.assertEqual(lf.lf_timing(1), (0.4843626347650026, 0.6500146558546335, 0.038))
        self.assertEqual(lf.lf_timing(0), lf.lf_timing(0.3))
        self.assertEqual(lf.lf_timing(5), lf.lf_timing(2.7))
        self.assertEqual(lf.lf_timing(0.3), (0.27969515718880367, 0.3522480809635793, 0.004399999999999999))

    def test_lf_pulse(self):
        points = 20000
        u = numpy.arange(points) / points
        for rd in (0.3, 0.6, 1.0, 1.7, 2.7):
            pulse = lf.lf_pulse(u, rd)
            _, te, _ = lf.lf_timing(rd)
            # The net flow is zero, the negative peak -1 (at te)
            self.assertLess(abs(pulse.mean()), 1e-3)
            self.assertLess(abs(pulse.min() + 1), 1e-2)
            self.assertLess(abs(float(lf.lf_pulse(numpy.array([te]), rd)[0]) + 1), 1e-9)
            self.assertEqual(float(lf.lf_pulse(numpy.array([0.0]), rd)[0]), 0.0)

    def test_lf_pulse_negative_net_flow(self):
        # A pulse whose zero-growth net flow is negative takes the other scan direction
        with unittest.mock.patch.object(lf, 'lf_timing', return_value=(0.2, 0.5, 0.05)):
            pulse = lf.lf_pulse(numpy.arange(1000) / 1000, 1.0)
        self.assertTrue(numpy.isfinite(pulse).all())

    def test_lf_pulse_slow_return(self):
        # A return phase as long as its time constant - the decay constant's iteration stops at its limit
        with unittest.mock.patch.object(lf, 'lf_timing', return_value=(0.4, 0.9, 0.1)):
            pulse = lf.lf_pulse(numpy.arange(1000) / 1000, 1.0)
        self.assertTrue(numpy.isfinite(pulse).all())

    def test_harmonic_table(self):
        table = lf.harmonic_table()
        self.assertEqual(table.shape, (len(lf.rd_grid()), params.LF_HARMONICS))
        self.assertEqual(len(lf.rd_grid()), 241)
        self.assertIs(lf.harmonic_table(), table)
        # lib/lf.js's golden values (Rd 1.0 - grid index 70), and lib/lf.js's own table (the parity check - tools/parity.js)
        row = table[70]
        self.assertLess(abs(row[1].real - 0.062216950987419264), 1e-15)
        self.assertLess(abs(row[1].imag + 0.1705324149150758), 1e-15)
        self.assertLess(abs(row[5].real + 0.014752542877681525), 1e-15)
        parity.inputs(self, 'lfHarmonics', {'rows': [0, 70, 240], 'count': 64})
        expected = parity.outputs('lfHarmonics')
        for index, real, imag in zip((0, 70, 240), expected['real'], expected['imag']):
            self.assertLess(numpy.abs(table[index, :64].real - real).max(), 1e-12)
            self.assertLess(numpy.abs(table[index, :64].imag - imag).max(), 1e-12)

    def test_open_fraction(self):
        self.assertAlmostEqual(float(lf.open_fraction(1.0)), 0.62)
        self.assertEqual(float(lf.open_fraction(5.0)), 0.95)
