# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring, protected-access

import math
import types
import unittest
import unittest.mock

import numpy
import torch

from picovox_train import dsynth, params

from . import parity


# A frame of parameter values (the defaults, with overrides) as a list in parameter order
def frame(**values):
    return [float(values.get(name, params.DEFAULTS[name])) for name in params.NAMES]


# Frames [T, COUNT] of a voiced vowel with a breathy onset and a fricative tail, varying F0
def speech_frames(count=40, f0=120.0):
    frames = []
    for index in range(count):
        frames.append(frame(f0=f0 + 20 * math.sin(index / 5), av=-20 if index < count * 0.6 else -90, rd=1.2,
                            ah=-50, af=-40 if index > count / 2 else -90, ap1=-6, jitter=0.01, shimmer=0.05, shelf=3))
    return frames


#
# Fake Metal libraries - each kernel computed in PyTorch, as its Metal source documents it (the backward kernels by
# autograd of the forward), so the autograd Functions' plumbing is checked against the CPU path
#

def _sum_reference(phase, amp_re, amp_im, weight, harmonics):
    k = torch.arange(1, harmonics + 1, dtype=phase.dtype)
    cycles = phase[..., None] * k
    angle = 2 * math.pi * (cycles - torch.floor(cycles))
    cos, sin = torch.cos(angle), torch.sin(angle)
    re0, im0 = amp_re[:, :-1, None, :harmonics], amp_im[:, :-1, None, :harmonics]
    re1, im1 = amp_re[:, 1:, None, :harmonics], amp_im[:, 1:, None, :harmonics]
    sums0 = (cos * re0).sum(-1) - (sin * im0).sum(-1)
    sums1 = (cos * re1).sum(-1) - (sin * im1).sum(-1)
    return sums0 * (1 - weight) + sums1 * weight


def _tract_reference(cos1, sin1, cos2, sin2, coef_b, coef_c, log_a, per_frame):
    trig = [value[..., None] for value in (cos1, sin1, cos2, sin2)]
    if not per_frame:
        trig = [value[None, None] for value in trig]
    b, c, a = coef_b[:, :, None, :], coef_c[:, :, None, :], log_a[:, :, None, :]
    real = 1 - b * trig[0] - c * trig[2]
    imag = b * trig[1] + c * trig[3]
    terms = a - 0.5 * torch.log(real * real + imag * imag)
    turns = -torch.atan2(imag, real)
    return terms[..., :-1].sum(-1) - terms[..., -1], turns[..., :-1].sum(-1) - turns[..., -1]


def _frication_reference(cos1, sin1, cos2, sin2, coef):
    trig = (cos1, sin1, cos2, sin2)
    def biquad(offset):
        return dsynth._biquad(None, *(coef[..., offset + ix:offset + ix + 1] for ix in range(5)), trig)
    bp1_re, bp1_im = biquad(3)
    bp2_re, bp2_im = biquad(8)
    hp_re, hp_im = biquad(13)
    ab, ap1, ap2 = coef[..., 0:1], coef[..., 1:2], coef[..., 2:3]
    real = ab + ap1 * (bp1_re * bp1_re - bp1_im * bp1_im) + ap2 * (bp2_re * bp2_re - bp2_im * bp2_im)
    imag = ap1 * (2 * bp1_re * bp1_im) + ap2 * (2 * bp2_re * bp2_im)
    power = hp_re * hp_re + hp_im * hp_im
    return torch.sqrt(real * real + imag * imag + 1e-30) * torch.sqrt(power * power + 1e-30)


def _amplitudes_reference(f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq, lf_re, lf_im, consts, has_eq):
    harmonics = tract_magnitude.shape[-1]
    k = torch.arange(1, harmonics + 1, dtype=f0.dtype)
    frequency = f0[..., None] * k
    omega = 2 * math.pi * frequency / 44100
    trig = (torch.cos(omega), torch.sin(omega), torch.cos(2 * omega), torch.sin(2 * omega))
    shelf_re, shelf_im = dsynth._biquad(None, *(shelf[..., ix:ix + 1] for ix in range(5)), trig)
    log_magnitude = tract_magnitude
    if has_eq:
        position = torch.clamp(torch.log(torch.clamp(frequency, min=1.0) / 60.0) / consts[2], 0, 23)
        eq_low = torch.clamp(torch.floor(position).long(), max=22)
        eq_weight = position - eq_low
        gains = eq.expand(-1, f0.shape[1], -1) if eq.dim() == 3 else eq[:, None].expand(-1, f0.shape[1], -1)
        db = torch.gather(gains, 2, eq_low) * (1 - eq_weight) + torch.gather(gains, 2, eq_low + 1) * eq_weight
        log_magnitude = log_magnitude + db * 0.11512925
    taper = 0.5 + 0.5 * torch.cos(math.pi * torch.clamp((frequency - consts[0]) / (consts[1] - consts[0]), 0, 1))
    magnitude = 2 * av[..., None] * torch.exp(log_magnitude + 0.5 * torch.log(shelf_re * shelf_re + shelf_im * shelf_im + 1e-30)) * taper
    phase = tract_phase + torch.atan2(shelf_im, shelf_re)
    filter_re, filter_im = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
    rows = low.long()
    w = weight[..., None]
    source_re = lf_re[rows, 1:harmonics + 1] * (1 - w) + lf_re[rows + 1, 1:harmonics + 1] * w
    source_im = lf_im[rows, 1:harmonics + 1] * (1 - w) + lf_im[rows + 1, 1:harmonics + 1] * w
    return source_re * filter_re - source_im * filter_im, source_re * filter_im + source_im * filter_re


class FakeLibrary:
    """The Metal kernels of dsynth.py's libraries, in PyTorch (the backward kernels run in a backward pass - autograd
    enabled for theirs)"""

    def __init__(self):
        self.calls = []

    def harmonic_sum(self, out, phase, amp_re, amp_im, weight, dims, active, threads):
        self.calls.append(('harmonic_sum', dims, threads))
        out.copy_(_sum_reference(phase, amp_re, amp_im, weight, min(dims[3], int(active[0]))))

    @torch.enable_grad()
    def harmonic_sum_backward(self, grad_re, grad_im, grad, phase, weight, dims, active, threads):
        self.calls.append(('harmonic_sum_backward', dims, threads))
        amp_re = torch.zeros(grad_re.shape, dtype=grad.dtype, requires_grad=True)
        amp_im = torch.zeros(grad_re.shape, dtype=grad.dtype, requires_grad=True)
        out = _sum_reference(phase, amp_re, amp_im, weight, min(dims[3], int(active[0])))
        values = torch.autograd.grad(out, (amp_re, amp_im), grad)
        grad_re.copy_(values[0])
        grad_im.copy_(values[1])

    def tract_response(self, magnitude, phase, cos1, sin1, cos2, sin2, coef_b, coef_c, log_a, dims, flags, threads):
        self.calls.append(('tract_response', dims, list(flags), threads))
        total, angle = _tract_reference(cos1, sin1, cos2, sin2, coef_b, coef_c, log_a, flags[0])
        magnitude.copy_(total)
        if flags[1]:
            phase.copy_(angle)

    def _tract_grads(self, grad_magnitude, grad_phase, trig, coef_b, coef_c, flags, inputs):
        total, angle = _tract_reference(*trig, coef_b, coef_c, inputs[-1], flags[0])
        outputs, grads = [total], [grad_magnitude]
        if flags[1]:
            outputs.append(angle)
            grads.append(grad_phase)
        return torch.autograd.grad(outputs, inputs, grads, allow_unused=True)

    @torch.enable_grad()
    def tract_response_backward(self, grad_b, grad_c, grad_log_a, grad_magnitude, grad_phase, cos1, sin1, cos2, sin2, coef_b,
                                coef_c, dims, flags, threads):
        self.calls.append(('tract_response_backward', dims, list(flags), threads))
        b, c = coef_b.clone().requires_grad_(), coef_c.clone().requires_grad_()
        log_a = torch.zeros_like(coef_b, requires_grad=True)
        values = self._tract_grads(grad_magnitude, grad_phase, (cos1, sin1, cos2, sin2), b, c, flags, (b, c, log_a))
        for buffer, value in zip((grad_b, grad_c, grad_log_a), values):
            buffer.copy_(value)

    @torch.enable_grad()
    def tract_response_trig_backward(self, grad_c1, grad_s1, grad_c2, grad_s2, grad_magnitude, grad_phase, cos1, sin1, cos2,
                                     sin2, coef_b, coef_c, dims, flags, threads):
        self.calls.append(('tract_response_trig_backward', dims, list(flags), threads))
        trig = [value.clone().requires_grad_() for value in (cos1, sin1, cos2, sin2)]
        log_a = torch.zeros_like(coef_b, requires_grad=True)
        values = self._tract_grads(grad_magnitude, grad_phase, trig, coef_b, coef_c, flags, (*trig, log_a))
        for buffer, value in zip((grad_c1, grad_s1, grad_c2, grad_s2), values):
            buffer.copy_(value)

    def frication(self, out, cos1, sin1, cos2, sin2, coef, dims, threads):
        self.calls.append(('frication', dims, threads))
        out.copy_(_frication_reference(cos1, sin1, cos2, sin2, coef))

    @torch.enable_grad()
    def frication_backward(self, grad_coef, grad, cos1, sin1, cos2, sin2, coef, dims, threads):
        self.calls.append(('frication_backward', dims, threads))
        leaf = coef.clone().requires_grad_()
        grad_coef.copy_(torch.autograd.grad(_frication_reference(cos1, sin1, cos2, sin2, leaf), leaf, grad)[0])

    def amplitudes(self, amp_re, amp_im, f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq, lf_re, lf_im, consts, dims,
                   has_eq, threads):
        self.calls.append(('amplitudes', dims, has_eq, threads))
        real, imag = _amplitudes_reference(f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq, lf_re, lf_im, consts, has_eq)
        amp_re.copy_(real)
        amp_im.copy_(imag)

    @torch.enable_grad()
    def amplitudes_backward(self, grad_av, grad_weight, grad_shelf, grad_eq, grad_tract_magnitude, grad_tract_phase, grad_re,
                            grad_im, f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq, lf_re, lf_im, consts, dims,
                            has_eq, threads):
        self.calls.append(('amplitudes_backward', dims, has_eq, threads))
        leaves = [value.clone().requires_grad_() for value in (av, weight, shelf, tract_magnitude, tract_phase)]
        per_frame_eq = eq[:, None].repeat(1, f0.shape[1], 1).requires_grad_() if has_eq else eq
        real, imag = _amplitudes_reference(f0, leaves[0], low, leaves[1], leaves[2], leaves[3], leaves[4], per_frame_eq, lf_re, lf_im,
                                           consts, has_eq)
        inputs = leaves + ([per_frame_eq] if has_eq else [])
        values = torch.autograd.grad((real, imag), inputs, (grad_re, grad_im))
        for buffer, value in zip((grad_av, grad_weight, grad_shelf, grad_tract_magnitude, grad_tract_phase), values):
            buffer.copy_(value)
        if has_eq:
            grad_eq.copy_(values[-1])


class TestFilters(unittest.TestCase):

    def test_sizes(self):
        self.assertEqual(dsynth.noise_fft_size(24000), 512)
        self.assertEqual(dsynth.noise_fft_size(44100), 1024)
        self.assertEqual(dsynth.sample_count(3, 24000), 240)
        self.assertEqual(dsynth.sample_count(0, 24000), 0)

    def test_resonator_log(self):
        frequency = torch.tensor([0.0, 300.0, 1000.0, 5000.0], dtype=torch.float64)
        center, bandwidth = torch.tensor(1000.0, dtype=torch.float64), torch.tensor(100.0, dtype=torch.float64)
        magnitude, phase = dsynth.resonator_log(frequency, center, bandwidth)
        # Unity gain at DC; the response is the Klatt resonator's A / (1 - B z^-1 - C z^-2)
        self.assertAlmostEqual(float(magnitude[0]), 0.0, places=12)
        radius = math.exp(-math.pi * 100 / params.REF_RATE)
        coef_c, coef_b = -radius * radius, 2 * radius * math.cos(2 * math.pi * 1000 / params.REF_RATE)
        z = numpy.exp(-1j * 2 * numpy.pi * frequency.numpy() / params.REF_RATE)
        response = (1 - coef_b - coef_c) / (1 - coef_b * z - coef_c * z * z)
        self.assertTrue(numpy.allclose(magnitude.numpy(), numpy.log(numpy.abs(response))))
        self.assertTrue(numpy.allclose(phase.numpy(), numpy.angle(response)))
        trig = dsynth.frequency_trig(frequency)
        magnitude_only, no_phase = dsynth.resonator_log(frequency, center, bandwidth, trig, phase=False)
        self.assertIsNone(no_phase)
        self.assertTrue(torch.equal(magnitude_only, magnitude))

    def test_tract_log(self):
        values = dsynth.split_values(torch.tensor([frame()], dtype=torch.float64))
        frequency = torch.linspace(0, 8000, 50, dtype=torch.float64)
        magnitude, phase = dsynth.tract_log(frequency, values)
        self.assertEqual(magnitude.shape, (1, 50))
        # The pole sum less the zero, as the resonators give them
        expected = 0
        for number in range(1, 7):
            expected = expected + dsynth.resonator_log(frequency, values[f'f{number}'], values[f'b{number}'])[0]
        expected = expected + dsynth.resonator_log(frequency, values['fnp'], values['bnp'])[0]
        for kx in range(1, params.HIGHER_POLES + 1):
            center = values['f6'] * (1 + kx * params.HIGHER_POLE_SPACING)
            expected = expected + dsynth.resonator_log(frequency, center, torch.full_like(center, 500.0 + 100 * (kx - 1)))[0]
        expected = expected - dsynth.resonator_log(frequency, values['fnz'], values['bnz'])[0]
        self.assertTrue(torch.allclose(magnitude, expected))
        magnitude_only, no_phase = dsynth.tract_log(frequency, values, phase=False)
        self.assertIsNone(no_phase)
        self.assertTrue(torch.allclose(magnitude_only, magnitude))
        self.assertEqual(phase.shape, (1, 50))

    def test_shelf_log(self):
        frequency = torch.tensor([0.0, 100.0, 2000.0, 15000.0], dtype=torch.float64)
        flat_magnitude, flat_phase = dsynth.shelf_log(frequency, torch.tensor(0.0, dtype=torch.float64))
        self.assertTrue(torch.allclose(flat_magnitude, torch.zeros(4, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(flat_phase, torch.zeros(4, dtype=torch.float64), atol=1e-12))
        magnitude, _ = dsynth.shelf_log(frequency, torch.tensor(12.0, dtype=torch.float64))
        # The shelf - unity at DC, the gain well above the corner, half the gain (dB) at the corner
        self.assertAlmostEqual(float(magnitude[0]), 0.0, places=9)
        self.assertAlmostEqual(float(magnitude[3]) * 20 / math.log(10), 12.0, delta=0.3)
        self.assertAlmostEqual(float(magnitude[2]) * 20 / math.log(10), 6.0, delta=0.5)

    def test_frication_filters(self):
        frequency = torch.tensor([100.0, 3000.0, 10000.0], dtype=torch.float64)
        real, imag = dsynth._bandpass(frequency, torch.tensor(3000.0, dtype=torch.float64), torch.tensor(500.0, dtype=torch.float64))
        self.assertAlmostEqual(math.hypot(float(real[1]), float(imag[1])), 1.0, places=6)
        self.assertLess(math.hypot(float(real[0]), float(imag[0])), 0.1)
        power = dsynth._highpass_power(frequency, torch.tensor(1000.0, dtype=torch.float64))
        self.assertLess(float(power[0]), 1e-3)
        self.assertAlmostEqual(float(power[2]), 1.0, delta=0.01)
        values = dsynth.split_values(torch.tensor([frame(ab=0.0, ap1=-90.0, ap2=-90.0, fhp=50.0)], dtype=torch.float64))
        flat = dsynth.frication_magnitude(frequency, values)
        self.assertTrue(torch.allclose(flat[0, 1:], torch.ones(2, dtype=torch.float64), atol=0.01))
        self.assertEqual(flat.shape, (1, 3))
        aspiration = dsynth.aspiration_magnitude(torch.tensor([0.0, 22050.0], dtype=torch.float64))
        self.assertAlmostEqual(float(aspiration[0]), 1.0)
        self.assertAlmostEqual(float(aspiration[1]), 1 / 3)

    def test_harmonic_taper(self):
        taper = dsynth.harmonic_taper(torch.tensor([0.0, 10560.0, 11000.0, 11400.0, 12000.0]), 24000)
        self.assertEqual(float(taper[0]), 1.0)
        self.assertEqual(float(taper[1]), 1.0)
        self.assertTrue(0 < float(taper[2]) < 1)
        self.assertAlmostEqual(float(taper[3]), 0.0, places=6)
        self.assertAlmostEqual(float(taper[4]), 0.0, places=6)

    def test_eq_log(self):
        gains = torch.arange(params.EQ_BAND_COUNT, dtype=torch.float64)[None]
        step = math.log(params.EQ_HIGH_HZ / params.EQ_LOW_HZ) / (params.EQ_BAND_COUNT - 1)
        frequency = torch.tensor([[0.0, 60.0, 60.0 * math.exp(2.5 * step), 16000.0, 20000.0]], dtype=torch.float64)
        db = dsynth.eq_log(frequency, gains) * 20 / math.log(10)
        self.assertTrue(torch.allclose(db, torch.tensor([[0.0, 0.0, 2.5, 23.0, 23.0]], dtype=torch.float64)))

    def test_cached(self):
        builds = []
        with unittest.mock.patch.dict(dsynth._CACHE, clear=True):
            self.assertEqual(dsynth._cached('a', lambda: builds.append('a') or 1), 1)
            self.assertEqual(dsynth._cached('a', lambda: builds.append('again') or 2), 1)
            with unittest.mock.patch.object(dsynth, '_CACHE_LIMIT', 1):
                self.assertEqual(dsynth._cached('b', lambda: 3), 3)
            self.assertEqual(list(dsynth._CACHE), ['b'])
        self.assertEqual(builds, ['a'])

    def test_interpolate_frames(self):
        frames = torch.tensor([[[0.0], [10.0], [20.0]]])
        positions = numpy.array([0.0, 0.5, 1.75, 2.0, 5.0])
        result = dsynth.interpolate_frames(frames, positions)
        self.assertEqual(result[0, :, 0].tolist(), [0.0, 5.0, 17.5, 20.0, 20.0])
        with unittest.mock.patch.dict(dsynth._CACHE, clear=True):
            self.assertTrue(torch.equal(dsynth.interpolate_frames(frames, positions, key='test'), result))
            self.assertTrue(torch.equal(dsynth.interpolate_frames(frames, positions, key='test'), result))

    def test_linear_values(self):
        values = torch.tensor([frame(av=0.0, ah=-20.0, shelf=6.0)])
        linear = dsynth.linear_values(values)
        self.assertAlmostEqual(float(linear[0, params.INDEX['av']]), 1.0)
        self.assertAlmostEqual(float(linear[0, params.INDEX['ah']]), 0.1, places=6)
        self.assertEqual(float(linear[0, params.INDEX['shelf']]), 6.0)
        split = dsynth.split_values(values, linear)
        self.assertEqual(split['f1'].shape, (1, 1))
        self.assertTrue(torch.equal(split['av'], linear[..., params.INDEX['av']:params.INDEX['av'] + 1]))

    def test_row_draws(self):
        draws = dsynth._row_draws([5, 6], 3, [2, 4])
        self.assertEqual(draws.shape, (2, 4))
        self.assertEqual(draws[0, 2:].tolist(), [0.0, 0.0])
        self.assertTrue(numpy.array_equal(draws[1], dsynth.rng.gauss(dsynth.rng.stream_seed(6, 3), 0, 4)))


class TestLibraries(unittest.TestCase):

    def test_loaders(self):
        for loader, cache in ((dsynth._harmonic_library, dsynth._HARMONIC_LIBRARY), (dsynth._tract_library, dsynth._TRACT_LIBRARY),
                              (dsynth._frication_library, dsynth._FRICATION_LIBRARY)):
            with unittest.mock.patch.dict(cache, clear=True), \
                    unittest.mock.patch.object(torch.mps, 'compile_shader', return_value='library') as compile_shader:
                self.assertEqual(loader(), 'library')
                self.assertEqual(loader(), 'library')
                compile_shader.assert_called_once()
            # No Metal - no library
            with unittest.mock.patch.dict(cache, clear=True), \
                    unittest.mock.patch.object(torch.mps, 'compile_shader', side_effect=RuntimeError('no Metal')):
                self.assertIsNone(loader())

    def test_amplitudes_library_for(self):
        self.assertIsNone(dsynth._amplitudes_library_for(torch.zeros(1)))
        values = unittest.mock.Mock(dtype=torch.float64)
        values.device.type = 'mps'
        self.assertIsNone(dsynth._amplitudes_library_for(values))
        values.dtype = torch.float32
        with unittest.mock.patch.dict(dsynth._AMPLITUDES_LIBRARY, clear=True):
            with unittest.mock.patch.object(torch.mps, 'compile_shader', return_value='library'):
                self.assertEqual(dsynth._amplitudes_library_for(values), 'library')
            self.assertEqual(dsynth._amplitudes_library_for(values), 'library')
        with unittest.mock.patch.dict(dsynth._AMPLITUDES_LIBRARY, clear=True), \
                unittest.mock.patch.object(torch.mps, 'compile_shader', side_effect=RuntimeError('no Metal')):
            self.assertIsNone(dsynth._amplitudes_library_for(values))

    def test_amplitudes_source_constants(self):
        # The amplitudes kernel's literals are params's values - the reference rate, the equalizer's lowest band and band
        # count, and ln(10) / 20 (the tests don't run the Metal path, which would otherwise part from the CPU path and
        # lib/render.js unnoticed)
        self.assertEqual((params.REF_RATE, params.EQ_LOW_HZ, params.EQ_BAND_COUNT, round(math.log(10) / 20, 8)),
                         (44100, 60, 24, 0.11512925))
        for literal in ('/ 44100.0f;', '/ 60.0f)', '0.0f, 23.0f)', '22L)', '* 24;', 'g_eq[24];', 'i < 24;', 'frame * 24 +',
                        '* 0.11512925f;'):
            self.assertIn(literal, dsynth._AMPLITUDES_SOURCE)


class TestKernels(unittest.TestCase):
    """The kernels' autograd Functions with the fake libraries (the loaders patched for each test - the backward passes
    load them too)"""

    def setUp(self):
        self.library = FakeLibrary()
        for loader in ('_harmonic_library', '_tract_library', '_frication_library'):
            self.enterContext(unittest.mock.patch.object(dsynth, loader, return_value=self.library))

    def test_kernel_gates(self):
        # The kernels apply only to MPS float32 [B, T, 1] parameters
        values = dsynth.split_values(torch.tensor([[frame()]], dtype=torch.float32))
        trig = dsynth.frequency_trig(torch.linspace(0, 8000, 10))
        self.assertIsNone(dsynth._tract_kernel(trig, [(values['f1'], values['b1'])], (values['fnz'], values['bnz']), True))
        self.assertIsNone(dsynth._frication_kernel(trig, values))

    def test_harmonic_sum(self):
        torch.manual_seed(0)
        library = self.library
        phase = torch.rand(2, 3, 5)
        amp_re = torch.randn(2, 4, 6, requires_grad=True)
        amp_im = torch.randn(2, 4, 6, requires_grad=True)
        weight = torch.rand(3, 5)
        with unittest.mock.patch.dict(dsynth._CACHE, clear=True):
            out = dsynth._HarmonicSum.apply(phase, amp_re, amp_im, weight)
            active = torch.tensor([4], dtype=torch.int32)
            limited = dsynth._HarmonicSum.apply(phase, amp_re, amp_im, weight, active)
        self.assertTrue(torch.allclose(out, _sum_reference(phase, amp_re, amp_im, weight, 6)))
        self.assertTrue(torch.allclose(limited, _sum_reference(phase, amp_re, amp_im, weight, 4)))
        self.assertEqual(library.calls[0], ('harmonic_sum', [2, 4, 5, 6], 30))
        grad = torch.randn_like(out)
        grads = torch.autograd.grad(limited, (amp_re, amp_im), grad)
        expected = torch.autograd.grad(_sum_reference(phase, amp_re, amp_im, weight, 4), (amp_re, amp_im), grad)
        for value, expected_value in zip(grads, expected):
            self.assertTrue(torch.allclose(value, expected_value, atol=1e-6))
        self.assertEqual(float(grads[0][..., 4:].abs().sum()), 0.0)

    def test_tract_metal(self):
        # The tract by the kernels equals tract_log's PyTorch sum - values and gradients, per-frame and shared frequencies
        torch.manual_seed(1)
        library = self.library
        frames = torch.tensor([speech_frames(4)], dtype=torch.float32, requires_grad=True)
        # The tract of the frames - by the kernels, or PyTorch
        def tract(per_frame, phase, kernel):
            values = dsynth.split_values(frames)
            frequency = values['f0'] * torch.arange(1, 7) if per_frame else torch.linspace(50, 8000, 12)
            if not kernel:
                return dsynth.tract_log(frequency, values, phase=phase)
            with unittest.mock.patch.object(dsynth, '_tract_kernel', dsynth._tract_metal):
                return dsynth.tract_log(frequency, values, phase=phase)
        for per_frame in (False, True):
            for phase in (True, False):
                expected = tract(per_frame, phase, False)
                actual = tract(per_frame, phase, True)
                self.assertTrue(torch.allclose(actual[0], expected[0], atol=1e-4))
                if phase:
                    self.assertTrue(torch.allclose(actual[1], expected[1], atol=1e-4))
                else:
                    self.assertIsNone(actual[1])
                loss = actual[0].sum() + (actual[1].sum() if phase else 0)
                expected_loss = expected[0].sum() + (expected[1].sum() if phase else 0)
                self.assertTrue(torch.allclose(torch.autograd.grad(loss, frames)[0], torch.autograd.grad(expected_loss, frames)[0],
                                               rtol=1e-3, atol=1e-3))
        # Only the phase's gradient - the magnitude's is zeros
        values = dsynth.split_values(frames)
        with unittest.mock.patch.object(dsynth, '_tract_kernel', dsynth._tract_metal):
            _, phase = dsynth.tract_log(torch.linspace(50, 8000, 12), values)
        expected = torch.autograd.grad(dsynth.tract_log(torch.linspace(50, 8000, 12), dsynth.split_values(frames))[1].sum(), frames)[0]
        self.assertTrue(torch.allclose(torch.autograd.grad(phase.sum(), frames)[0], expected, rtol=1e-3, atol=1e-3))
        self.assertIn('tract_response_trig_backward', [call[0] for call in library.calls])

    def test_tract_response_backward(self):
        # Without a magnitude gradient (autograd's unmaterialized gradient) - as a zero gradient
        trig = [value.contiguous() for value in dsynth.frequency_trig(torch.linspace(50, 8000, 6))]
        coef_b, coef_c = torch.rand(1, 2, 3), -torch.rand(1, 2, 3)
        ctx = types.SimpleNamespace(saved_tensors=(*trig, coef_b, coef_c), dims=[1, 2, 6, 3], flags=[0, 1],
                                    needs_input_grad=(False,) * 8)
        grad_phase = torch.randn(1, 2, 6)
        grads = dsynth._TractResponse.backward(ctx, None, grad_phase)
        expected = dsynth._TractResponse.backward(ctx, torch.zeros(1, 2, 6), grad_phase)
        self.assertEqual(grads[:4], (None, None, None, None))
        for value, expected_value in zip(grads[4:7], expected[4:7]):
            self.assertTrue(torch.equal(value, expected_value))

    def test_tract_metal_shapes(self):
        # Frequencies of another shape, or not float32 - no kernel
        values = dsynth.split_values(torch.tensor([speech_frames(4)], dtype=torch.float32))
        poles = [(values['f1'], values['b1'])]
        zero = (values['fnz'], values['bnz'])
        self.assertIsNone(dsynth._tract_metal(dsynth.frequency_trig(torch.ones(2, 3, 5)), poles, zero, True))
        self.assertIsNone(dsynth._tract_metal(dsynth.frequency_trig(torch.ones(5, dtype=torch.float64)), poles, zero, True))

    def test_frication_metal(self):
        library = self.library
        frames = torch.tensor([speech_frames(4)], dtype=torch.float32, requires_grad=True)
        frequency = torch.linspace(50, 11000, 16)
        trig = dsynth.frequency_trig(frequency)
        expected = dsynth.frication_magnitude(frequency, dsynth.split_values(frames), trig)
        with unittest.mock.patch.object(dsynth, '_frication_kernel', dsynth._frication_metal):
            actual = dsynth.frication_magnitude(frequency, dsynth.split_values(frames), trig)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-4))
        grad = torch.autograd.grad(actual.sum(), frames)[0]
        self.assertTrue(torch.allclose(grad, torch.autograd.grad(expected.sum(), frames)[0], rtol=1e-3, atol=1e-6))
        self.assertEqual(library.calls[0], ('frication', [4, 16], 64))

    def test_harmonic_amplitudes_metal(self):
        # The amplitudes by the kernel (a constant F0) equal the PyTorch path's - values and gradients, with and without
        # the channel equalizer, and with only one part's gradient
        synth = dsynth.DSynth(24000)
        frames = torch.tensor([speech_frames(5), speech_frames(5, f0=200.0)], dtype=torch.float32, requires_grad=True)
        eq = torch.randn(2, params.EQ_BAND_COUNT, generator=torch.Generator().manual_seed(2), requires_grad=True)
        cases = [(None, (1.0, 2.0)), (eq, (1.0, 2.0)), (None, (1.0, 0.0)), (None, (0.0, 1.0))]

        # The PyTorch path's amplitudes and gradients
        def amplitudes_and_grads(case_eq, weights):
            real, imag = synth.harmonic_amplitudes(frames, 20, case_eq, constant_f0=True)
            inputs = (frames,) if case_eq is None else (frames, case_eq)
            loss = (weights[0] * real.sum() if weights[0] else 0) + (weights[1] * imag.sum() if weights[1] else 0)
            return real.detach(), imag.detach(), torch.autograd.grad(loss, inputs)
        expected = [amplitudes_and_grads(*case) for case in cases]
        with unittest.mock.patch.object(dsynth, '_amplitudes_library_for', return_value=self.library):
            actual = [amplitudes_and_grads(*case) for case in cases]
        for (real, imag, grads), (expected_real, expected_imag, expected_grads) in zip(actual, expected):
            self.assertTrue(torch.allclose(real, expected_real, rtol=1e-4, atol=1e-7))
            self.assertTrue(torch.allclose(imag, expected_imag, rtol=1e-4, atol=1e-7))
            for value, expected_value in zip(grads, expected_grads):
                self.assertTrue(torch.allclose(value, expected_value, rtol=1e-3, atol=1e-6))
        self.assertIn(('amplitudes', [10, 20, 5, params.LF_HARMONICS], 1, 200), self.library.calls)


class TestDSynth(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_forward(self):
        synth = dsynth.DSynth(24000, noise='counter')
        values = torch.tensor(speech_frames(), dtype=torch.float32)
        audio = synth(values, seed=3)
        self.assertEqual(audio.shape, (dsynth.sample_count(40, 24000),))
        self.assertTrue(torch.isfinite(audio).all())
        self.assertGreater(float(audio.abs().max()), 0.01)
        # A batch, seeds by item, the parts
        batch = synth(values[None].repeat(2, 1, 1), seed=[3, 4], return_parts=True)
        self.assertTrue(torch.allclose(batch['audio'][0], audio))
        self.assertFalse(torch.allclose(batch['audio'][1], audio))
        self.assertTrue(torch.allclose(batch['voicing'] + batch['noise'], batch['audio']))
        parts = synth(values, seed=3, return_parts=True)
        self.assertEqual(parts['voicing'].shape, audio.shape)
        # An int seed counts up by item
        self.assertTrue(torch.allclose(synth(values[None].repeat(2, 1, 1), seed=3)[1], batch['audio'][1]))
        # A single frame renders nothing
        self.assertEqual(synth(values[:1]).shape, (0,))
        self.assertEqual(synth(values[None, :1]).shape, (1, 0))

    def test_noise(self):
        values = torch.tensor(speech_frames(), dtype=torch.float32)
        torch_noise = dsynth.DSynth(24000)(values, seed=5)
        self.assertTrue(torch.allclose(dsynth.DSynth(24000)(values, seed=5), torch_noise))
        self.assertFalse(torch.allclose(dsynth.DSynth(24000, noise='counter')(values, seed=5), torch_noise))

    def test_eq(self):
        synth = dsynth.DSynth(16000, noise='counter')
        values = torch.tensor(speech_frames(), dtype=torch.float32)
        flat = synth(values, eq=torch.zeros(params.EQ_BAND_COUNT))
        self.assertTrue(torch.allclose(flat, synth(values), atol=1e-6))
        loud = synth(values[None], eq=torch.full((1, params.EQ_BAND_COUNT), 6.0))
        self.assertTrue(torch.allclose(loud[0], 10 ** (6 / 20) * flat, rtol=1e-3, atol=1e-5))

    def test_gradients(self):
        synth = dsynth.DSynth(16000)
        values = torch.tensor(speech_frames(20), dtype=torch.float64, requires_grad=True)
        synth.double()
        audio = synth(values[None], seed=1)
        grad = torch.autograd.grad((audio ** 2).sum(), values)[0]
        self.assertGreater(float(grad[:, params.INDEX['av']].abs().sum()), 0)
        self.assertGreater(float(grad[:, params.INDEX['f1']].abs().sum()), 0)
        self.assertGreater(float(grad[:, params.INDEX['f0']].abs().sum()), 0)
        # A constant F0 isn't differentiated
        constant = synth(values[None], seed=1, constant_f0=True)
        self.assertTrue(torch.allclose(constant, audio))
        grad = torch.autograd.grad((constant ** 2).sum(), values)[0]
        self.assertEqual(float(grad[:, params.INDEX['f0']].abs().sum()), 0.0)

    def test_plans(self):
        synth = dsynth.DSynth(24000, noise='counter')
        values = torch.tensor([speech_frames(), speech_frames(f0=180.0)], dtype=torch.float32)
        expected = synth(values, seed=[1, 2])
        frames = values.double().numpy()
        f0, jitter = frames[..., params.INDEX['f0']], frames[..., params.INDEX['jitter']]
        plan = synth.plan_cycles(f0, jitter, [1, 2])
        self.assertEqual(plan['count'], dsynth.sample_count(40, 24000))
        self.assertEqual(plan['fraction'].shape, (2, plan['count']))
        # A plan made ahead renders the same
        self.assertTrue(torch.equal(synth(values, seed=[1, 2], plan=dict(plan)), expected))
        self.assertEqual(synth.plan_mismatches, 0)
        # With its host tensors - in the values' dtype, or not
        host_plan = {**plan, 'host': synth.plan_host_tensors(plan)}
        self.assertTrue(torch.equal(synth(values, seed=[1, 2], plan=host_plan), expected))
        double_plan = {**plan, 'host': synth.plan_host_tensors(plan, torch.float64)}
        self.assertTrue(torch.equal(synth(values, seed=[1, 2], plan=double_plan), expected))
        # A plan of other frames, or other seeds, is made again
        other = synth.plan_cycles(f0 + 1, jitter, [1, 2])
        self.assertTrue(torch.equal(synth(values, seed=[1, 2], plan=other), expected))
        self.assertTrue(torch.equal(synth(values, seed=[1, 2], plan=synth.plan_cycles(f0, jitter + 0.001, [1, 2])), expected))
        self.assertTrue(torch.equal(synth(values, seed=[1, 2], plan=synth.plan_cycles(f0, jitter, [1, 3])), expected))
        self.assertEqual(synth.plan_mismatches, 3)
        self.assertEqual(synth.deferred_plan_mismatches(), 0)

    def test_deferred_plan(self):
        # A plan finished on the device (as on MPS - _can_defer) renders as the host's plan
        synth = dsynth.DSynth(24000, noise='counter')
        values = torch.tensor([speech_frames(), speech_frames(f0=180.0)], dtype=torch.float32)
        expected = synth(values, seed=[1, 2])
        frames = values.double().numpy()
        plan = synth.plan_cycles(frames[..., params.INDEX['f0']], frames[..., params.INDEX['jitter']], [1, 2])
        plan['host'] = synth.plan_host_tensors(plan)
        with unittest.mock.patch.object(synth, '_can_defer', return_value=True):
            audio = synth(values, seed=[1, 2], plan=plan)
            self.assertEqual(synth.deferred_plan_mismatches(), 0)
            # A plan of other frames is counted on the device
            moved = values.clone()
            moved[..., params.INDEX['f0']] += 1
            synth(moved, seed=[1, 2], plan=plan)
            self.assertEqual(synth.deferred_plan_mismatches(), 1)
        self.assertTrue(torch.allclose(audio, expected, atol=1e-6))
        # The device's open phase and harmonic count are the host's - and unvoiced frames' fallback count
        grid = synth._grid_tensors(plan['count'], 40, values.device, values.dtype)
        for av in (None, params.FLOOR_DB):
            frames_av = values.clone()
            if av is not None:
                frames_av[..., params.INDEX['av']] = av
            tensors, harmonics = synth._deferred_plan(frames_av, plan, grid)
            host = synth.finish_plan(dict(plan), frames_av[..., params.INDEX['av']].double().numpy(),
                                     frames_av[..., params.INDEX['rd']].double().numpy())
            self.assertEqual(int(tensors['harmonics'][0]), host['harmonics'])
            self.assertGreaterEqual(harmonics, host['harmonics'])
            self.assertEqual(harmonics % 16, 0)
            # (the device's Rd is float32 - an open-phase edge may round the other way)
            self.assertLess((tensors['open'].numpy() != host['open']).mean(), 1e-3)

    def test_can_defer(self):
        synth = dsynth.DSynth(24000)
        values = unittest.mock.Mock(dtype=torch.float32)
        values.device.type = 'mps'
        host = {'fraction': torch.zeros(1), 'f0': torch.zeros(1)}
        with unittest.mock.patch.object(dsynth, '_harmonic_library', return_value='library'):
            self.assertTrue(synth._can_defer(values, {'host': host}))
            self.assertFalse(synth._can_defer(values, {}))
            self.assertFalse(synth._can_defer(values, {'host': {'fraction': torch.zeros(1)}}))
            self.assertFalse(synth._can_defer(values, {'host': {**host, 'fraction': torch.zeros(1, dtype=torch.float64)}}))
            self.assertFalse(synth._can_defer(torch.zeros(1), {'host': host}))
        with unittest.mock.patch.object(dsynth, '_harmonic_library', return_value=None):
            self.assertFalse(synth._can_defer(values, {'host': host}))

    def test_harmonic_count(self):
        synth = dsynth.DSynth(24000)
        f0 = numpy.array([[100.0, 30.0, 200.0]])
        self.assertEqual(synth._harmonic_count(f0, numpy.array([[0.0, -90.0, 0.0]])), math.ceil(0.475 * 24000 / 100))
        self.assertEqual(synth._harmonic_count(f0, numpy.array([[0.0, 0.0, 0.0]])), math.ceil(0.475 * 24000 / 40))
        self.assertEqual(synth._harmonic_count(f0, numpy.full((1, 3), -90.0)), math.ceil(0.475 * 24000 / 1000))
        self.assertEqual(dsynth.DSynth(1000000)._harmonic_count(f0, numpy.zeros((1, 3))), params.LF_HARMONICS - 1)

    def test_javascript(self):
        # lib/render.js renders the same frames to the same samples (its counter noise; float32 output - the parity check,
        # tools/parity.js)
        values = speech_frames()
        eq = [float(value) for value in numpy.linspace(-6, 6, params.EQ_BAND_COUNT)]
        parity.inputs(self, 'render', {'sampleRate': 24000, 'seed': 7, 'frames': values, 'eq': eq})
        expected = parity.outputs('render')
        plain, equalized = numpy.array(expected['samples']), numpy.array(expected['samplesEq'])
        synth = dsynth.DSynth(24000, noise='counter').double()
        audio = synth(torch.tensor(values, dtype=torch.float64), seed=7).numpy()
        audio_eq = synth(torch.tensor(values, dtype=torch.float64), seed=7, eq=torch.tensor(eq, dtype=torch.float64)).numpy()
        self.assertEqual(len(plain), len(audio))
        self.assertLess(numpy.abs(audio - plain).max(), 1e-4)
        self.assertLess(numpy.abs(audio_eq - equalized).max(), 2e-4)
        self.assertGreater(numpy.abs(audio).max(), 0.05)
