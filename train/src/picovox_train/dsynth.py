# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The differentiable source-filter synthesizer - renders parameter frames (params.py) to audio in PyTorch, batched,
differentiably with respect to every parameter's amplitude and filter effect, at any sample rate. lib/render.js is
the same synthesizer in JavaScript.

    synth = DSynth(sample_rate=22050).to(device)
    audio = synth(values)          # values [B, T, params.COUNT] in parameter units -> audio [B, N]

The design (DDSP-style - no recursive filters, so nothing backpropagates through long IIR recursions):

- Voicing is harmonic (additive) synthesis. Each frame's complex harmonic amplitudes are the product of the voicing
  amplitude, the LF pulse's Fourier coefficients for the frame's Rd (lf.py's table, interpolated in Rd), the source
  shelf's response, and the vocal tract's response (formants, nasal pole and zero, higher poles) - each the analytic
  frequency response of the former lib/klatt.js's filter at k * F0 - with a taper near the Nyquist frequency. The amplitudes are
  interpolated linearly between frames per sample, and summed against the harmonics' phases, so the result is a pulse
  train filtered by the tract, with the tract's phase.
- The glottal phase, cycle by cycle, is computed in float64 on the CPU (it's not differentiated - F0's gradient comes
  through the harmonic amplitudes, which sample the envelope at k * F0): F0 linearly interpolated per sample, with
  jitter (each cycle's period perturbed by a Gaussian draw times the jitter) and shimmer (each cycle's amplitude, the
  draws interpolated across the cycle).
- Noise (aspiration and frication) is white noise scaled per sample by its level (ah, af interpolated - so onsets
  and bursts are as sharp as the frames, as in the former lib/klatt.js), the aspiration also modulated by the glottal cycle
  (NOISE_MODULATION), then filtered in the frequency domain - a short-time Fourier transform (Hann windows of
  noise_fft_size() samples, 16-23 ms, hop a quarter of that), each frame's spectrum multiplied by the filters'
  magnitude responses (the spectral shapes) at the frame's interpolated parameters, and overlap-added.

The noise is torch.randn by default (training) or the counter-based generator of rng.py (noise='counter'), which
lib/render.js also uses, so the two synthesizers can be compared sample by sample.
"""

import math

import numpy
import torch

from . import lf, params, rng


# The noise STFT size for a sample rate - the power of two at or above 16 ms
def noise_fft_size(sample_rate):
    return 1 << math.ceil(math.log2(sample_rate * 0.016))


# The number of samples a sequence of frames renders
def sample_count(frame_count, sample_rate):
    return max(0, math.floor((frame_count - 1) * sample_rate / params.FRAME_RATE))


#
# Filter frequency responses - the former lib/klatt.js's digital filters designed at params.REF_RATE, evaluated at frequencies in
# Hz (w = 2 pi f / REF_RATE). Arguments broadcast; responses are (real, imaginary) pairs or log magnitudes and phases.
#

def _omega(frequency):
    return 2 * math.pi * frequency / params.REF_RATE


# The trigonometry of frequencies shared by every filter response at them - cos w, sin w, cos 2w, sin 2w (computed
# once per set of frequencies rather than once per filter)
def frequency_trig(frequency):
    omega = _omega(frequency)
    return torch.cos(omega), torch.sin(omega), torch.cos(2 * omega), torch.sin(2 * omega)


# A resonator's (unity gain at DC) log magnitude and phase at frequencies - the antiresonator's are their negatives.
# trig is the frequencies' frequency_trig (computed if not given); with phase=False the phase is None.
def resonator_log(frequency, center, bandwidth, trig=None, phase=True):
    radius = torch.exp(-math.pi * bandwidth / params.REF_RATE)
    coef_c = -radius * radius
    coef_b = 2 * radius * torch.cos(_omega(center))
    coef_a = 1 - coef_b - coef_c
    cos1, sin1, cos2, sin2 = frequency_trig(frequency) if trig is None else trig
    den_re = 1 - coef_b * cos1 - coef_c * cos2
    den_im = coef_b * sin1 + coef_c * sin2
    log_magnitude = torch.log(coef_a) - 0.5 * torch.log(den_re * den_re + den_im * den_im)
    return log_magnitude, (-torch.atan2(den_im, den_re) if phase else None)


# The vocal tract's log magnitude and phase at frequencies [..., M] for parameters (a dict of [..., 1] tensors); with
# phase=False the phase is None (and not computed)
def tract_log(frequency, values, trig=None, phase=True):
    trig = frequency_trig(frequency) if trig is None else trig
    poles = [(values[f'f{kx}'], values[f'b{kx}']) for kx in range(1, 7)]
    poles.append((values['fnp'], values['bnp']))
    for kx in range(1, params.HIGHER_POLES + 1):
        center = values['f6'] * (1 + kx * params.HIGHER_POLE_SPACING)
        bandwidth = params.HIGHER_POLE_BANDWIDTH + params.HIGHER_POLE_BANDWIDTH_STEP * (kx - 1)
        poles.append((center, torch.full_like(center, bandwidth)))
    response = _tract_kernel(trig, poles, (values['fnz'], values['bnz']), phase)
    if response is not None:
        return response
    log_magnitude = None
    total_phase = None
    for center, bandwidth in poles:
        pole_magnitude, pole_phase = resonator_log(frequency, center, bandwidth, trig, phase)
        # (the first term alone - adding it to zeros is exact)
        log_magnitude = pole_magnitude if log_magnitude is None else log_magnitude + pole_magnitude
        if phase:
            total_phase = pole_phase if total_phase is None else total_phase + pole_phase
    zero_magnitude, zero_phase = resonator_log(frequency, values['fnz'], values['bnz'], trig, phase)
    return log_magnitude - zero_magnitude, (total_phase - zero_phase if phase else None)


# A biquad's complex response (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2) at frequencies
def _biquad(frequency, b0, b1, b2, a1, a2, trig=None):
    cos1, sin1, cos2, sin2 = frequency_trig(frequency) if trig is None else trig
    num_re = b0 + b1 * cos1 + b2 * cos2
    num_im = -(b1 * sin1 + b2 * sin2)
    den_re = 1 + a1 * cos1 + a2 * cos2
    den_im = -(a1 * sin1 + a2 * sin2)
    den = den_re * den_re + den_im * den_im
    return (num_re * den_re + num_im * den_im) / den, (num_im * den_re - num_re * den_im) / den


# The voicing's high shelf (the former lib/klatt.js HighShelf at SHELF_HZ) - log magnitude and phase
def shelf_log(frequency, gain_db, trig=None):
    real, imag = _biquad(frequency, *_shelf_coefficients(gain_db), trig)
    return 0.5 * torch.log(real * real + imag * imag + 1e-30), torch.atan2(imag, real)


# The high shelf's biquad coefficients (b0, b1, b2, a1, a2)
def _shelf_coefficients(gain_db):
    amp = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * params.SHELF_HZ / params.REF_RATE
    alpha = math.sin(w0) / 2 * math.sqrt(2)
    cos = math.cos(w0)
    sqrt_a = 2 * torch.sqrt(amp) * alpha
    a0 = (amp + 1) - (amp - 1) * cos + sqrt_a
    b0 = amp * ((amp + 1) + (amp - 1) * cos + sqrt_a) / a0
    b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos) / a0
    b2 = amp * ((amp + 1) + (amp - 1) * cos - sqrt_a) / a0
    a1 = 2 * ((amp - 1) - (amp + 1) * cos) / a0
    a2 = ((amp + 1) - (amp - 1) * cos - sqrt_a) / a0
    return b0, b1, b2, a1, a2


# The frication bandpass's (the former lib/klatt.js Bandpass, 0 dB peak) biquad coefficients (b0, b1, b2, a1, a2)
def _bandpass_coefficients(center, bandwidth):
    center_clamped = torch.clamp(center, max=0.45 * params.REF_RATE)
    w0 = _omega(center_clamped)
    alpha = torch.sin(w0) / (2 * torch.clamp(center / bandwidth, min=0.3))
    a0 = 1 + alpha
    zero = torch.zeros_like(alpha)
    return alpha / a0, zero, -alpha / a0, -2 * torch.cos(w0) / a0, (1 - alpha) / a0


# The frication bandpass - complex response
def _bandpass(frequency, center, bandwidth, trig=None):
    return _biquad(frequency, *_bandpass_coefficients(center, bandwidth), trig)


# The frication high-pass's (the former lib/klatt.js HighPass, Butterworth) biquad coefficients (b0, b1, b2, a1, a2)
def _highpass_coefficients(cutoff):
    w0 = _omega(torch.clamp(cutoff, max=0.45 * params.REF_RATE))
    alpha = torch.sin(w0) / math.sqrt(2)
    cos = torch.cos(w0)
    a0 = 1 + alpha
    return (1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0


# The frication high-pass - squared magnitude
def _highpass_power(frequency, cutoff, trig=None):
    real, imag = _biquad(frequency, *_highpass_coefficients(cutoff), trig)
    return real * real + imag * imag


# The frication filter's magnitude (relative to af) - |ab + ap1 BP1^2 + ap2 BP2^2| * |HP|^2
def frication_magnitude(frequency, values, trig=None):
    trig = frequency_trig(frequency) if trig is None else trig
    response = _frication_kernel(trig, values)
    if response is not None:
        return response
    bp1_re, bp1_im = _bandpass(frequency, values['fp1'], values['bp1'], trig)
    bp2_re, bp2_im = _bandpass(frequency, values['fp2'], values['bp2'], trig)
    real = values['ab'] + values['ap1'] * (bp1_re * bp1_re - bp1_im * bp1_im) + values['ap2'] * (bp2_re * bp2_re - bp2_im * bp2_im)
    imag = values['ap1'] * (2 * bp1_re * bp1_im) + values['ap2'] * (2 * bp2_re * bp2_im)
    power = _highpass_power(frequency, values['fhp'], trig)
    power = power * power
    return torch.sqrt(real * real + imag * imag + 1e-30) * torch.sqrt(power + 1e-30)


# The aspiration noise's lowpass (the former lib/klatt.js's one-pole 0.5 / (1 - 0.5 z^-1) at 44.1 kHz) - magnitude
def aspiration_magnitude(frequency):
    return 0.5 / torch.sqrt(1.25 - torch.cos(_omega(frequency)))


# The harmonic taper at frequencies for a sample rate
def harmonic_taper(frequency, sample_rate):
    start = params.HARMONIC_TAPER_START * sample_rate
    end = params.HARMONIC_TAPER_END * sample_rate
    position = torch.clamp((frequency - start) / (end - start), 0, 1)
    return 0.5 + 0.5 * torch.cos(math.pi * position)


# The channel equalizer's natural-log gain at frequencies [B, ...] for gains eq [B, EQ_BAND_COUNT] (dB)
def eq_log(frequency, eq):
    step = math.log(params.EQ_HIGH_HZ / params.EQ_LOW_HZ) / (params.EQ_BAND_COUNT - 1)
    position = torch.clamp(torch.log(torch.clamp(frequency, min=1.0) / params.EQ_LOW_HZ) / step, 0, params.EQ_BAND_COUNT - 1)
    low = torch.clamp(torch.floor(position).long(), max=params.EQ_BAND_COUNT - 2)
    weight = position - low
    shape = frequency.shape
    flat_low = low.reshape(shape[0], -1)
    gains = eq.reshape(eq.shape[0], params.EQ_BAND_COUNT)
    db = torch.gather(gains, 1, flat_low) * (1 - weight.reshape(shape[0], -1)) + \
        torch.gather(gains, 1, flat_low + 1) * weight.reshape(shape[0], -1)
    return (db * (math.log(10) / 20)).reshape(shape)


# Small caches of constant tensors (index grids, windows, frequency grids) by key - host-to-device copies on MPS wait
# for the whole GPU queue, so the synthesizer makes each constant once
_CACHE = {}
_CACHE_LIMIT = 512


def _cached(key, build):
    value = _CACHE.get(key)
    if value is None:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        value = _CACHE[key] = build()
    return value


# The parameter columns that are levels (dB) and the others, and the order that restores the columns from the two
_DB_COLUMNS = [ix for ix, name in enumerate(params.NAMES) if name in params.DB_NAMES]
_OTHER_COLUMNS = [ix for ix, name in enumerate(params.NAMES) if name not in params.DB_NAMES]
_COLUMN_ORDER = [int(ix) for ix in numpy.argsort(_DB_COLUMNS + _OTHER_COLUMNS)]


def _column_indexes(device):
    return _cached(('columns', str(device)), lambda: tuple(torch.tensor(columns, device=device) for columns in
                                                           (_DB_COLUMNS, _OTHER_COLUMNS, _COLUMN_ORDER)))


# Split parameter values [..., COUNT] into a dict of [..., 1] tensors, levels converted to linear amplitudes; linear is
# the values' linear_values, if already computed
def split_values(values, linear=None):
    linear = linear_values(values) if linear is None else linear
    return {name: linear[..., ix:ix + 1] for ix, name in enumerate(params.NAMES)}


# Interpolate frames [B, T, C] at fractional frame positions [M] (numpy) - [B, M, C]; key (optional) caches the
# positions' index and weight tensors
def interpolate_frames(frames, positions, key=None):
    count = frames.shape[1]

    def build():
        index = numpy.clip(numpy.floor(positions).astype(numpy.int64), 0, max(0, count - 2))
        weight = torch.tensor(numpy.clip(positions - index, 0, 1), dtype=frames.dtype, device=frames.device)[None, :, None]
        index_t = torch.tensor(index, device=frames.device)
        return index_t, torch.clamp(index_t + 1, max=count - 1), weight
    if key is None:
        index_t, index_next, weight = build()
    else:
        index_t, index_next, weight = _cached(('interpolate', key, count, str(frames.device), frames.dtype), build)
    return frames[:, index_t] * (1 - weight) + frames[:, index_next] * weight


# Linear (per-sample interpolable) parameter values - levels as linear amplitudes, others as they are
def linear_values(values):
    db_columns, other_columns, order = _column_indexes(values.device)
    levels = 10 ** (values.index_select(-1, db_columns) / 20)
    return torch.cat([levels, values.index_select(-1, other_columns)], dim=-1).index_select(-1, order)


#
# The harmonic sum on Apple GPUs (MPS, float32) - a Metal kernel for the forward and one for the backward, so the
# [B, T - 1, H, K] phases, cosines, and sines (half a gigabyte each for a training batch) are never stored: the same
# float32 operations as the PyTorch path (phase x k, its fraction, x 2 pi, cos, sin - torch's cos and sin are Metal's
# precise ones), summed in another order
#

_HARMONIC_SOURCE = '''
#include <metal_stdlib>
using namespace metal;

// The interval-major sample grid's harmonic sums: out[b, t, h] = (1 - w) (sum_k re[t, k] cos - sum_k im[t, k] sin)
// + w (sum_k re[t + 1, k] cos - sum_k im[t + 1, k] sin), cos and sin of 2 pi frac(phase[b, t, h] k), k = 1 ... K -
// K the active harmonic count (active[0], at most the amplitudes' harmonics)
kernel void harmonic_sum(device float* out [[buffer(0)]], device const float* phase [[buffer(1)]],
                         device const float* amp_re [[buffer(2)]], device const float* amp_im [[buffer(3)]],
                         device const float* weight [[buffer(4)]], constant long4& dims [[buffer(5)]],
                         device const int* active [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    const long batch = dims.x, frames = dims.y, width = dims.z, stride = dims.w;
    const long harmonics = min(stride, (long)active[0]);
    if ((long)index >= batch * (frames - 1) * width) {
        return;
    }
    const long h = index % width;
    const long interval = (index / width) % (frames - 1);
    const long b = index / width / (frames - 1);
    const float fraction = phase[index];
    device const float* re0 = amp_re + (b * frames + interval) * stride;
    device const float* im0 = amp_im + (b * frames + interval) * stride;
    device const float* re1 = re0 + stride;
    device const float* im1 = im0 + stride;
    float cos0 = 0.0f, sin0 = 0.0f, cos1 = 0.0f, sin1 = 0.0f;
    for (long k = 0; k < harmonics; k++) {
        const float cycles = fraction * float(k + 1);
        const float angle = 6.2831855f * (cycles - floor(cycles));
        const float c = precise::cos(angle);
        const float s = precise::sin(angle);
        cos0 = fma(c, re0[k], cos0);
        sin0 = fma(s, im0[k], sin0);
        cos1 = fma(c, re1[k], cos1);
        sin1 = fma(s, im1[k], sin1);
    }
    const float w = weight[interval * width + h];
    out[index] = (cos0 - sin0) * (1.0f - w) + (cos1 - sin1) * w;
}

// The amplitudes' gradients [B, T, K] from the sums' gradient grad[b, t, h] (zero past the active harmonic count)
kernel void harmonic_sum_backward(device float* grad_re [[buffer(0)]], device float* grad_im [[buffer(1)]],
                                  device const float* grad [[buffer(2)]], device const float* phase [[buffer(3)]],
                                  device const float* weight [[buffer(4)]], constant long4& dims [[buffer(5)]],
                                  device const int* active [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    const long batch = dims.x, frames = dims.y, width = dims.z, harmonics = dims.w;
    if ((long)index >= batch * frames * harmonics) {
        return;
    }
    const long k = index % harmonics;
    const long t = (index / harmonics) % frames;
    const long b = index / harmonics / frames;
    float re = 0.0f, im = 0.0f;
    if (k >= (long)active[0]) {
        grad_re[index] = re;
        grad_im[index] = im;
        return;
    }
    // The frame starts interval t (weight 1 - w) and ends interval t - 1 (weight w)
    for (int side = 0; side < 2; side++) {
        const long interval = side == 0 ? t : t - 1;
        if (interval < 0 || interval >= frames - 1) {
            continue;
        }
        const long base = (b * (frames - 1) + interval) * width;
        for (long h = 0; h < width; h++) {
            const float g = grad[base + h];
            if (g == 0.0f) {
                continue;
            }
            const float w = weight[interval * width + h];
            const float scaled = side == 0 ? g * (1.0f - w) : g * w;
            const float cycles = phase[base + h] * float(k + 1);
            const float angle = 6.2831855f * (cycles - floor(cycles));
            re = fma(scaled, precise::cos(angle), re);
            im = fma(-scaled, precise::sin(angle), im);
        }
    }
    grad_re[index] = re;
    grad_im[index] = im;
}
'''
_HARMONIC_LIBRARY = {}


# The harmonic sum's Metal library (None where there is none)
def _harmonic_library():
    if 'library' not in _HARMONIC_LIBRARY:
        try:
            _HARMONIC_LIBRARY['library'] = torch.mps.compile_shader(_HARMONIC_SOURCE)
        except Exception:  # pylint: disable=broad-except
            _HARMONIC_LIBRARY['library'] = None
    return _HARMONIC_LIBRARY['library']


# (an autograd Function needs only forward and backward, not the other abstract methods)
class _HarmonicSum(torch.autograd.Function):  # pylint: disable=abstract-method
    """The harmonic sums [B, T - 1, H] of phases [B, T - 1, H], amplitudes [B, T, K] (real, imaginary), and the
    interpolation weights [T - 1, H] - differentiable in the amplitudes; active is the summed harmonic count (an int32
    device tensor [1], at most K - the harmonics past it are left out), or None for all K"""

    @staticmethod
    def forward(ctx, phase, amp_re, amp_im, weight, active=None):  # pylint: disable=arguments-differ
        batch, frames, harmonics = amp_re.shape
        width = phase.shape[-1]
        dims = [batch, frames, width, harmonics]
        if active is None:
            active = _cached(('active', harmonics, str(amp_re.device)),
                             lambda: torch.tensor([harmonics], dtype=torch.int32, device=amp_re.device))
        out = torch.empty_like(phase)
        _harmonic_library().harmonic_sum(out, phase, amp_re, amp_im, weight, dims, active, threads=out.numel())
        ctx.save_for_backward(phase, weight, active)
        ctx.dims = dims
        return out

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        phase, weight, active = ctx.saved_tensors
        batch, frames, _, harmonics = ctx.dims
        grad_re = torch.empty(batch, frames, harmonics, dtype=grad.dtype, device=grad.device)
        grad_im = torch.empty_like(grad_re)
        _harmonic_library().harmonic_sum_backward(grad_re, grad_im, grad.contiguous(), phase, weight, ctx.dims, active,
                                                  threads=grad_re.numel())
        return None, grad_re, grad_im, None, None


#
# The vocal tract's response on Apple GPUs (MPS, float32) - Metal kernels for tract_log's sum over its resonators (the
# forward per frequency; the backward per resonator, summed over the frequencies), so the dozens of [B, T, M]
# intermediates of each of its 14 resonators are never stored. The resonators' coefficients (b, c, log a - small,
# [B, T, 14]) are made by PyTorch exactly as resonator_log makes them, and differentiated by it.
#

_TRACT_SOURCE = '''
#include <metal_stdlib>
using namespace metal;

// The tract's log magnitude and phase at [B, T, M] (trig per_frame) or [M] frequencies' cos w, sin w, cos 2w, sin 2w -
// the poles' terms (resonators 0 ... R - 2) summed, less the zero's (resonator R - 1), as tract_log sums them
kernel void tract_response(device float* magnitude [[buffer(0)]], device float* phase [[buffer(1)]],
                           device const float* cos1 [[buffer(2)]], device const float* sin1 [[buffer(3)]],
                           device const float* cos2 [[buffer(4)]], device const float* sin2 [[buffer(5)]],
                           device const float* coef_b [[buffer(6)]], device const float* coef_c [[buffer(7)]],
                           device const float* log_a [[buffer(8)]], constant long4& dims [[buffer(9)]],
                           constant long2& flags [[buffer(10)]], uint index [[thread_position_in_grid]]) {
    const long frames = dims.x * dims.y, bins = dims.z, resonators = dims.w;
    if ((long)index >= frames * bins) {
        return;
    }
    const long frame = index / bins;
    const long at = flags.x ? (long)index : (long)index % bins;
    const float c1 = cos1[at], s1 = sin1[at], c2 = cos2[at], s2 = sin2[at];
    float total = 0.0f, angle = 0.0f;
    for (long i = 0; i < resonators; i++) {
        const long coef = frame * resonators + i;
        const float b = coef_b[coef], c = coef_c[coef];
        const float re = 1.0f - b * c1 - c * c2;
        const float im = b * s1 + c * s2;
        const float term = log_a[coef] - 0.5f * precise::log(re * re + im * im);
        total = i == 0 ? term : (i < resonators - 1 ? total + term : total - term);
        if (flags.y) {
            const float turn = -precise::atan2(im, re);
            angle = i == 0 ? turn : (i < resonators - 1 ? angle + turn : angle - turn);
        }
    }
    magnitude[index] = total;
    if (flags.y) {
        phase[index] = angle;
    }
}

// The coefficients' gradients [B, T, R] - each resonator's, summed over the frequencies
kernel void tract_response_backward(device float* grad_b [[buffer(0)]], device float* grad_c [[buffer(1)]],
                                    device float* grad_log_a [[buffer(2)]], device const float* grad_magnitude [[buffer(3)]],
                                    device const float* grad_phase [[buffer(4)]], device const float* cos1 [[buffer(5)]],
                                    device const float* sin1 [[buffer(6)]], device const float* cos2 [[buffer(7)]],
                                    device const float* sin2 [[buffer(8)]], device const float* coef_b [[buffer(9)]],
                                    device const float* coef_c [[buffer(10)]], constant long4& dims [[buffer(11)]],
                                    constant long2& flags [[buffer(12)]], uint index [[thread_position_in_grid]]) {
    const long frames = dims.x * dims.y, bins = dims.z, resonators = dims.w;
    if ((long)index >= frames * resonators) {
        return;
    }
    const long i = index % resonators;
    const long frame = index / resonators;
    const float sign = i < resonators - 1 ? 1.0f : -1.0f;
    const float b = coef_b[index], c = coef_c[index];
    float gb = 0.0f, gc = 0.0f, ga = 0.0f;
    for (long m = 0; m < bins; m++) {
        const long element = frame * bins + m;
        const float gm = grad_magnitude[element];
        const float gp = flags.y ? grad_phase[element] : 0.0f;
        if (gm == 0.0f && gp == 0.0f) {
            continue;
        }
        const long at = flags.x ? element : m;
        const float c1 = cos1[at], s1 = sin1[at], c2 = cos2[at], s2 = sin2[at];
        const float re = 1.0f - b * c1 - c * c2;
        const float im = b * s1 + c * s2;
        const float q = re * re + im * im;
        ga += gm;
        gb += (gm * (re * c1 - im * s1) - gp * (re * s1 + im * c1)) / q;
        gc += (gm * (re * c2 - im * s2) - gp * (re * s2 + im * c2)) / q;
    }
    grad_b[index] = sign * gb;
    grad_c[index] = sign * gc;
    grad_log_a[index] = sign * ga;
}

// The per-frame frequencies' trigonometry's gradients [B, T, M]
kernel void tract_response_trig_backward(device float* grad_c1 [[buffer(0)]], device float* grad_s1 [[buffer(1)]],
                                         device float* grad_c2 [[buffer(2)]], device float* grad_s2 [[buffer(3)]],
                                         device const float* grad_magnitude [[buffer(4)]], device const float* grad_phase [[buffer(5)]],
                                         device const float* cos1 [[buffer(6)]], device const float* sin1 [[buffer(7)]],
                                         device const float* cos2 [[buffer(8)]], device const float* sin2 [[buffer(9)]],
                                         device const float* coef_b [[buffer(10)]], device const float* coef_c [[buffer(11)]],
                                         constant long4& dims [[buffer(12)]], constant long2& flags [[buffer(13)]],
                                         uint index [[thread_position_in_grid]]) {
    const long frames = dims.x * dims.y, bins = dims.z, resonators = dims.w;
    if ((long)index >= frames * bins) {
        return;
    }
    const long frame = index / bins;
    const float gm = grad_magnitude[index];
    const float gp = flags.y ? grad_phase[index] : 0.0f;
    const float c1 = cos1[index], s1 = sin1[index], c2 = cos2[index], s2 = sin2[index];
    float g_c1 = 0.0f, g_s1 = 0.0f, g_c2 = 0.0f, g_s2 = 0.0f;
    for (long i = 0; i < resonators; i++) {
        const float sign = i < resonators - 1 ? 1.0f : -1.0f;
        const float b = coef_b[frame * resonators + i], c = coef_c[frame * resonators + i];
        const float re = 1.0f - b * c1 - c * c2;
        const float im = b * s1 + c * s2;
        const float scale = sign / (re * re + im * im);
        g_c1 += scale * (gm * re - gp * im) * b;
        g_s1 -= scale * (gm * im + gp * re) * b;
        g_c2 += scale * (gm * re - gp * im) * c;
        g_s2 -= scale * (gm * im + gp * re) * c;
    }
    grad_c1[index] = g_c1;
    grad_s1[index] = g_s1;
    grad_c2[index] = g_c2;
    grad_s2[index] = g_s2;
}
'''
_TRACT_LIBRARY = {}


# The tract kernels' Metal library (None where there is none)
def _tract_library():
    if 'library' not in _TRACT_LIBRARY:
        try:
            _TRACT_LIBRARY['library'] = torch.mps.compile_shader(_TRACT_SOURCE)
        except Exception:  # pylint: disable=broad-except
            _TRACT_LIBRARY['library'] = None
    return _TRACT_LIBRARY['library']


# (an autograd Function needs only forward and backward, not the other abstract methods)
class _TractResponse(torch.autograd.Function):  # pylint: disable=abstract-method
    """The tract's log magnitude [B, T, M] (and with_phase its phase) from its frequencies' trigonometry ([B, T, M] or
    [M] each) and its resonators' coefficients b, c, log a [B, T, R] (the last resonator the zero)"""

    @staticmethod
    def forward(ctx, cos1, sin1, cos2, sin2, coef_b, coef_c, log_a, with_phase):  # pylint: disable=arguments-differ
        batch, frames, resonators = coef_b.shape
        bins = cos1.shape[-1]
        per_frame = cos1.dim() == 3
        dims, flags = [batch, frames, bins, resonators], [int(per_frame), int(with_phase)]
        magnitude = torch.empty(batch, frames, bins, dtype=coef_b.dtype, device=coef_b.device)
        phase = torch.empty_like(magnitude) if with_phase else magnitude
        _tract_library().tract_response(magnitude, phase, cos1, sin1, cos2, sin2, coef_b, coef_c, log_a, dims, flags,
                                        threads=magnitude.numel())
        ctx.save_for_backward(cos1, sin1, cos2, sin2, coef_b, coef_c)
        ctx.dims, ctx.flags = dims, flags
        return (magnitude, phase) if with_phase else magnitude

    @staticmethod
    def backward(ctx, grad_magnitude, grad_phase=None):  # pylint: disable=arguments-differ
        cos1, sin1, cos2, sin2, coef_b, coef_c = ctx.saved_tensors
        dims, flags = ctx.dims, list(ctx.flags)
        if grad_magnitude is None:
            grad_magnitude = torch.zeros(dims[:3], dtype=coef_b.dtype, device=coef_b.device)
        grad_magnitude = grad_magnitude.contiguous()
        if grad_phase is None:
            flags[1] = 0
            grad_phase = grad_magnitude
        grad_phase = grad_phase.contiguous()
        grad_b = torch.empty_like(coef_b)
        grad_c = torch.empty_like(coef_b)
        grad_log_a = torch.empty_like(coef_b)
        library = _tract_library()
        library.tract_response_backward(grad_b, grad_c, grad_log_a, grad_magnitude, grad_phase, cos1, sin1, cos2, sin2,
                                        coef_b, coef_c, dims, flags, threads=coef_b.numel())
        grad_trig = [None] * 4
        if flags[0] and any(ctx.needs_input_grad[:4]):
            grad_trig = [torch.empty_like(grad_magnitude) for _ in range(4)]
            library.tract_response_trig_backward(*grad_trig, grad_magnitude, grad_phase, cos1, sin1, cos2, sin2, coef_b,
                                                 coef_c, dims, flags, threads=grad_magnitude.numel())
        return (*grad_trig, grad_b, grad_c, grad_log_a, None)


# The tract's log magnitude and phase by the Metal kernels, or None where they don't apply (not MPS float32, or
# frequencies or parameters not shaped [B, T, M] / [M] and [B, T, 1]) - the resonators' coefficients made as
# resonator_log makes them
def _tract_kernel(trig, poles, zero, phase):
    center = poles[0][0]
    if center.device.type != 'mps' or center.dtype != torch.float32 or center.dim() != 3 or center.shape[-1] != 1 or \
            _tract_library() is None:
        return None
    return _tract_metal(trig, poles, zero, phase)  # pragma: no cover - MPS only (the tests run on the CPU)


# _tract_kernel's response of MPS float32 [B, T, 1] parameters, or None where the frequencies don't apply
def _tract_metal(trig, poles, zero, phase):
    batch, frames = poles[0][0].shape[:2]
    if not (trig[0].dim() == 1 or tuple(trig[0].shape[:2]) == (batch, frames)) or trig[0].dtype != torch.float32:
        return None
    resonators = [*poles, zero]
    centers = torch.cat([torch.broadcast_to(center, (batch, frames, 1)) for center, _ in resonators], dim=-1)
    bandwidths = torch.cat([torch.broadcast_to(bandwidth, (batch, frames, 1)) for _, bandwidth in resonators], dim=-1)
    radius = torch.exp(-math.pi * bandwidths / params.REF_RATE)
    coef_c = -radius * radius
    coef_b = 2 * radius * torch.cos(_omega(centers))
    log_a = torch.log(1 - coef_b - coef_c)
    trig = [torch.broadcast_to(value, (batch, frames, value.shape[-1])).contiguous() if value.dim() == 3 else value.contiguous()
            for value in trig]
    response = _TractResponse.apply(*trig, coef_b.contiguous(), coef_c.contiguous(), log_a.contiguous(), phase)
    return response if phase else (response, None)


#
# The frication filter's magnitude on Apple GPUs (MPS, float32) - a Metal kernel for frication_magnitude's [B, T, M]
# part (the forward per frequency; the backward per frame, summed over the frequencies), so its dozens of [B, T, M]
# intermediates are never stored. The filters' per-frame coefficients ([B, T, 18] - ab, ap1, ap2, and the two
# bandpasses' and the high-pass's b0, b1, b2, a1, a2) are made by PyTorch exactly as frication_magnitude makes them,
# and differentiated by it.
#

_FRICATION_SOURCE = '''
#include <metal_stdlib>
using namespace metal;

// A biquad's complex response at a frequency's cos w, sin w, cos 2w, sin 2w - (re, im), and its numerator and
// denominator (nr, ni, dr, di) and the denominator's squared magnitude q
struct Biquad {
    float re, im, nr, ni, dr, di, q;
};

static Biquad biquad(device const float* coef, const float c1, const float s1, const float c2, const float s2) {
    Biquad out;
    const float b0 = coef[0], b1 = coef[1], b2 = coef[2], a1 = coef[3], a2 = coef[4];
    out.nr = b0 + b1 * c1 + b2 * c2;
    out.ni = -(b1 * s1 + b2 * s2);
    out.dr = 1.0f + a1 * c1 + a2 * c2;
    out.di = -(a1 * s1 + a2 * s2);
    out.q = out.dr * out.dr + out.di * out.di;
    out.re = (out.nr * out.dr + out.ni * out.di) / out.q;
    out.im = (out.ni * out.dr - out.nr * out.di) / out.q;
    return out;
}

// A biquad's coefficients' gradients (added to grad[0 ... 4]) from its response's gradient (gre, gim)
static void biquad_backward(thread float* grad, const Biquad f, const float gre, const float gim, const float c1,
                            const float s1, const float c2, const float s2) {
    const float g_nr = (gre * f.dr - gim * f.di) / f.q;
    const float g_ni = (gre * f.di + gim * f.dr) / f.q;
    const float g_dr = (gre * (f.nr - 2.0f * f.re * f.dr) + gim * (f.ni - 2.0f * f.im * f.dr)) / f.q;
    const float g_di = (gre * (f.ni - 2.0f * f.re * f.di) - gim * (f.nr + 2.0f * f.im * f.di)) / f.q;
    grad[0] += g_nr;
    grad[1] += g_nr * c1 - g_ni * s1;
    grad[2] += g_nr * c2 - g_ni * s2;
    grad[3] += g_dr * c1 - g_di * s1;
    grad[4] += g_dr * c2 - g_di * s2;
}

// The frication magnitude at [B, T, M]: |ab + ap1 BP1^2 + ap2 BP2^2| |HP|^2 at [M] frequencies' trigonometry
kernel void frication(device float* out [[buffer(0)]], device const float* cos1 [[buffer(1)]],
                      device const float* sin1 [[buffer(2)]], device const float* cos2 [[buffer(3)]],
                      device const float* sin2 [[buffer(4)]], device const float* coef [[buffer(5)]],
                      constant long2& dims [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    const long frames = dims.x, bins = dims.y;
    if ((long)index >= frames * bins) {
        return;
    }
    const long m = index % bins;
    device const float* k = coef + (index / bins) * 18;
    const float c1 = cos1[m], s1 = sin1[m], c2 = cos2[m], s2 = sin2[m];
    const Biquad bp1 = biquad(k + 3, c1, s1, c2, s2);
    const Biquad bp2 = biquad(k + 8, c1, s1, c2, s2);
    const Biquad hp = biquad(k + 13, c1, s1, c2, s2);
    const float real = k[0] + k[1] * (bp1.re * bp1.re - bp1.im * bp1.im) + k[2] * (bp2.re * bp2.re - bp2.im * bp2.im);
    const float imag = k[1] * (2.0f * bp1.re * bp1.im) + k[2] * (2.0f * bp2.re * bp2.im);
    float power = hp.re * hp.re + hp.im * hp.im;
    power = power * power;
    out[index] = precise::sqrt(real * real + imag * imag + 1e-30f) * precise::sqrt(power + 1e-30f);
}

// The coefficients' gradients [B, T, 18], summed over the frequencies
kernel void frication_backward(device float* grad_coef [[buffer(0)]], device const float* grad [[buffer(1)]],
                               device const float* cos1 [[buffer(2)]], device const float* sin1 [[buffer(3)]],
                               device const float* cos2 [[buffer(4)]], device const float* sin2 [[buffer(5)]],
                               device const float* coef [[buffer(6)]], constant long2& dims [[buffer(7)]],
                               uint index [[thread_position_in_grid]]) {
    const long frames = dims.x, bins = dims.y;
    if ((long)index >= frames) {
        return;
    }
    device const float* k = coef + index * 18;
    float g[18];
    for (int i = 0; i < 18; i++) {
        g[i] = 0.0f;
    }
    for (long m = 0; m < bins; m++) {
        const float go = grad[index * bins + m];
        if (go == 0.0f) {
            continue;
        }
        const float c1 = cos1[m], s1 = sin1[m], c2 = cos2[m], s2 = sin2[m];
        const Biquad bp1 = biquad(k + 3, c1, s1, c2, s2);
        const Biquad bp2 = biquad(k + 8, c1, s1, c2, s2);
        const Biquad hp = biquad(k + 13, c1, s1, c2, s2);
        const float sq1 = bp1.re * bp1.re - bp1.im * bp1.im, cross1 = 2.0f * bp1.re * bp1.im;
        const float sq2 = bp2.re * bp2.re - bp2.im * bp2.im, cross2 = 2.0f * bp2.re * bp2.im;
        const float real = k[0] + k[1] * sq1 + k[2] * sq2;
        const float imag = k[1] * cross1 + k[2] * cross2;
        const float p = hp.re * hp.re + hp.im * hp.im;
        const float amplitude = precise::sqrt(real * real + imag * imag + 1e-30f);
        const float highpass = precise::sqrt(p * p + 1e-30f);
        // out = amplitude x highpass
        const float g_amplitude = go * highpass;
        const float g_real = g_amplitude * real / amplitude;
        const float g_imag = g_amplitude * imag / amplitude;
        const float g_p = go * amplitude * p / highpass;
        g[0] += g_real;
        g[1] += g_real * sq1 + g_imag * cross1;
        g[2] += g_real * sq2 + g_imag * cross2;
        biquad_backward(g + 3, bp1, 2.0f * k[1] * (g_real * bp1.re + g_imag * bp1.im),
                        2.0f * k[1] * (g_imag * bp1.re - g_real * bp1.im), c1, s1, c2, s2);
        biquad_backward(g + 8, bp2, 2.0f * k[2] * (g_real * bp2.re + g_imag * bp2.im),
                        2.0f * k[2] * (g_imag * bp2.re - g_real * bp2.im), c1, s1, c2, s2);
        biquad_backward(g + 13, hp, 2.0f * g_p * hp.re, 2.0f * g_p * hp.im, c1, s1, c2, s2);
    }
    for (int i = 0; i < 18; i++) {
        grad_coef[index * 18 + i] = g[i];
    }
}
'''
_FRICATION_LIBRARY = {}


# The frication kernels' Metal library (None where there is none)
def _frication_library():
    if 'library' not in _FRICATION_LIBRARY:
        try:
            _FRICATION_LIBRARY['library'] = torch.mps.compile_shader(_FRICATION_SOURCE)
        except Exception:  # pylint: disable=broad-except
            _FRICATION_LIBRARY['library'] = None
    return _FRICATION_LIBRARY['library']


# (an autograd Function needs only forward and backward, not the other abstract methods)
class _FricationResponse(torch.autograd.Function):  # pylint: disable=abstract-method
    """The frication magnitude [B, T, M] from [M] frequencies' trigonometry and the filters' per-frame coefficients
    [B, T, 18] - differentiable in the coefficients"""

    @staticmethod
    def forward(ctx, cos1, sin1, cos2, sin2, coef):  # pylint: disable=arguments-differ
        batch, frames, _ = coef.shape
        bins = cos1.shape[-1]
        dims = [batch * frames, bins]
        out = torch.empty(batch, frames, bins, dtype=coef.dtype, device=coef.device)
        _frication_library().frication(out, cos1, sin1, cos2, sin2, coef, dims, threads=out.numel())
        ctx.save_for_backward(cos1, sin1, cos2, sin2, coef)
        ctx.dims = dims
        return out

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        cos1, sin1, cos2, sin2, coef = ctx.saved_tensors
        grad_coef = torch.empty_like(coef)
        _frication_library().frication_backward(grad_coef, grad.contiguous(), cos1, sin1, cos2, sin2, coef, ctx.dims,
                                                threads=ctx.dims[0])
        return None, None, None, None, grad_coef


# The frication magnitude by the Metal kernels, or None where they don't apply (not MPS float32, or frequencies or
# parameters not shaped [M] and [B, T, 1], or the frequencies differentiated) - the coefficients made as
# frication_magnitude makes them
def _frication_kernel(trig, values):
    center = values['fp1']
    if center.device.type != 'mps' or center.dtype != torch.float32 or center.dim() != 3 or center.shape[-1] != 1 or \
            trig[0].dim() != 1 or trig[0].dtype != torch.float32 or any(value.requires_grad for value in trig) or \
            _frication_library() is None:
        return None
    return _frication_metal(trig, values)  # pragma: no cover - MPS only (the tests run on the CPU)


# _frication_kernel's magnitude of MPS float32 [B, T, 1] parameters at [M] frequencies
def _frication_metal(trig, values):
    batch, frames = values['fp1'].shape[:2]
    columns = [values['ab'], values['ap1'], values['ap2'], *_bandpass_coefficients(values['fp1'], values['bp1']),
               *_bandpass_coefficients(values['fp2'], values['bp2']), *_highpass_coefficients(values['fhp'])]
    coef = torch.cat([torch.broadcast_to(column, (batch, frames, 1)) for column in columns], dim=-1).contiguous()
    return _FricationResponse.apply(*(value.contiguous() for value in trig), coef)


#
# The harmonic amplitudes on Apple GPUs (MPS, float32, a constant F0) - a Metal kernel for harmonic_amplitudes after the
# tract: the high shelf, the channel equalizer, the harmonic taper, the magnitude and phase, the LF source's
# interpolation in Rd, and the complex product (the forward per harmonic; the backward per frame, over the harmonics),
# so their [B, T, K] intermediates are never stored. The shelf's per-frame biquad coefficients are made by PyTorch as
# shelf_log makes them, and the tract's response by its kernel.
#

_AMPLITUDES_SOURCE = '''
#include <metal_stdlib>
using namespace metal;

struct Biquad {
    float re, im, nr, ni, dr, di, q;
};

static Biquad biquad(device const float* coef, const float c1, const float s1, const float c2, const float s2) {
    Biquad out;
    const float b0 = coef[0], b1 = coef[1], b2 = coef[2], a1 = coef[3], a2 = coef[4];
    out.nr = b0 + b1 * c1 + b2 * c2;
    out.ni = -(b1 * s1 + b2 * s2);
    out.dr = 1.0f + a1 * c1 + a2 * c2;
    out.di = -(a1 * s1 + a2 * s2);
    out.q = out.dr * out.dr + out.di * out.di;
    out.re = (out.nr * out.dr + out.ni * out.di) / out.q;
    out.im = (out.ni * out.dr - out.nr * out.di) / out.q;
    return out;
}

// One harmonic's quantities - its frequency's trigonometry, the shelf, the equalizer's interpolation, the taper, the
// magnitude and phase, the filter (fr, fi), and the source (sr, si) and its Rd slope (dsr, dsi)
struct Harmonic {
    float c1, s1, c2, s2, eq_weight, taper, exp_l, magnitude, phase, cos_phase, sin_phase, fr, fi, sr, si, dsr, dsi;
    long eq_low;
    Biquad shelf;
};

// consts: taper start, taper end, the equalizer's log-frequency step; dims: frames (B x T), K, T, the LF table's
// harmonics (its row length), whether there is an equalizer
static Harmonic harmonic(const long frame, const long k, device const float* f0, device const float* av,
                         device const int* low, device const float* weight, device const float* shelf,
                         device const float* tract_magnitude, device const float* tract_phase, device const float* eq,
                         device const float* lf_re, device const float* lf_im, constant float4& consts,
                         constant long4& dims, constant long& has_eq) {
    Harmonic h;
    const long harmonics = dims.y, frames_per_item = dims.z, table = dims.w;
    const long element = frame * harmonics + k;
    const float frequency = f0[frame] * float(k + 1);
    const float omega = 6.2831855f * frequency / 44100.0f;
    const float omega2 = 2.0f * omega;
    h.c1 = precise::cos(omega);
    h.s1 = precise::sin(omega);
    h.c2 = precise::cos(omega2);
    h.s2 = precise::sin(omega2);
    h.shelf = biquad(shelf + frame * 5, h.c1, h.s1, h.c2, h.s2);
    const float shelf_magnitude = 0.5f * precise::log(h.shelf.re * h.shelf.re + h.shelf.im * h.shelf.im + 1e-30f);
    const float shelf_phase = precise::atan2(h.shelf.im, h.shelf.re);
    float log_magnitude = tract_magnitude[element];
    h.eq_low = 0;
    h.eq_weight = 0.0f;
    if (has_eq) {
        const float position = clamp(precise::log(max(frequency, 1.0f) / 60.0f) / consts.z, 0.0f, 23.0f);
        h.eq_low = min((long)floor(position), 22L);
        h.eq_weight = position - float(h.eq_low);
        device const float* gains = eq + (frame / frames_per_item) * 24;
        const float db = gains[h.eq_low] * (1.0f - h.eq_weight) + gains[h.eq_low + 1] * h.eq_weight;
        log_magnitude = log_magnitude + db * 0.11512925f;
    }
    const float taper_position = clamp((frequency - consts.x) / (consts.y - consts.x), 0.0f, 1.0f);
    h.taper = 0.5f + 0.5f * precise::cos(3.1415927f * taper_position);
    h.exp_l = precise::exp(log_magnitude + shelf_magnitude);
    h.magnitude = 2.0f * av[frame] * h.exp_l * h.taper;
    h.phase = tract_phase[element] + shelf_phase;
    h.cos_phase = precise::cos(h.phase);
    h.sin_phase = precise::sin(h.phase);
    h.fr = h.magnitude * h.cos_phase;
    h.fi = h.magnitude * h.sin_phase;
    const float w = weight[frame];
    device const float* row0 = lf_re + (long)low[frame] * table + k + 1;
    device const float* row0i = lf_im + (long)low[frame] * table + k + 1;
    h.sr = row0[0] * (1.0f - w) + row0[table] * w;
    h.si = row0i[0] * (1.0f - w) + row0i[table] * w;
    h.dsr = row0[table] - row0[0];
    h.dsi = row0i[table] - row0i[0];
    return h;
}

kernel void amplitudes(device float* amp_re [[buffer(0)]], device float* amp_im [[buffer(1)]],
                       device const float* f0 [[buffer(2)]], device const float* av [[buffer(3)]],
                       device const int* low [[buffer(4)]], device const float* weight [[buffer(5)]],
                       device const float* shelf [[buffer(6)]], device const float* tract_magnitude [[buffer(7)]],
                       device const float* tract_phase [[buffer(8)]], device const float* eq [[buffer(9)]],
                       device const float* lf_re [[buffer(10)]], device const float* lf_im [[buffer(11)]],
                       constant float4& consts [[buffer(12)]], constant long4& dims [[buffer(13)]],
                       constant long& has_eq [[buffer(14)]], uint index [[thread_position_in_grid]]) {
    const long frames = dims.x, harmonics = dims.y;
    if ((long)index >= frames * harmonics) {
        return;
    }
    const Harmonic h = harmonic(index / harmonics, index % harmonics, f0, av, low, weight, shelf, tract_magnitude,
                                tract_phase, eq, lf_re, lf_im, consts, dims, has_eq);
    amp_re[index] = h.sr * h.fr - h.si * h.fi;
    amp_im[index] = h.sr * h.fi + h.si * h.fr;
}

// Per frame: the voicing's, the Rd weight's, and the shelf coefficients' gradients, the equalizer gains' per-frame
// gradients [B, T, 24], and per harmonic the tract's log magnitude and phase gradients
kernel void amplitudes_backward(device float* grad_av [[buffer(0)]], device float* grad_weight [[buffer(1)]],
                                device float* grad_shelf [[buffer(2)]], device float* grad_eq [[buffer(3)]],
                                device float* grad_tract_magnitude [[buffer(4)]], device float* grad_tract_phase [[buffer(5)]],
                                device const float* grad_re [[buffer(6)]], device const float* grad_im [[buffer(7)]],
                                device const float* f0 [[buffer(8)]], device const float* av [[buffer(9)]],
                                device const int* low [[buffer(10)]], device const float* weight [[buffer(11)]],
                                device const float* shelf [[buffer(12)]], device const float* tract_magnitude [[buffer(13)]],
                                device const float* tract_phase [[buffer(14)]], device const float* eq [[buffer(15)]],
                                device const float* lf_re [[buffer(16)]], device const float* lf_im [[buffer(17)]],
                                constant float4& consts [[buffer(18)]], constant long4& dims [[buffer(19)]],
                                constant long& has_eq [[buffer(20)]], uint index [[thread_position_in_grid]]) {
    const long frames = dims.x, harmonics = dims.y;
    if ((long)index >= frames) {
        return;
    }
    const long frame = index;
    float g_av = 0.0f, g_weight = 0.0f;
    float g_shelf[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float g_eq[24];
    for (int i = 0; i < 24; i++) {
        g_eq[i] = 0.0f;
    }
    for (long k = 0; k < harmonics; k++) {
        const long element = frame * harmonics + k;
        const float gre = grad_re[element], gim = grad_im[element];
        if (gre == 0.0f && gim == 0.0f) {
            grad_tract_magnitude[element] = 0.0f;
            grad_tract_phase[element] = 0.0f;
            continue;
        }
        const Harmonic h = harmonic(frame, k, f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq, lf_re, lf_im,
                                    consts, dims, has_eq);
        // amp = source x filter (complex)
        const float g_sr = gre * h.fr + gim * h.fi;
        const float g_si = gim * h.fr - gre * h.fi;
        const float g_fr = gre * h.sr + gim * h.si;
        const float g_fi = gim * h.sr - gre * h.si;
        g_weight += g_sr * h.dsr + g_si * h.dsi;
        // filter = magnitude (cos, sin)(phase)
        const float g_magnitude = g_fr * h.cos_phase + g_fi * h.sin_phase;
        const float g_phase = g_fi * h.fr - g_fr * h.fi;
        // magnitude = 2 av exp(L) taper, L = tract + eq + shelf
        g_av += g_magnitude * 2.0f * h.exp_l * h.taper;
        const float g_l = g_magnitude * h.magnitude;
        grad_tract_magnitude[element] = g_l;
        grad_tract_phase[element] = g_phase;
        if (has_eq) {
            const float g_db = g_l * 0.11512925f;
            g_eq[h.eq_low] += g_db * (1.0f - h.eq_weight);
            g_eq[h.eq_low + 1] += g_db * h.eq_weight;
        }
        // the shelf's log magnitude and phase
        const float p = h.shelf.re * h.shelf.re + h.shelf.im * h.shelf.im;
        const float g_shelf_re = g_l * h.shelf.re / (p + 1e-30f) - g_phase * h.shelf.im / p;
        const float g_shelf_im = g_l * h.shelf.im / (p + 1e-30f) + g_phase * h.shelf.re / p;
        const Biquad f = h.shelf;
        const float g_nr = (g_shelf_re * f.dr - g_shelf_im * f.di) / f.q;
        const float g_ni = (g_shelf_re * f.di + g_shelf_im * f.dr) / f.q;
        const float g_dr = (g_shelf_re * (f.nr - 2.0f * f.re * f.dr) + g_shelf_im * (f.ni - 2.0f * f.im * f.dr)) / f.q;
        const float g_di = (g_shelf_re * (f.ni - 2.0f * f.re * f.di) - g_shelf_im * (f.nr + 2.0f * f.im * f.di)) / f.q;
        g_shelf[0] += g_nr;
        g_shelf[1] += g_nr * h.c1 - g_ni * h.s1;
        g_shelf[2] += g_nr * h.c2 - g_ni * h.s2;
        g_shelf[3] += g_dr * h.c1 - g_di * h.s1;
        g_shelf[4] += g_dr * h.c2 - g_di * h.s2;
    }
    grad_av[frame] = g_av;
    grad_weight[frame] = g_weight;
    for (int i = 0; i < 5; i++) {
        grad_shelf[frame * 5 + i] = g_shelf[i];
    }
    for (int i = 0; i < 24; i++) {
        grad_eq[frame * 24 + i] = g_eq[i];
    }
}
'''
_AMPLITUDES_LIBRARY = {}


# The harmonic amplitudes' Metal library for values (None where there is none, or the values aren't MPS float32)
def _amplitudes_library_for(values):
    if values.device.type != 'mps' or values.dtype != torch.float32:
        return None
    if 'library' not in _AMPLITUDES_LIBRARY:
        try:
            _AMPLITUDES_LIBRARY['library'] = torch.mps.compile_shader(_AMPLITUDES_SOURCE)
        except Exception:  # pylint: disable=broad-except
            _AMPLITUDES_LIBRARY['library'] = None
    return _AMPLITUDES_LIBRARY['library']


# (an autograd Function needs only forward and backward, not the other abstract methods)
class _HarmonicAmplitudes(torch.autograd.Function):  # pylint: disable=abstract-method
    """The complex harmonic amplitudes [B, T, K] (real, imaginary) from the frames' F0 [B, T] (constant), voicing
    (linear) [B, T], LF table row (int32) and Rd weight [B, T], shelf coefficients [B, T, 5], the tract's log magnitude
    and phase [B, T, K], and the channel equalizer's gains (dB) [B, 24] or None - differentiable in all but the F0 and
    the row"""

    @staticmethod
    def forward(ctx, f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq, lf_re, lf_im, sample_rate):  # pylint: disable=arguments-differ
        batch, frames, harmonics = tract_magnitude.shape
        dims = [batch * frames, harmonics, frames, lf_re.shape[1]]
        consts = [params.HARMONIC_TAPER_START * sample_rate, params.HARMONIC_TAPER_END * sample_rate,
                  math.log(params.EQ_HIGH_HZ / params.EQ_LOW_HZ) / (params.EQ_BAND_COUNT - 1), 0.0]
        has_eq = int(eq is not None)
        eq_buffer = eq if eq is not None else av
        amp_re = torch.empty_like(tract_magnitude)
        amp_im = torch.empty_like(tract_magnitude)
        _amplitudes_library_for(av).amplitudes(amp_re, amp_im, f0, av, low, weight, shelf, tract_magnitude, tract_phase,
                                               eq_buffer, lf_re, lf_im, consts, dims, has_eq, threads=amp_re.numel())
        ctx.save_for_backward(f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq_buffer, lf_re, lf_im)
        ctx.dims, ctx.consts, ctx.has_eq = dims, consts, has_eq
        return amp_re, amp_im

    @staticmethod
    def backward(ctx, grad_re, grad_im):  # pylint: disable=arguments-differ
        f0, av, low, weight, shelf, tract_magnitude, tract_phase, eq_buffer, lf_re, lf_im = ctx.saved_tensors
        batch, frames, _ = tract_magnitude.shape
        grad_re = torch.zeros_like(tract_magnitude) if grad_re is None else grad_re.contiguous()
        grad_im = torch.zeros_like(tract_magnitude) if grad_im is None else grad_im.contiguous()
        grad_av = torch.empty_like(av)
        grad_weight = torch.empty_like(weight)
        grad_shelf = torch.empty_like(shelf)
        grad_eq = torch.empty(batch, frames, params.EQ_BAND_COUNT, dtype=av.dtype, device=av.device)
        grad_tract_magnitude = torch.empty_like(tract_magnitude)
        grad_tract_phase = torch.empty_like(tract_magnitude)
        _amplitudes_library_for(av).amplitudes_backward(
            grad_av, grad_weight, grad_shelf, grad_eq, grad_tract_magnitude, grad_tract_phase, grad_re, grad_im, f0, av, low,
            weight, shelf, tract_magnitude, tract_phase, eq_buffer, lf_re, lf_im, ctx.consts, ctx.dims, ctx.has_eq,
            threads=ctx.dims[0])
        grad_eq = grad_eq.sum(dim=1) if ctx.has_eq else None
        return (None, grad_av, None, grad_weight, grad_shelf, grad_tract_magnitude, grad_tract_phase, grad_eq, None, None,
                None)


# Gaussian draws [B, max count] of a stream for each seed (rows zero-padded past their counts)
def _row_draws(seeds, stream, counts):
    draws = numpy.zeros((len(seeds), int(max(counts))))
    for ix, (seed, count) in enumerate(zip(seeds, counts)):
        draws[ix, :count] = rng.gauss(rng.stream_seed(seed, stream), 0, int(count))
    return draws


class DSynth(torch.nn.Module):
    """The differentiable synthesizer - see the module documentation"""

    def __init__(self, sample_rate=22050, noise='torch'):
        super().__init__()
        self.sample_rate = sample_rate
        self.noise = noise
        self.noise_fft = noise_fft_size(sample_rate)
        self.noise_hop = self.noise_fft // 4
        table = lf.harmonic_table()
        self.register_buffer('lf_re', torch.tensor(table.real, dtype=torch.float32), persistent=False)
        self.register_buffer('lf_im', torch.tensor(table.imag, dtype=torch.float32), persistent=False)
        window = 0.5 - 0.5 * numpy.cos(2 * math.pi * numpy.arange(self.noise_fft) / self.noise_fft)
        self.register_buffer('window', torch.tensor(window, dtype=torch.float32), persistent=False)
        # The host tensors of the latest uploads (non-blocking copies read them later) - kept until the next forward's
        # synchronizing read, by when the copies are done (a deferred plan's are kept by the caller - see forward)
        self._uploads = None
        # The plans made ahead that didn't match the frames (and were made again), and the deferred plans' count of
        # mismatches (a device tensor, read by deferred_plan_mismatches)
        self.plan_mismatches = 0
        self._deferred_mismatches = None

    def plan_cycles(self, f0_frames, jitter_frames, seeds):
        """The glottal cycle plan of a batch (numpy, float64, not differentiated) from its frames' F0 and jitter
        [B, T] (float64) and its seeds - the per-sample frame index and interpolation weight [N] and, per item [B, N],
        the phase within the glottal cycle (fraction) and the shimmer draws interpolated across the cycles. It needs
        nothing the network predicts when the F0 is the recording's (training), so it can be made ahead, on another
        thread; finish_plan adds what depends on the voicing and Rd. N must be positive."""
        sample_rate = self.sample_rate
        batch, frame_count = f0_frames.shape
        count = sample_count(frame_count, sample_rate)
        positions = numpy.arange(count) * params.FRAME_RATE / sample_rate
        index = numpy.clip(numpy.floor(positions).astype(numpy.int64), 0, frame_count - 2)
        weight = positions - index

        def per_sample(columns):
            return columns[:, index] * (1 - weight) + columns[:, index + 1] * weight

        f0 = numpy.maximum(params.F0_MIN, per_sample(f0_frames))
        jitter = per_sample(jitter_frames)

        # The unperturbed glottal phase (cycles), the jitter of each of its cycles, and the perturbed phase
        start = numpy.zeros((batch, 1))
        phase0 = numpy.concatenate([start, numpy.cumsum(f0[:, :-1] / sample_rate, axis=1)], axis=1)
        cycle0 = numpy.floor(phase0).astype(numpy.int64)
        jitter_draws = _row_draws(seeds, rng.STREAM_JITTER, cycle0[:, -1] + 2)
        frequency = f0 / numpy.maximum(0.5, 1 + jitter * numpy.take_along_axis(jitter_draws, cycle0, axis=1))
        phase = numpy.concatenate([start, numpy.cumsum(frequency[:, :-1] / sample_rate, axis=1)], axis=1)
        cycle = numpy.floor(phase).astype(numpy.int64)
        fraction = phase - cycle

        # The shimmer draws, interpolated across each cycle
        shimmer_draws = _row_draws(seeds, rng.STREAM_SHIMMER, cycle[:, -1] + 3)
        shimmer = numpy.take_along_axis(shimmer_draws, cycle, axis=1) * (1 - fraction) + \
            numpy.take_along_axis(shimmer_draws, cycle + 1, axis=1) * fraction
        return {'seeds': list(seeds), 'frameCount': frame_count, 'count': count, 'index': index, 'weight': weight,
                'f0': f0_frames, 'jitter': jitter_frames, 'fraction': fraction, 'shimmer': shimmer}

    def finish_plan(self, plan, av_frames, rd_frames):
        """Complete a cycle plan with the frames' voicing and Rd [B, T] (float64) - the glottis's open phase per sample
        and the harmonic count"""
        index, weight = plan['index'], plan['weight']
        rd = rd_frames[:, index] * (1 - weight) + rd_frames[:, index + 1] * weight
        plan['open'] = plan['fraction'] < lf.open_fraction(rd)
        plan['harmonics'] = self._harmonic_count(plan['f0'], av_frames)
        return plan

    def plan_host_tensors(self, plan, dtype=torch.float32):
        """A cycle plan's per-item host tensors, ready to upload - the cycle phases on the voicing's sample grid and the
        shimmer (in the values' dtype), and the noise sequences (float32). Not the open phase (finish_plan)."""
        grid, _, _, _ = self._sample_grid(plan['count'], plan['frameCount'])
        noises = [self._noise_host(plan['count'], seed) for seed in plan['seeds']]
        return {
            'fraction': torch.tensor(plan['fraction'][:, grid], dtype=dtype), 'shimmer': torch.tensor(plan['shimmer'], dtype=dtype),
            'aspiration': torch.stack([noise[0] for noise in noises]), 'frication': torch.stack([noise[1] for noise in noises]),
            'f0': torch.tensor(plan['f0'], dtype=dtype), 'jitter': torch.tensor(plan['jitter'], dtype=dtype)
        }

    def deferred_plan_mismatches(self):
        """The plans made ahead and finished on the device that didn't match their frames (counted on the device -
        reading it synchronizes); such a rendering used the plan's F0 and jitter, not the frames'"""
        return 0 if self._deferred_mismatches is None else int(self._deferred_mismatches.cpu())

    # Whether forward can finish a plan made ahead on the device, without reading the frames (MPS float32, the
    # harmonic sum's Metal kernel, and the plan's host tensors in the values' dtype)
    def _can_defer(self, values, plan):
        host = plan.get('host')
        return values.device.type == 'mps' and values.dtype == torch.float32 and _harmonic_library() is not None and \
            host is not None and host['fraction'].dtype == values.dtype and 'f0' in host

    # A plan made ahead, finished on the device without reading the frames (no synchronization): its tensors (the
    # open phase from the frames' Rd, and the active harmonic count from their voicing and F0 - dsynth.py
    # _harmonic_count's, as an int32 device tensor), and the harmonics to make (an upper bound of the active count from
    # the plan's F0 on the host, rounded up to a multiple of 16, so the shapes repeat). The frames' F0 and jitter are
    # checked against the plan's on the device (deferred_plan_mismatches). The plan's host tensors must live until the
    # copies are done - the caller keeps them (plan['host']) until a later synchronization.
    def _deferred_plan(self, values, plan, grid):
        device = values.device
        host = plan['host']
        self._uploads = host
        tensors = {key: value.to(device, non_blocking=True) for key, value in host.items()}
        f0 = values[..., params.INDEX['f0']]
        jitter = values[..., params.INDEX['jitter']]
        mismatch = ((f0 != tensors['f0']) | (jitter != tensors['jitter'])).any().int()
        self._deferred_mismatches = mismatch if self._deferred_mismatches is None else self._deferred_mismatches + mismatch

        # The open phase - the cycle fraction below lf.open_fraction of the per-sample Rd
        batch = values.shape[0]
        index, weight = grid['index'], grid['weight']
        rd = values[:, index, params.INDEX['rd']] * (1 - weight) + values[:, index + 1, params.INDEX['rd']] * weight
        fraction = tensors['fraction'].reshape(batch, -1).index_select(1, grid['cells'])
        tensors['open'] = fraction < torch.clamp(0.45 + 0.17 * rd, max=0.95)

        # The active harmonic count (_harmonic_count), and its upper bound (every frame voiced)
        voiced = values[..., params.INDEX['av']] > params.FLOOR_DB + 30
        lowest = torch.where(voiced, torch.clamp(f0, min=params.F0_MIN), math.inf).amin()
        lowest = torch.where(torch.isinf(lowest), 1000.0, lowest)
        taper_end = params.HARMONIC_TAPER_END * self.sample_rate
        tensors['harmonics'] = torch.clamp(torch.ceil(taper_end / lowest), max=params.LF_HARMONICS - 1).int().reshape(1)
        bound = max(self._harmonic_count(plan['f0'], numpy.full(plan['f0'].shape, params.FLOOR_DB + 31)),
                    self._harmonic_count(plan['f0'], numpy.full(plan['f0'].shape, params.FLOOR_DB)))
        harmonics = min(params.LF_HARMONICS - 1, 16 * math.ceil(bound / 16))
        return tensors, harmonics

    # Upload a finished plan's tensors to a device - non-blocking (on MPS a blocking copy waits for the whole GPU
    # queue), so the host tensors are kept until the next forward's synchronizing read
    def _plan_tensors(self, plan, device, dtype):
        host = plan.get('host')
        if host is None or host['fraction'].dtype != dtype:
            host = self.plan_host_tensors(plan, dtype)
        host = {**{key: value for key, value in host.items() if key not in ('f0', 'jitter')}, 'open': torch.from_numpy(plan['open'])}
        self._uploads = host
        return {key: value.to(device, non_blocking=True) for key, value in host.items()}

    # The voicing's sample grid for a sample count and frame count - the samples grouped by frame interval, a grid
    # [T - 1, H] of sample indices (H the most samples in an interval): (the clamped grid, its interval-major
    # positions of the samples in order, H, the samples' frame index and weight)
    def _sample_grid(self, count, frame_count):
        def build():
            positions = numpy.arange(count) * params.FRAME_RATE / self.sample_rate
            index = numpy.clip(numpy.floor(positions).astype(numpy.int64), 0, frame_count - 2)
            starts = numpy.searchsorted(index, numpy.arange(frame_count - 1))
            ends = numpy.searchsorted(index, numpy.arange(frame_count - 1), side='right')
            width = int((ends - starts).max())
            grid = starts[:, None] + numpy.arange(width)[None, :]
            valid = grid < ends[:, None]
            # Every sample is in exactly one interval, in order, so the valid grid cells are the samples
            assert numpy.array_equal(grid[valid], numpy.arange(count))
            return numpy.minimum(grid, count - 1), numpy.flatnonzero(valid), width, (index, positions - index)
        return _cached(('grid', self.sample_rate, count, frame_count), build)

    # The sample grid's device tensors - the per-sample frame index and weight, the weight on the grid, and the valid
    # cells' positions
    def _grid_tensors(self, count, frame_count, device, dtype):
        def build():
            grid, cells, _, (index, weight) = self._sample_grid(count, frame_count)
            return {'index': torch.tensor(index, device=device), 'weight': torch.tensor(weight, dtype=dtype, device=device),
                    'gridWeight': torch.tensor(weight[grid], dtype=dtype, device=device)[None],
                    'cells': torch.tensor(cells, device=device)}
        return _cached(('gridTensors', self.sample_rate, count, frame_count, str(device), dtype), build)

    # The noise sequences (unit variance at the reference rate's level) for one item - host float32 tensors
    def _noise_host(self, count, seed):
        scale = math.sqrt(self.sample_rate / params.REF_RATE)
        if self.noise == 'counter':
            aspiration = rng.gauss(rng.stream_seed(seed, rng.STREAM_ASPIRATION), 0, count)
            frication = rng.gauss(rng.stream_seed(seed, rng.STREAM_FRICATION), 0, count)
            return (torch.tensor(aspiration * scale, dtype=torch.float32), torch.tensor(frication * scale, dtype=torch.float32))
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(2, count, generator=generator) * scale
        return noise[0], noise[1]

    # The harmonic count for a batch's frames' F0 and voicing [B, T] - enough for the lowest voiced F0
    def _harmonic_count(self, f0_frames, av_frames):
        voiced = av_frames > params.FLOOR_DB + 30
        f0 = numpy.maximum(params.F0_MIN, f0_frames)
        lowest = f0[voiced].min() if voiced.any() else 1000.0
        return int(min(params.LF_HARMONICS - 1, math.ceil(params.HARMONIC_TAPER_END * self.sample_rate / lowest)))

    # The frames' complex harmonic amplitudes [B, T, K] (real, imaginary) and the harmonic numbers; linear is the
    # values' linear_values, if already computed; constant_f0 - the frames' F0 is a constant (a recording's), so the
    # amplitudes aren't differentiated in it
    def harmonic_amplitudes(self, values, harmonics, eq=None, linear=None, constant_f0=False):
        split = split_values(values, linear)
        k = torch.arange(1, harmonics + 1, device=values.device, dtype=values.dtype)
        f0 = torch.clamp(split['f0'].detach() if constant_f0 else split['f0'], min=params.F0_MIN)
        frequency = f0 * k
        trig = frequency_trig(frequency)

        # The LF pulse's coefficients, interpolated in Rd
        grid = (torch.clamp(split['rd'][..., 0], params.LF_RD_MIN, params.LF_RD_MAX) - params.LF_RD_MIN) / params.LF_RD_STEP
        low = torch.clamp(torch.floor(grid).long(), 0, self.lf_re.shape[0] - 2)
        weight = (grid - low)[..., None]
        table_re = self.lf_re[:, 1:harmonics + 1]
        table_im = self.lf_im[:, 1:harmonics + 1]
        source_re = table_re[low] * (1 - weight) + table_re[low + 1] * weight
        source_im = table_im[low] * (1 - weight) + table_im[low + 1] * weight

        # The source shelf and the tract, as log magnitude and phase (on Apple GPUs with a constant F0, everything but
        # the tract in a Metal kernel)
        tract_magnitude, tract_phase = tract_log(frequency, split, trig)
        if constant_f0 and _amplitudes_library_for(values) is not None:
            shelf = torch.cat([torch.broadcast_to(coefficient, split['shelf'].shape)
                               for coefficient in _shelf_coefficients(split['shelf'])], dim=-1)
            return _HarmonicAmplitudes.apply(f0[..., 0].contiguous(), split['av'][..., 0].contiguous(), low.int().contiguous(),
                                             weight[..., 0].contiguous(), shelf.contiguous(), tract_magnitude.contiguous(),
                                             tract_phase.contiguous(), None if eq is None else eq.contiguous(), self.lf_re, self.lf_im,
                                             self.sample_rate)
        shelf_magnitude, shelf_phase = shelf_log(frequency, split['shelf'], trig)
        if eq is not None:
            tract_magnitude = tract_magnitude + eq_log(frequency, eq)
        magnitude = 2 * split['av'] * torch.exp(tract_magnitude + shelf_magnitude) * harmonic_taper(frequency, self.sample_rate)
        phase = tract_phase + shelf_phase
        filter_re = magnitude * torch.cos(phase)
        filter_im = magnitude * torch.sin(phase)
        return source_re * filter_re - source_im * filter_im, source_re * filter_im + source_im * filter_re

    # The harmonic (voicing) signal [B, N]
    def _voicing(self, values, linear, tensors, grid, harmonics, eq=None, constant_f0=False):
        batch = values.shape[0]
        amp_re, amp_im = self.harmonic_amplitudes(values, harmonics, eq, linear, constant_f0)
        # (a deferred plan's harmonic count is on the device - the amplitudes are made for its upper bound)
        active = tensors.get('harmonics')

        # Sum the harmonics over the samples grouped by frame interval: Re(sum_k A_k exp(2 pi i k phase)), the
        # amplitudes interpolated between frames - on Apple GPUs in the Metal kernels, else the interval's two frames'
        # sums in one product each
        weight = grid['gridWeight']
        if values.device.type == 'mps' and values.dtype == torch.float32 and _harmonic_library() is not None:
            # (MPS only - the tests run on the CPU)
            signal = _HarmonicSum.apply(tensors['fraction'].contiguous(), amp_re.contiguous(), amp_im.contiguous(),  # pragma: no cover
                                        weight[0].contiguous(), active)
        else:
            k = torch.arange(1, harmonics + 1, device=values.device, dtype=values.dtype)
            angle = tensors['fraction'][..., None] * k
            angle = 2 * math.pi * torch.remainder(angle, 1.0)
            cos = torch.cos(angle)
            sin = angle.sin_()
            sums = torch.matmul(cos, torch.stack([amp_re[:, :-1], amp_re[:, 1:]], dim=-1)) - \
                torch.matmul(sin, torch.stack([amp_im[:, :-1], amp_im[:, 1:]], dim=-1))
            signal = sums[..., 0] * (1 - weight) + sums[..., 1] * weight
        return signal.reshape(batch, -1).index_select(1, grid['cells'])

    # The noise (aspiration and frication) signal [B, N]
    def _noise_signal(self, values, linear, tensors, grid, count, eq=None):
        device = values.device
        batch = values.shape[0]
        fft_size, hop = self.noise_fft, self.noise_hop

        # The noise levels per sample (so onsets are as sharp as the frames), and the aspiration's glottal modulation -
        # full while open, reduced by the voicing while closed
        index, weight = grid['index'], grid['weight']
        def per_sample(name):
            return linear[:, index, params.INDEX[name]] * (1 - weight) + linear[:, index + 1, params.INDEX[name]] * weight
        av = per_sample('av')
        modulation = torch.where(tensors['open'], torch.ones_like(av), 1 - params.NOISE_MODULATION * torch.clamp(2 * av, max=1))
        aspiration = tensors['aspiration'] * per_sample('ah') * modulation
        frication = tensors['frication'] * per_sample('af')

        # Frame the noise - frame j centered on sample j * hop
        frames_count = count // hop + 1
        padded_length = (frames_count - 1) * hop + fft_size
        def framed(signal):
            padded = torch.nn.functional.pad(signal, (fft_size // 2, padded_length - count - fft_size // 2))
            return padded.unfold(1, fft_size, hop) * self.window
        spectrum_aspiration = torch.fft.rfft(framed(aspiration))
        spectrum_frication = torch.fft.rfft(framed(frication))

        # The filters at each noise frame's parameters
        positions = numpy.arange(frames_count) * hop * params.FRAME_RATE / self.sample_rate
        at_frames = interpolate_frames(linear, positions, key=('noise', self.sample_rate, frames_count, hop))
        split = {name: at_frames[..., ix:ix + 1] for ix, name in enumerate(params.NAMES)}
        frequency, trig, aspiration_filter = _cached(('noiseFrequency', self.sample_rate, fft_size, str(device), values.dtype),
                                                     lambda: self._noise_frequency(device, values.dtype))
        tract_magnitude, _ = tract_log(frequency, split, trig, phase=False)
        filter_aspiration = aspiration_filter * torch.exp(tract_magnitude)
        filter_frication = frication_magnitude(frequency, split, trig)
        if eq is not None:
            channel = torch.exp(eq_log(frequency.expand(batch, -1), eq))[:, None, :]
            filter_aspiration = filter_aspiration * channel
            filter_frication = filter_frication * channel
        filtered = spectrum_aspiration * filter_aspiration + spectrum_frication * filter_frication
        frames_out = torch.fft.irfft(filtered, n=fft_size) * self.window

        # Overlap-add (hop is a quarter frame) and normalize by the windows' squared sum
        quarters = frames_out.reshape(batch, frames_count, 4, hop)
        total = torch.zeros(batch, frames_count + 3, hop, dtype=values.dtype, device=device)
        for quarter in range(4):
            total = total + torch.nn.functional.pad(quarters[:, :, quarter], (0, 0, quarter, 3 - quarter))
        window_sum = _cached(('windowSum', fft_size, frames_count, str(device), values.dtype),
                             lambda: self._window_sum(frames_count, values.dtype))
        signal = (total / torch.clamp(window_sum, min=1e-3)).reshape(batch, -1)
        return signal[:, fft_size // 2:fft_size // 2 + count]

    # The noise STFT's bin frequencies, their trigonometry, and the aspiration filter at them
    def _noise_frequency(self, device, dtype):
        frequency = torch.arange(self.noise_fft // 2 + 1, device=device, dtype=dtype) * self.sample_rate / self.noise_fft
        return frequency, frequency_trig(frequency), aspiration_magnitude(frequency)

    # The overlapped windows' squared sum per output sample [frames + 3, hop]
    def _window_sum(self, frames_count, dtype):
        hop = self.noise_hop
        window_sum = torch.zeros(frames_count + 3, hop, dtype=dtype, device=self.window.device)
        window_squared = (self.window * self.window).reshape(4, hop)
        for quarter in range(4):
            window_sum[quarter:quarter + frames_count] += window_squared[quarter]
        return window_sum

    def forward(self, values, seed=1, return_parts=False, eq=None, plan=None, constant_f0=False):
        """Render parameter values [B, T, COUNT] (or [T, COUNT]) to audio [B, N] (or [N]); seed is an int or a list; eq is
        the optional channel equalizer's gains (dB) [B, EQ_BAND_COUNT] (or [EQ_BAND_COUNT]); plan is the
        optional cycle plan made ahead (plan_cycles, of these values' F0 and jitter and these seeds, optionally with its
        plan_host_tensors as 'host') - it leaves only a few columns to read from the device here; constant_f0 says the
        values' F0 is a constant (a recording's, as training renders) - the rendering isn't differentiated in
        it (the F0 column's gradient is zero), which saves the harmonic amplitudes' frequency gradients"""
        squeeze = values.dim() == 2
        if squeeze:
            values = values[None]
            if eq is not None and eq.dim() == 1:
                eq = eq[None]
        batch, frame_count, _ = values.shape
        seeds = list(seed) if isinstance(seed, (list, tuple)) else [int(seed) + ix for ix in range(batch)]
        count = sample_count(frame_count, self.sample_rate)
        if count == 0:
            empty = values.new_zeros(batch, 0)
            return empty[0] if squeeze else empty

        # The cycle plan - the one read from the device (it synchronizes); the host tensors of the previous call's
        # uploads are released after it - or a plan made ahead, finished on the device (no synchronization)
        grid = self._grid_tensors(count, frame_count, values.device, values.dtype)
        linear = linear_values(values)
        tensors = None
        if plan is not None and self._can_defer(values, plan) and plan['seeds'] == seeds:
            tensors, harmonics = self._deferred_plan(values, plan, grid)
        elif plan is None:
            frames_cpu = values.detach().cpu().double().numpy()  # (MPS: .to('cpu', float64) in one step gives zeros)
            plan = self.plan_cycles(frames_cpu[..., params.INDEX['f0']], frames_cpu[..., params.INDEX['jitter']], seeds)
            self.finish_plan(plan, frames_cpu[..., params.INDEX['av']], frames_cpu[..., params.INDEX['rd']])
        else:
            # (the plan is checked against the frames' F0 and jitter, read with the voicing and Rd - if they differ it's
            # made again here, so a plan made ahead never changes the rendering)
            names = ('av', 'rd', 'f0', 'jitter')
            columns = torch.stack([values.detach()[..., params.INDEX[name]] for name in names], dim=-1).cpu().double().numpy()
            av, rd, f0, jitter = (columns[..., ix] for ix in range(len(names)))
            if plan['seeds'] != seeds or not numpy.array_equal(plan['f0'], f0) or not numpy.array_equal(plan['jitter'], jitter):
                self.plan_mismatches += 1
                plan = self.plan_cycles(f0, jitter, seeds)
            plan = self.finish_plan(dict(plan), av, rd)
        if tensors is None:
            tensors = self._plan_tensors(plan, values.device, values.dtype)
            harmonics = plan['harmonics']

        # The voicing, with shimmer, and the noise
        voicing = self._voicing(values, linear, tensors, grid, harmonics, eq, constant_f0)
        index, weight = grid['index'], grid['weight']
        shimmer_amount = linear[:, index, params.INDEX['shimmer']] * (1 - weight) + linear[:, index + 1, params.INDEX['shimmer']] * weight
        voicing = voicing * torch.relu(1 + shimmer_amount * tensors['shimmer'])
        noise = self._noise_signal(values, linear, tensors, grid, count, eq)
        audio = voicing + noise
        if return_parts:
            parts = {'audio': audio, 'voicing': voicing, 'noise': noise}
            return {key: value[0] for key, value in parts.items()} if squeeze else parts
        return audio[0] if squeeze else audio
