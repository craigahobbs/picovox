# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The differentiable source-filter synthesizer - renders parameter frames (nn/params.py) to audio in PyTorch, batched,
differentiably with respect to every parameter's amplitude and filter effect, at any sample rate. nn/js/render.js is
the same synthesizer in JavaScript.

    synth = DSynth(sample_rate=22050).to(device)
    audio = synth(values)          # values [B, T, params.COUNT] in parameter units -> audio [B, N]

The design (DDSP-style - no recursive filters, so nothing backpropagates through long IIR recursions):

- Voicing is harmonic (additive) synthesis. Each frame's complex harmonic amplitudes are the product of the voicing
  amplitude, the LF pulse's Fourier coefficients for the frame's Rd (nn/lf.py's table, interpolated in Rd), the source
  shelf's response, and the vocal tract's response (formants, nasal pole and zero, higher poles) - each the analytic
  frequency response of lib/klatt.js's filter at k * F0 - with a taper near the Nyquist frequency. The amplitudes are
  interpolated linearly between frames per sample, and summed against the harmonics' phases, so the result is a pulse
  train filtered by the tract, with the tract's phase.
- The glottal phase, cycle by cycle, is computed in float64 on the CPU (it's not differentiated - F0's gradient comes
  through the harmonic amplitudes, which sample the envelope at k * F0): F0 linearly interpolated per sample, with
  jitter (each cycle's period perturbed by a Gaussian draw times the jitter) and shimmer (each cycle's amplitude, the
  draws interpolated across the cycle).
- Noise (aspiration and frication) is white noise scaled per sample by its level (ah, af interpolated - so onsets
  and bursts are as sharp as the frames, as in lib/klatt.js), the aspiration also modulated by the glottal cycle
  (NOISE_MODULATION), then filtered in the frequency domain - a short-time Fourier transform (Hann windows of
  noise_fft_size() samples, 16-23 ms, hop a quarter of that), each frame's spectrum multiplied by the filters'
  magnitude responses (the spectral shapes) at the frame's interpolated parameters, and overlap-added.

The noise is torch.randn by default (training) or the counter-based generator of nn/rng.py (noise='counter'), which
nn/js/render.js also uses, so the two synthesizers can be compared sample by sample.
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
# Filter frequency responses - lib/klatt.js's digital filters designed at params.REF_RATE, evaluated at frequencies in
# Hz (w = 2 pi f / REF_RATE). Arguments broadcast; responses are (real, imaginary) pairs or log magnitudes and phases.
#

def _omega(frequency):
    return 2 * math.pi * frequency / params.REF_RATE


# A resonator's (unity gain at DC) log magnitude and phase at frequencies - the antiresonator's are their negatives
def resonator_log(frequency, center, bandwidth):
    radius = torch.exp(-math.pi * bandwidth / params.REF_RATE)
    coef_c = -radius * radius
    coef_b = 2 * radius * torch.cos(_omega(center))
    coef_a = 1 - coef_b - coef_c
    omega = _omega(frequency)
    den_re = 1 - coef_b * torch.cos(omega) - coef_c * torch.cos(2 * omega)
    den_im = coef_b * torch.sin(omega) + coef_c * torch.sin(2 * omega)
    log_magnitude = torch.log(coef_a) - 0.5 * torch.log(den_re * den_re + den_im * den_im)
    phase = -torch.atan2(den_im, den_re)
    return log_magnitude, phase


# The vocal tract's log magnitude and phase at frequencies [..., M] for parameters (a dict of [..., 1] tensors)
def tract_log(frequency, values):
    log_magnitude = torch.zeros_like(frequency)
    phase = torch.zeros_like(frequency)
    poles = [(values[f'f{kx}'], values[f'b{kx}']) for kx in range(1, 7)]
    poles.append((values['fnp'], values['bnp']))
    for kx in range(1, params.HIGHER_POLES + 1):
        center = values['f6'] * (1 + kx * params.HIGHER_POLE_SPACING)
        bandwidth = params.HIGHER_POLE_BANDWIDTH + params.HIGHER_POLE_BANDWIDTH_STEP * (kx - 1)
        poles.append((center, torch.full_like(center, bandwidth)))
    for center, bandwidth in poles:
        pole_magnitude, pole_phase = resonator_log(frequency, center, bandwidth)
        log_magnitude = log_magnitude + pole_magnitude
        phase = phase + pole_phase
    zero_magnitude, zero_phase = resonator_log(frequency, values['fnz'], values['bnz'])
    return log_magnitude - zero_magnitude, phase - zero_phase


# A biquad's complex response (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2) at frequencies
def _biquad(frequency, b0, b1, b2, a1, a2):
    omega = _omega(frequency)
    cos1, sin1, cos2, sin2 = torch.cos(omega), torch.sin(omega), torch.cos(2 * omega), torch.sin(2 * omega)
    num_re = b0 + b1 * cos1 + b2 * cos2
    num_im = -(b1 * sin1 + b2 * sin2)
    den_re = 1 + a1 * cos1 + a2 * cos2
    den_im = -(a1 * sin1 + a2 * sin2)
    den = den_re * den_re + den_im * den_im
    return (num_re * den_re + num_im * den_im) / den, (num_im * den_re - num_re * den_im) / den


# The voicing's high shelf (lib/klatt.js HighShelf at SHELF_HZ) - log magnitude and phase
def shelf_log(frequency, gain_db):
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
    real, imag = _biquad(frequency, b0, b1, b2, a1, a2)
    return 0.5 * torch.log(real * real + imag * imag + 1e-30), torch.atan2(imag, real)


# The frication bandpass (lib/klatt.js Bandpass, 0 dB peak) - complex response
def _bandpass(frequency, center, bandwidth):
    center_clamped = torch.clamp(center, max=0.45 * params.REF_RATE)
    w0 = _omega(center_clamped)
    alpha = torch.sin(w0) / (2 * torch.clamp(center / bandwidth, min=0.3))
    a0 = 1 + alpha
    zero = torch.zeros_like(alpha)
    return _biquad(frequency, alpha / a0, zero, -alpha / a0, -2 * torch.cos(w0) / a0, (1 - alpha) / a0)


# The frication high-pass (lib/klatt.js HighPass, Butterworth) - squared magnitude
def _highpass_power(frequency, cutoff):
    w0 = _omega(torch.clamp(cutoff, max=0.45 * params.REF_RATE))
    alpha = torch.sin(w0) / math.sqrt(2)
    cos = torch.cos(w0)
    a0 = 1 + alpha
    real, imag = _biquad(frequency, (1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0)
    return real * real + imag * imag


# The frication filter's magnitude (relative to af) - |ab + ap1 BP1^2 + ap2 BP2^2| * |HP|^2
def frication_magnitude(frequency, values):
    bp1_re, bp1_im = _bandpass(frequency, values['fp1'], values['bp1'])
    bp2_re, bp2_im = _bandpass(frequency, values['fp2'], values['bp2'])
    real = values['ab'] + values['ap1'] * (bp1_re * bp1_re - bp1_im * bp1_im) + values['ap2'] * (bp2_re * bp2_re - bp2_im * bp2_im)
    imag = values['ap1'] * (2 * bp1_re * bp1_im) + values['ap2'] * (2 * bp2_re * bp2_im)
    power = _highpass_power(frequency, values['fhp'])
    power = power * power
    return torch.sqrt(real * real + imag * imag + 1e-30) * torch.sqrt(power + 1e-30)


# The aspiration noise's lowpass (lib/klatt.js's one-pole 0.5 / (1 - 0.5 z^-1) at 44.1 kHz) - magnitude
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


# Band gains (dB) [..., bands] interpolated linearly over log frequency at a fixed frequency grid [F] - natural-log
# gains [..., F]
def band_log_grid(frequency, gains, low_hz, high_hz):
    bands = gains.shape[-1]
    step = math.log(high_hz / low_hz) / (bands - 1)
    position = torch.clamp(torch.log(torch.clamp(frequency, min=1.0) / low_hz) / step, 0, bands - 1)
    low = torch.clamp(torch.floor(position).long(), max=bands - 2)
    weight = position - low
    db = gains[..., low] * (1 - weight) + gains[..., low + 1] * weight
    return db * (math.log(10) / 20)


# Split parameter values [..., COUNT] into a dict of [..., 1] tensors, levels converted to linear amplitudes
def split_values(values):
    out = {}
    for ix, name in enumerate(params.NAMES):
        value = values[..., ix:ix + 1]
        out[name] = 10 ** (value / 20) if name in params.DB_NAMES else value
    return out


# Interpolate frames [B, T, C] at fractional frame positions [M] (numpy) - [B, M, C]
def interpolate_frames(frames, positions):
    count = frames.shape[1]
    index = numpy.clip(numpy.floor(positions).astype(numpy.int64), 0, max(0, count - 2))
    weight = torch.tensor(numpy.clip(positions - index, 0, 1), dtype=frames.dtype, device=frames.device)[None, :, None]
    index_t = torch.tensor(index, device=frames.device)
    index_next = torch.clamp(index_t + 1, max=count - 1)
    return frames[:, index_t] * (1 - weight) + frames[:, index_next] * weight


# Linear (per-sample interpolable) parameter values - levels as linear amplitudes, others as they are
def linear_values(values):
    columns = []
    for ix, name in enumerate(params.NAMES):
        value = values[..., ix]
        columns.append(10 ** (value / 20) if name in params.DB_NAMES else value)
    return torch.stack(columns, dim=-1)


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

    # The per-sample glottal cycle plan (numpy, float64, not differentiated) for one item's frames [T, COUNT]
    def _cycle_plan(self, frames, seed):
        sample_rate = self.sample_rate
        count = sample_count(len(frames), sample_rate)
        positions = numpy.arange(count) * params.FRAME_RATE / sample_rate
        index = numpy.clip(numpy.floor(positions).astype(numpy.int64), 0, len(frames) - 2)
        weight = positions - index

        def per_sample(name):
            column = frames[:, params.INDEX[name]]
            return column[index] * (1 - weight) + column[index + 1] * weight

        f0 = numpy.maximum(params.F0_MIN, per_sample('f0'))
        jitter = per_sample('jitter')
        rd = per_sample('rd')

        # The unperturbed glottal phase (cycles), the jitter of each of its cycles, and the perturbed phase
        phase0 = numpy.concatenate([[0.0], numpy.cumsum(f0[:-1] / sample_rate)]) if count else numpy.zeros(0)
        cycle0 = numpy.floor(phase0).astype(numpy.int64)
        cycles = int(cycle0[-1]) + 2 if count else 1
        jitter_draws = rng.gauss(rng.stream_seed(seed, rng.STREAM_JITTER), 0, cycles)
        frequency = f0 / numpy.maximum(0.5, 1 + jitter * jitter_draws[cycle0]) if count else f0
        phase = numpy.concatenate([[0.0], numpy.cumsum(frequency[:-1] / sample_rate)]) if count else numpy.zeros(0)
        cycle = numpy.floor(phase).astype(numpy.int64)
        fraction = phase - cycle

        # The shimmer draws, interpolated across each cycle, and the glottis's open phase
        shimmer_draws = rng.gauss(rng.stream_seed(seed, rng.STREAM_SHIMMER), 0, int(cycle[-1]) + 3 if count else 2)
        shimmer = shimmer_draws[cycle] * (1 - fraction) + shimmer_draws[cycle + 1] * fraction if count else numpy.zeros(0)
        is_open = fraction < lf.open_fraction(rd)
        return {'fraction': fraction, 'shimmer': shimmer, 'open': is_open, 'index': index, 'weight': weight}

    # The noise sequences (unit variance at the reference rate's level) for one item
    def _noise(self, count, seed, device):
        scale = math.sqrt(self.sample_rate / params.REF_RATE)
        if self.noise == 'counter':
            aspiration = rng.gauss(rng.stream_seed(seed, rng.STREAM_ASPIRATION), 0, count)
            frication = rng.gauss(rng.stream_seed(seed, rng.STREAM_FRICATION), 0, count)
            return (torch.tensor(aspiration * scale, dtype=torch.float32, device=device),
                    torch.tensor(frication * scale, dtype=torch.float32, device=device))
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(2, count, generator=generator) * scale
        return noise[0].to(device), noise[1].to(device)

    # The residual noise sequence for one item
    def _residual_noise(self, count, seed, device):
        scale = math.sqrt(self.sample_rate / params.REF_RATE)
        if self.noise == 'counter':
            residual = rng.gauss(rng.stream_seed(seed, rng.STREAM_RESIDUAL), 0, count)
            return torch.tensor(residual * scale, dtype=torch.float32, device=device)
        generator = torch.Generator(device='cpu').manual_seed(int(seed) + 7919)
        return (torch.randn(count, generator=generator) * scale).to(device)

    # The harmonic count for a batch - enough for the lowest voiced F0
    def _harmonic_count(self, frames):
        voiced = frames[..., params.INDEX['av']] > params.FLOOR_DB + 30
        f0 = numpy.maximum(params.F0_MIN, frames[..., params.INDEX['f0']])
        lowest = f0[voiced].min() if voiced.any() else 1000.0
        return int(min(params.LF_HARMONICS - 1, math.ceil(params.HARMONIC_TAPER_END * self.sample_rate / lowest)))

    # The frames' complex harmonic amplitudes [B, T, K] (real, imaginary) and the harmonic numbers
    def harmonic_amplitudes(self, values, harmonics, eq=None):
        split = split_values(values)
        k = torch.arange(1, harmonics + 1, device=values.device, dtype=values.dtype)
        f0 = torch.clamp(split['f0'], min=params.F0_MIN)
        frequency = f0 * k

        # The LF pulse's coefficients, interpolated in Rd
        grid = (torch.clamp(split['rd'][..., 0], params.LF_RD_MIN, params.LF_RD_MAX) - params.LF_RD_MIN) / params.LF_RD_STEP
        low = torch.clamp(torch.floor(grid).long(), 0, self.lf_re.shape[0] - 2)
        weight = (grid - low)[..., None]
        table_re = self.lf_re[:, 1:harmonics + 1]
        table_im = self.lf_im[:, 1:harmonics + 1]
        source_re = table_re[low] * (1 - weight) + table_re[low + 1] * weight
        source_im = table_im[low] * (1 - weight) + table_im[low + 1] * weight

        # The source shelf and the tract, as log magnitude and phase
        tract_magnitude, tract_phase = tract_log(frequency, split)
        shelf_magnitude, shelf_phase = shelf_log(frequency, split['shelf'])
        if eq is not None:
            tract_magnitude = tract_magnitude + eq_log(frequency, eq)
        magnitude = 2 * split['av'] * torch.exp(tract_magnitude + shelf_magnitude) * harmonic_taper(frequency, self.sample_rate)
        phase = tract_phase + shelf_phase
        filter_re = magnitude * torch.cos(phase)
        filter_im = magnitude * torch.sin(phase)
        return source_re * filter_re - source_im * filter_im, source_re * filter_im + source_im * filter_re

    # The harmonic (voicing) signal [B, N]
    def _voicing(self, values, plans, harmonics, count, eq=None):
        device = values.device
        batch, frame_count, _ = values.shape
        amp_re, amp_im = self.harmonic_amplitudes(values, harmonics, eq)

        # Group the samples by frame interval - a grid [T - 1, H] of sample indices (H the most samples in an interval)
        index = plans[0]['index']
        starts = numpy.searchsorted(index, numpy.arange(frame_count - 1))
        ends = numpy.searchsorted(index, numpy.arange(frame_count - 1), side='right')
        width = int((ends - starts).max())
        grid = starts[:, None] + numpy.arange(width)[None, :]
        valid = grid < ends[:, None]
        grid_clamped = numpy.minimum(grid, count - 1)
        weight = torch.tensor(plans[0]['weight'][grid_clamped], dtype=values.dtype, device=device)[None, :, :, None]
        fraction = torch.tensor(numpy.stack([plan['fraction'][grid_clamped] for plan in plans]), dtype=values.dtype, device=device)

        # Sum the harmonics: Re(sum_k A_k exp(2 pi i k phase)), the amplitudes interpolated between frames
        k = torch.arange(1, harmonics + 1, device=device, dtype=values.dtype)
        angle = 2 * math.pi * torch.remainder(fraction[..., None] * k, 1.0)
        cos, sin = torch.cos(angle), torch.sin(angle)
        first = torch.matmul(cos, amp_re[:, :-1, :, None]) - torch.matmul(sin, amp_im[:, :-1, :, None])
        second = torch.matmul(cos, amp_re[:, 1:, :, None]) - torch.matmul(sin, amp_im[:, 1:, :, None])
        signal = (first * (1 - weight) + second * weight)[..., 0]
        flat = torch.tensor(grid[valid], device=device)
        out = torch.zeros(batch, count, dtype=values.dtype, device=device)
        return out.index_copy(1, flat, signal[:, torch.tensor(valid, device=device)])

    # The noise (aspiration and frication) signal [B, N]
    def _noise_signal(self, values, linear, plans, seeds, count, eq=None, residual=None):
        device = values.device
        batch = values.shape[0]
        fft_size, hop = self.noise_fft, self.noise_hop

        # The noise levels per sample (so onsets are as sharp as the frames), and the aspiration's glottal modulation -
        # full while open, reduced by the voicing while closed
        index = torch.tensor(plans[0]['index'], device=device)
        weight = torch.tensor(plans[0]['weight'], dtype=values.dtype, device=device)
        def per_sample(name):
            return linear[:, index, params.INDEX[name]] * (1 - weight) + linear[:, index + 1, params.INDEX[name]] * weight
        av = per_sample('av')
        is_open = torch.tensor(numpy.stack([plan['open'] for plan in plans]), device=device)
        modulation = torch.where(is_open, torch.ones_like(av), 1 - params.NOISE_MODULATION * torch.clamp(2 * av, max=1))
        noises = [self._noise(count, seed, device) for seed in seeds]
        aspiration = torch.stack([noise[0] for noise in noises]) * per_sample('ah') * modulation
        frication = torch.stack([noise[1] for noise in noises]) * per_sample('af')

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
        at_frames = interpolate_frames(linear, positions)
        split = {name: at_frames[..., ix:ix + 1] for ix, name in enumerate(params.NAMES)}
        frequency = torch.arange(fft_size // 2 + 1, device=device, dtype=values.dtype) * self.sample_rate / fft_size
        tract_magnitude, _ = tract_log(frequency, split)
        filter_aspiration = aspiration_magnitude(frequency) * torch.exp(tract_magnitude)
        filter_frication = frication_magnitude(frequency, split)
        if eq is not None:
            channel = torch.exp(eq_log(frequency.expand(batch, -1), eq))[:, None, :]
            filter_aspiration = filter_aspiration * channel
            filter_frication = filter_frication * channel
        filtered = spectrum_aspiration * filter_aspiration + spectrum_frication * filter_frication
        if residual is not None:
            # The residual noise - its band levels (dB) interpolated to the noise frames
            residual_db = interpolate_frames(residual, positions)
            filter_residual = torch.exp(band_log_grid(frequency, residual_db, params.RESIDUAL_LOW_HZ, params.RESIDUAL_HIGH_HZ))
            if eq is not None:
                filter_residual = filter_residual * channel
            residual_noise = torch.stack([self._residual_noise(count, seed, device) for seed in seeds])
            filtered = filtered + torch.fft.rfft(framed(residual_noise)) * filter_residual
        frames_out = torch.fft.irfft(filtered, n=fft_size) * self.window

        # Overlap-add (hop is a quarter frame) and normalize by the windows' squared sum
        quarters = frames_out.reshape(batch, frames_count, 4, hop)
        total = torch.zeros(batch, frames_count + 3, hop, dtype=values.dtype, device=device)
        for quarter in range(4):
            total = total + torch.nn.functional.pad(quarters[:, :, quarter], (0, 0, quarter, 3 - quarter))
        window_sum = torch.zeros(frames_count + 3, hop, dtype=values.dtype, device=device)
        window_squared = (self.window * self.window).reshape(4, hop)
        for quarter in range(4):
            window_sum[quarter:quarter + frames_count] += window_squared[quarter]
        signal = (total / torch.clamp(window_sum, min=1e-3)).reshape(batch, -1)
        return signal[:, fft_size // 2:fft_size // 2 + count]

    def forward(self, values, seed=1, return_parts=False, eq=None, residual=None):
        """Render parameter values [B, T, COUNT] (or [T, COUNT]) to audio [B, N] (or [N]); seed is an int or a list; eq is
        the optional channel equalizer's gains (dB) [B, EQ_BAND_COUNT] (or [EQ_BAND_COUNT]); residual is the optional
        (experimental, PyTorch only) residual noise's band levels (dB) [B, T, RESIDUAL_BAND_COUNT]"""
        squeeze = values.dim() == 2
        if squeeze:
            values = values[None]
            if eq is not None and eq.dim() == 1:
                eq = eq[None]
            if residual is not None and residual.dim() == 2:
                residual = residual[None]
        batch, frame_count, _ = values.shape
        seeds = list(seed) if isinstance(seed, (list, tuple)) else [int(seed) + ix for ix in range(batch)]
        count = sample_count(frame_count, self.sample_rate)
        if count == 0:
            empty = values.new_zeros(batch, 0)
            return empty[0] if squeeze else empty
        frames_cpu = values.detach().cpu().double().numpy()  # (MPS: .to('cpu', float64) in one step gives zeros)
        plans = [self._cycle_plan(frames_cpu[ix], seeds[ix]) for ix in range(batch)]
        harmonics = self._harmonic_count(frames_cpu)
        linear = linear_values(values)

        # The voicing, with shimmer, and the noise
        voicing = self._voicing(values, plans, harmonics, count, eq)
        index = torch.tensor(plans[0]['index'], device=values.device)
        weight = torch.tensor(plans[0]['weight'], dtype=values.dtype, device=values.device)
        shimmer_amount = linear[:, index, params.INDEX['shimmer']] * (1 - weight) + linear[:, index + 1, params.INDEX['shimmer']] * weight
        draws = torch.tensor(numpy.stack([plan['shimmer'] for plan in plans]), dtype=values.dtype, device=values.device)
        voicing = voicing * torch.relu(1 + shimmer_amount * draws)
        noise = self._noise_signal(values, linear, plans, seeds, count, eq, residual)
        audio = voicing + noise
        if return_parts:
            parts = {'audio': audio, 'voicing': voicing, 'noise': noise}
            return {key: value[0] for key, value in parts.items()} if squeeze else parts
        return audio[0] if squeeze else audio
