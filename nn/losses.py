# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The training losses and targets shared by the trainers (nn/trainmulti.py, nn/train.py): the parameter priors, the F0
target track, the multi-resolution STFT and log-mel loss, the duration loss, and the parameter frames made ready to
render (the end frame, silence past each utterance, the recorded F0).
"""

import math

import numpy
import torch

from . import params, spectral
from .model import BOS, FIRST_PHONE, PAD, is_boundary


# Parameter priors per speaker (the analysis-by-synthesis fits' medians, nn/fit.py) - the head's starting point
PRIORS = {
    'slt': {'f0': 185, 'av': -30, 'rd': 1.0, 'shelf': 10, 'ah': -65, 'af': -50, 'f1': 600, 'f2': 1800, 'f3': 3000, 'f4': 4100,
            'f5': 5100, 'f6': 5900, 'b1': 150, 'b2': 180, 'b3': 200, 'b4': 220, 'b5': 300, 'b6': 320, 'fnp': 260, 'bnp': 100,
            'fnz': 290, 'bnz': 100, 'fp1': 2400, 'bp1': 600, 'ap1': -4, 'fp2': 5800, 'bp2': 700, 'ap2': -6, 'ab': -32,
            'fhp': 2000},
    'bdl': {'f0': 122, 'av': -30, 'rd': 1.0, 'shelf': 10, 'ah': -65, 'af': -45, 'f1': 500, 'f2': 1550, 'f3': 2700, 'f4': 3500,
            'f5': 4200, 'f6': 5300, 'b1': 90, 'b2': 110, 'b3': 150, 'b4': 160, 'b5': 200, 'b6': 500, 'fnp': 270, 'bnp': 100,
            'fnz': 270, 'bnz': 100, 'fp1': 2600, 'bp1': 450, 'ap1': -5, 'fp2': 5500, 'bp2': 550, 'ap2': -5, 'ab': -33,
            'fhp': 1800}
}

# Formant guide indexes (aux tracks f1-f3) and the parameter indexes
F_INDEX = [params.INDEX[f'f{number}'] for number in range(1, 7)]


# The 200 Hz F0 target (2F + 1 parameter frames, at t x 5 ms) of an utterance's 100 Hz Praat track (frame k centered at
# (k + 0.5) x 10 ms) - log-interpolated through the unvoiced frames, held at the ends, within the speaker's range
def f0_track(f0, voiced, frame_count, floor, ceiling):
    centers = (numpy.arange(frame_count) + 0.5) * 0.01
    times = numpy.arange(2 * frame_count + 1) * 0.005
    good = voiced & (f0 > 0)
    if good.sum() < 2:
        return numpy.full(len(times), math.sqrt(floor * ceiling), dtype=numpy.float32)
    log_f0 = numpy.interp(times, centers[good], numpy.log(f0[good]))
    return numpy.clip(numpy.exp(log_f0), floor, ceiling).astype(numpy.float32)


# The multi-resolution STFT loss below a frequency (spectral convergence + mean absolute log magnitude) plus a log-mel L1
# (each signal's STFT is taken once per size - the log-mel terms reuse the resolutions' spectra)
def spectral_loss(predicted, target, rate, max_hz, fft_sizes=(2048, 1024, 512, 256, 128), mel_weight=1.0, mel_eps=0.0):
    spectra = {}

    def spectra_of(fft_size):
        if fft_size not in spectra:
            spectra[fft_size] = (spectral.stft(predicted, fft_size, fft_size // 4), spectral.stft(target, fft_size, fft_size // 4))
        return spectra[fft_size]
    total = 0
    for fft_size in fft_sizes:
        bins = int(max_hz * fft_size / rate) + 1
        spectrum_predicted, spectrum_target = spectra_of(fft_size)
        magnitude_predicted = spectral.spectrum_magnitude(spectrum_predicted)[..., :bins]
        magnitude_target = spectral.spectrum_magnitude(spectrum_target)[..., :bins]
        convergence = torch.linalg.norm(magnitude_target - magnitude_predicted, dim=(1, 2)) / \
            (torch.linalg.norm(magnitude_target, dim=(1, 2)) + 1e-7)
        log_distance = (torch.log(magnitude_target) - torch.log(magnitude_predicted)).abs().mean(dim=(1, 2))
        total = total + convergence.mean() + log_distance.mean()
    total = total / len(fft_sizes)
    if mel_weight:
        mel_total = 0
        for fft_size in (1024, 256):
            bank = mel_bank(rate, fft_size, max_hz, predicted.device)
            spectrum_predicted, spectrum_target = spectra_of(fft_size)
            mel_predicted = torch.log(spectral.spectrum_magnitude(spectrum_predicted, eps=mel_eps) ** 2 @ bank.T + 1e-5)
            mel_target = torch.log(spectral.spectrum_magnitude(spectrum_target, eps=mel_eps) ** 2 @ bank.T + 1e-5)
            mel_total = mel_total + (mel_predicted - mel_target).abs().mean()
        total = total + mel_weight * mel_total / 2 * 0.25
    return total


_MEL_BANKS = {}


def mel_bank(rate, fft_size, max_hz, device):
    key = (rate, fft_size, max_hz, str(device))
    if key not in _MEL_BANKS:
        _MEL_BANKS[key] = torch.tensor(spectral.mel_filterbank(rate, fft_size, 64 if fft_size >= 512 else 32, high=max_hz),
                                       dtype=torch.float32, device=device)
    return _MEL_BANKS[key]


# The duration loss - phones: MSE of log(d + 1); boundary tokens: pause BCE and the paused duration's MSE; <bos> and the
# last token (the recording's edge silences) at edge_weight
def duration_loss(outputs, tokens, durations, edge_weight=0.1):
    log_duration, pause_logit, log_paused = outputs.unbind(-1)
    valid = tokens != PAD
    lengths = valid.sum(dim=1)
    target = torch.log1p(durations.float())
    weight = valid.float()
    weight = torch.where(tokens == BOS, weight * edge_weight, weight)
    last = torch.zeros_like(weight).scatter_(1, (lengths - 1)[:, None], 1.0)
    weight = torch.where(last > 0, weight * edge_weight, weight)
    phone = (tokens >= FIRST_PHONE).float() * weight
    boundary = (is_boundary(tokens) | (tokens == BOS)).float() * weight
    paused = (durations > 0).float()
    phone_loss = ((log_duration - target) ** 2 * phone).sum() / phone.sum().clamp(min=1)
    pause_loss = (torch.nn.functional.binary_cross_entropy_with_logits(pause_logit, paused, reduction='none') * boundary).sum() / \
        boundary.sum().clamp(min=1)
    paused_weight = boundary * paused
    paused_loss = ((log_paused - target) ** 2 * paused_weight).sum() / paused_weight.sum().clamp(min=1)
    return phone_loss, pause_loss, paused_loss


# A silent parameter frame (normalized and in parameter units)
def silent_values(device):
    values = torch.tensor([params.DEFAULTS[name] for name in params.NAMES], dtype=torch.float32, device=device)
    return values


# The silent frame and the mask of the levels that silence it (av, ah, af) on a device - made once per device
_SILENCE = {}


def _silence(device):
    key = str(device)
    if key not in _SILENCE:
        silence_mask = torch.zeros(params.COUNT, device=device)
        for name in ('av', 'ah', 'af'):
            silence_mask[params.INDEX[name]] = 1
        _SILENCE[key] = (silent_values(device), silence_mask)
    return _SILENCE[key]


# The model's parameter frames for a batch, ready to render - [B, 2F + 1, COUNT]: the end frame appended, frames past
# each utterance's end silent, and F0 replaced by the recorded F0 (teacher_f0) when given
def render_values(model, unit, batch, teacher_f0=True):
    values = model.parameters_from_unit(unit)
    values = torch.cat([values, values[:, -1:]], dim=1)
    total = values.shape[1]
    # (the batch's 'renderCounts' - 2 x its frame counts, on the device - if it has them: on MPS a blocking
    # host-to-device copy waits for the whole GPU queue)
    counts = batch.get('renderCounts')
    if counts is None:
        counts = torch.tensor([2 * count for count in batch['frameCounts']], device=values.device)
    past = (torch.arange(total, device=values.device)[None] > counts[:, None]).float()[..., None]
    silence, silence_mask = _silence(values.device)
    values = values * (1 - past * silence_mask) + silence * past * silence_mask
    if teacher_f0:
        f0 = batch['f0'][:, :total]
        values = torch.cat([values[..., :params.INDEX['f0']], f0[..., None], values[..., params.INDEX['f0'] + 1:]], dim=-1)
    return values
