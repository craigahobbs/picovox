# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Spectral losses and distances - the multi-resolution STFT loss that trains through nn/dsynth.py, log-mel spectra, and
the log-spectral distance used to compare renderings and recordings.
"""

import functools
import math

import numpy
import torch


# A Hann window, cached per size and device
@functools.lru_cache(maxsize=32)
def _hann(size, device):
    return torch.hann_window(size, device=device)


# The STFT magnitude [B, frames, bins] of audio [B, N]
def stft_magnitude(audio, fft_size, hop, eps=1e-7):
    spectrum = torch.stft(audio, fft_size, hop_length=hop, window=_hann(fft_size, str(audio.device)), center=True,
                          return_complex=True)
    return torch.sqrt(spectrum.real ** 2 + spectrum.imag ** 2 + eps).transpose(1, 2)


# The multi-resolution STFT loss - for each FFT size, the spectral convergence plus the mean absolute log-magnitude
# difference (Yamamoto et al., 2020; DDSP), averaged over the resolutions. Optionally weighted per frame ([B, T] at
# any rate - resampled to each resolution's frames).
def multi_resolution_stft_loss(predicted, target, fft_sizes=(2048, 1024, 512, 256, 128), overlap=0.75, log_weight=1.0,
                               convergence_weight=1.0):
    total = 0
    for fft_size in fft_sizes:
        hop = int(fft_size * (1 - overlap))
        magnitude_predicted = stft_magnitude(predicted, fft_size, hop)
        magnitude_target = stft_magnitude(target, fft_size, hop)
        convergence = torch.linalg.norm(magnitude_target - magnitude_predicted, dim=(1, 2)) / \
            (torch.linalg.norm(magnitude_target, dim=(1, 2)) + 1e-7)
        log_distance = (torch.log(magnitude_target) - torch.log(magnitude_predicted)).abs().mean(dim=(1, 2))
        total = total + convergence_weight * convergence.mean() + log_weight * log_distance.mean()
    return total / len(fft_sizes)


# A mel filterbank [mels, bins] (Slaney-style triangles on the HTK mel scale), as a numpy array
@functools.lru_cache(maxsize=16)
def mel_filterbank(sample_rate, fft_size, mels=80, low=40.0, high=None):
    high = high or sample_rate / 2
    def to_mel(frequency):
        return 2595 * numpy.log10(1 + frequency / 700)
    def from_mel(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    edges = from_mel(numpy.linspace(to_mel(low), to_mel(high), mels + 2))
    bins = numpy.arange(fft_size // 2 + 1) * sample_rate / fft_size
    bank = numpy.zeros((mels, len(bins)))
    for ix in range(mels):
        left, center, right = edges[ix], edges[ix + 1], edges[ix + 2]
        rising = (bins - left) / (center - left)
        falling = (right - bins) / (right - center)
        bank[ix] = numpy.maximum(0, numpy.minimum(rising, falling)) * 2 / (right - left)
    return bank


# Log-mel power spectrogram (dB) [frames, mels] of a numpy signal
def log_mel(signal, sample_rate, fft_size=None, hop=None, mels=80, high=None):
    fft_size = fft_size or (1 << math.ceil(math.log2(sample_rate * 0.025)))
    hop = hop or int(round(sample_rate * 0.005))
    audio = torch.tensor(numpy.asarray(signal, dtype=numpy.float64))[None]
    power = stft_magnitude(audio, fft_size, hop, eps=0)[0].numpy() ** 2
    bank = mel_filterbank(sample_rate, fft_size, mels, high=high)
    return 10 * numpy.log10(power @ bank.T + 1e-10)


# The log-spectral distance (dB) between two signals' log-mel spectrograms - the RMS over mel bands of the dB
# difference, averaged over the frames where the reference is within floor_db of its loudest frame. The signals are
# first level-matched (by RMS, or not with match=False); the shorter length is compared.
def log_spectral_distance(reference, other, sample_rate, floor_db=50.0, match=True, high=None):
    length = min(len(reference), len(other))
    reference = numpy.asarray(reference[:length], dtype=numpy.float64)
    other = numpy.asarray(other[:length], dtype=numpy.float64)
    if match:
        other = other * numpy.sqrt((reference ** 2).mean() / max((other ** 2).mean(), 1e-20))
    mel_reference = log_mel(reference, sample_rate, high=high)
    mel_other = log_mel(other, sample_rate, high=high)
    level = 10 * numpy.log10((10 ** (mel_reference / 10)).sum(axis=1))
    active = level > level.max() - floor_db
    difference = numpy.maximum(mel_reference, mel_reference.max() - 80) - numpy.maximum(mel_other, mel_reference.max() - 80)
    return float(numpy.sqrt((difference[active] ** 2).mean(axis=1)).mean())


# The long-term average spectra difference (dB per mel band, other minus reference) of level-matched signals
def ltas_difference(reference, other, sample_rate, high=None):
    length = min(len(reference), len(other))
    reference = numpy.asarray(reference[:length], dtype=numpy.float64)
    other = numpy.asarray(other[:length], dtype=numpy.float64)
    other = other * numpy.sqrt((reference ** 2).mean() / max((other ** 2).mean(), 1e-20))
    mel_reference = 10 ** (log_mel(reference, sample_rate, high=high) / 10)
    mel_other = 10 ** (log_mel(other, sample_rate, high=high) / 10)
    return 10 * numpy.log10(mel_other.mean(axis=0) / mel_reference.mean(axis=0))
