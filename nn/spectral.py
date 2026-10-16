# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Spectral analysis - the STFT and its magnitude (nn/losses.py's multi-resolution STFT and log-mel loss, and
nn/discriminator.py's critic input), the mel filterbank, log-mel spectra, and the log-spectral distance used to compare
renderings (nn/parity.py).
"""

import functools
import math

import numpy
import torch


# A Hann window, cached per size and device
@functools.lru_cache(maxsize=32)
def _hann(size, device):
    return torch.hann_window(size, device=device)


# The complex STFT [B, bins, frames] of audio [B, N] (Hann window, centered)
def stft(audio, fft_size, hop):
    return torch.stft(audio, fft_size, hop_length=hop, window=_hann(fft_size, str(audio.device)), center=True,
                      return_complex=True)


# The magnitude [B, frames, bins] of a complex STFT [B, bins, frames]
def spectrum_magnitude(spectrum, eps=1e-7):
    return torch.sqrt(spectrum.real ** 2 + spectrum.imag ** 2 + eps).transpose(1, 2)


# The STFT magnitude [B, frames, bins] of audio [B, N]
def stft_magnitude(audio, fft_size, hop, eps=1e-7):
    return spectrum_magnitude(stft(audio, fft_size, hop), eps)


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

