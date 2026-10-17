# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Spectral analysis - the STFT and its magnitude (losses.py's multi-resolution STFT and log-mel loss, and
discriminator.py's critic input), and the mel filterbank.
"""

import functools

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
