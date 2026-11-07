# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
WAV files - 16-bit PCM read with the standard library's wave module, exactly as libsndfile (soundfile) reads them: a
sample read is its integer divided by 32768 - the training speech (the TTS engine's) and the package's own audio
(lib/wav.js) are 16-bit PCM - and resampling (resample - a windowed-sinc filter, torch's convolution, differentiable;
read_mono's long filter for the corpora, the perceptual loss's short one, perceptual.py).
"""

import math
import wave

import numpy
import torch


# The resampling filter - a windowed sinc (Hann) of zeros zero crossings each side, its cutoff RESAMPLE_ROLLOFF of the
# lower rate's Nyquist frequency: RESAMPLE_ZEROS, the perceptual loss's (torchaudio's sinc_interp_hann defaults - resample
# reproduces its arithmetic bit for bit), and READ_ZEROS, the corpora's (a long filter - the quality of an offline
# resampler, soxr's, which it replaced)
RESAMPLE_ZEROS = 6
READ_ZEROS = 64
RESAMPLE_ROLLOFF = 0.99

# The resampling kernels, by (rates, zeros, device, dtype)
_KERNELS = {}


# The resampling filter of a rate change - (its kernels [new, 1, width], the kernels' half width, the rates reduced by
# their gcd): each of the new output samples of a period is a kernel over the input, the convolution's stride the
# reduced input rate
def resample_kernel(orig_rate, new_rate, device, dtype, zeros=RESAMPLE_ZEROS):
    key = (orig_rate, new_rate, zeros, str(device), dtype)
    if key not in _KERNELS:
        gcd = math.gcd(orig_rate, new_rate)
        orig, new = orig_rate // gcd, new_rate // gcd
        base = min(orig, new) * RESAMPLE_ROLLOFF
        width = math.ceil(zeros * orig / base)
        index = torch.arange(-width, width + orig, dtype=dtype, device=device)[None, None] / orig
        times = torch.arange(0, -new, -1, dtype=dtype, device=device)[:, None, None] / new + index
        times *= base
        times = times.clamp_(-zeros, zeros)
        window = torch.cos(times * math.pi / zeros / 2) ** 2
        times *= math.pi
        kernels = torch.where(times == 0, torch.tensor(1.0).to(times), times.sin() / times)
        kernels *= window * (base / orig)
        _KERNELS[key] = (kernels, width, orig, new)
    return _KERNELS[key]


# Resample audio [B, N] (a torch tensor) from one rate to another (differentiably)
def resample(audio, orig_rate, new_rate, zeros=RESAMPLE_ZEROS):
    kernels, width, orig, new = resample_kernel(orig_rate, new_rate, audio.device, audio.dtype, zeros)
    padded = torch.nn.functional.pad(audio, (width, width + orig))
    resampled = torch.nn.functional.conv1d(padded[:, None], kernels, stride=orig).transpose(1, 2).reshape(audio.shape[0], -1)
    return resampled[..., :math.ceil(new * audio.shape[1] / orig)]


# Read a 16-bit PCM WAV file - (samples [frames, channels], rate)
def read_wav(path, dtype='float64'):
    with wave.open(path) as file:
        if file.getsampwidth() != 2:
            raise ValueError(f'{path}: {8 * file.getsampwidth()}-bit samples - only 16-bit PCM is read')
        integers = numpy.frombuffer(file.readframes(file.getnframes()), dtype='<i2').reshape(-1, file.getnchannels())
        return integers.astype(dtype) / numpy.array(32768, dtype=dtype), file.getframerate()


# Read a WAV file mixed to mono at a rate (resampled by the long filter if it's another), from start to end seconds
# (default: all of it) - a contiguous array
def read_mono(path, rate, dtype='float64', start=None, end=None):
    samples, file_rate = read_wav(path, dtype)
    if start is not None:
        samples = samples[int(start * file_rate):int(end * file_rate)]
    samples = samples.mean(axis=1)
    if file_rate != rate:
        samples = resample(torch.from_numpy(numpy.ascontiguousarray(samples)[None]), file_rate, rate, READ_ZEROS)[0].numpy()
    return numpy.ascontiguousarray(samples, dtype=dtype)
