# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
WAV files - 16-bit PCM read with the standard library's wave module, exactly as libsndfile (soundfile) reads them: a
sample read is its integer divided by 32768 - the training speech (the TTS engine's) and the package's own audio
(lib/wav.js) are 16-bit PCM.
"""

import wave

import numpy


# Read a 16-bit PCM WAV file - (samples [frames, channels], rate)
def read_wav(path, dtype='float64'):
    with wave.open(path) as file:
        if file.getsampwidth() != 2:
            raise ValueError(f'{path}: {8 * file.getsampwidth()}-bit samples - only 16-bit PCM is read')
        integers = numpy.frombuffer(file.readframes(file.getnframes()), dtype='<i2').reshape(-1, file.getnchannels())
        return integers.astype(dtype) / numpy.array(32768, dtype=dtype), file.getframerate()


# Read a WAV file mixed to mono at a rate (resampled by soxr at a quality if it's another), from start to end seconds
# (default: all of it) - a contiguous array
def read_mono(path, rate, dtype='float64', quality='VHQ', start=None, end=None):
    import soxr  # pylint: disable=import-outside-toplevel
    samples, file_rate = read_wav(path, dtype)
    if start is not None:
        samples = samples[int(start * file_rate):int(end * file_rate)]
    samples = samples.mean(axis=1)
    if file_rate != rate:
        samples = soxr.resample(samples, file_rate, rate, quality=quality)
    return numpy.ascontiguousarray(samples, dtype=dtype)
