# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
WAV files - 16-bit PCM read and written with the standard library's wave module, exactly as libsndfile (soundfile)
reads and writes them: a sample read is its integer divided by 32768; a sample written is round(x * 2^31) (half to
even, clipped to 32 bits) shifted right 16 bits - every corpus's audio (CMU ARCTIC, LibriTTS-R, CREMA-D) and the
package's own (lib/wav.js) is 16-bit PCM.
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


# Write samples ([frames], or [frames, channels]; -1 to 1 - beyond are clipped) as a 16-bit PCM WAV file
def write_wav(path, samples, rate):
    samples = numpy.asarray(samples, dtype=numpy.float64)
    scaled = numpy.clip(numpy.rint(samples * 2 ** 31), -2 ** 31, 2 ** 31 - 1)
    with wave.Wave_write(path) as file:
        file.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes((scaled.astype(numpy.int64) >> 16).astype('<i2').tobytes())
