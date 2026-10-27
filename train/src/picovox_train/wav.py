# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
WAV files - 16-bit PCM read with the standard library's wave module, exactly as libsndfile (soundfile) reads them: a
sample read is its integer divided by 32768 - every corpus's audio (CMU ARCTIC, LibriTTS-R, CREMA-D) and the package's
own (lib/wav.js) is 16-bit PCM.
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
