# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The unit tests' shared helpers - tiny synthetic corpora (data.py's format), tiny model configurations, output capture,
and packaged model modules' contents
"""

import base64
import contextlib
import io
import json
import os
import re
import time
import wave
import zlib

import numpy

from picovox_train import data, frontend


# Write samples ([frames], or [frames, channels]; -1 to 1 - beyond are clipped) as a 16-bit PCM WAV file, exactly as
# libsndfile (soundfile) writes them: round(x * 2^31) (half to even, clipped to 32 bits) shifted right 16 bits
def write_wav(path, samples, rate):
    samples = numpy.asarray(samples, dtype=numpy.float64)
    scaled = numpy.clip(numpy.rint(samples * 2 ** 31), -2 ** 31, 2 ** 31 - 1)
    with wave.Wave_write(path) as file:
        file.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes((scaled.astype(numpy.int64) >> 16).astype('<i2').tobytes())


# A tiny MultiModel configuration (model.py) - a few thousand parameters, fast on the CPU
def tiny_config(**overrides):
    return {
        'vocab': len(frontend.vocabulary()), 'dim': 16, 'encoderLayers': 1, 'heads': 2, 'encoderKernel': 3, 'ffnMultiple': 2,
        'durationKernel': 3, 'durationDropout': 0.2, 'decoderDim': 16, 'decoderLayers': 1, 'decoderKernel': 3, 'subframes': 2,
        'speakers': 4, 'styleDim': 4, 'jitter': 0.005, 'shimmer': 0.02, 'positionFeatures': 5, 'kind': 'multi',
        'speakerDim': 4, 'styles': 5, **overrides
    }


# The texts of the synthetic utterances
TEXTS = ['Call me Ishmael.', 'It was the best of times.', 'Is it cold?', 'Wait, look here!', 'The cat sat on the mat.',
         'We went home.', 'Who is there?', 'A dark night.']


# Write a tiny corpus directory in data.py's format (index.json, tokens.npy, durations.npy, seconds.npy, audio.npy,
# frames.npy, alignment.jsonl) - speakers {key: {sex, index[, pitchRanges]}}, count utterances per speaker (the splits
# cycle train, train, val, test; the last one excluded), synthetic voiced audio. emotions gives the utterances emotion and
# intensity fields (cremad.py's), cycling. Returns the index.
def write_corpus(directory, speakers=None, count=4, seed=0, emotions=None, frames_per_token=3):
    rng = numpy.random.default_rng(seed)
    speakers = speakers or {'arctic-slt': {'sex': 'female', 'index': 0}, 'arctic-bdl': {'sex': 'male', 'index': 1}}
    os.makedirs(directory, exist_ok=True)
    splits = ['train', 'train', 'val', 'test']
    tokens_all, durations_all, seconds_all, audio_all, frames_all = [], [], [], [], []
    entries = []
    token_offset = frame_offset = 0
    for speaker_number, (key, info) in enumerate(speakers.items()):
        for number in range(count):
            text = TEXTS[(speaker_number + number) % len(TEXTS)]
            tokenized = frontend.tokenize(text)
            ids = numpy.asarray(tokenized['ids'], dtype=numpy.int16)
            durations = numpy.asarray([frames_per_token + int(rng.integers(0, 3)) if token >= 9 else int(rng.integers(0, 2))
                                       for token in ids], dtype=numpy.int16)
            frame_count = int(durations.sum())
            f0 = 120.0 if info.get('sex') == 'male' else 200.0
            times = numpy.arange(frame_count * data.HOP) / data.SAMPLE_RATE
            audio = 0.3 * numpy.sin(2 * numpy.pi * f0 * times) + 0.01 * rng.standard_normal(len(times))
            tracks = numpy.zeros((frame_count, len(data.CHANNELS)), dtype=numpy.float32)
            voiced = (numpy.arange(frame_count) % 7) != 0
            tracks[:, data.CHANNELS.index('f0')] = numpy.where(voiced, f0, 0.0)
            tracks[:, data.CHANNELS.index('voiced')] = voiced
            tracks[:, data.CHANNELS.index('strength')] = numpy.where(voiced, 0.8, 0.1)
            for formant, value in zip(('f1', 'f2', 'f3', 'f4'), (500.0, 1500.0, 2500.0, 3500.0)):
                tracks[:, data.CHANNELS.index(formant)] = value
            entry = {
                'id': f'{key}/{number:04d}', 'speaker': key, 'sentence': f'{number:04d}', 'split': splits[number % 4],
                'text': text, 'include': number != count - 1 or count == 1, 'reason': '' if number != count - 1 else 'excluded',
                'tokenOffset': token_offset, 'tokenCount': len(ids), 'frameOffset': frame_offset, 'frameCount': frame_count,
                'sampleOffset': frame_offset * data.HOP, 'sampleCount': frame_count * data.HOP, 'trimStart': 0.0
            }
            if emotions:
                emotion, intensity = emotions[number % len(emotions)]
                entry.update({'emotion': emotion, 'intensity': intensity})
            entries.append(entry)
            tokens_all.append(ids)
            durations_all.append(durations)
            seconds_all.append(durations.astype(numpy.float32) * 0.01)
            audio_all.append(numpy.clip(numpy.round(audio * 32768), -32768, 32767).astype(numpy.int16))
            frames_all.append(tracks)
            token_offset += len(ids)
            frame_offset += frame_count
    index = {
        'version': data.FORMAT_VERSION, 'sampleRate': data.SAMPLE_RATE, 'hop': data.HOP, 'frameRate': data.SAMPLE_RATE / data.HOP,
        'channels': data.CHANNELS, 'vocab': frontend.vocabulary(),
        'speakers': {key: {'pitchFloor': 60.0, 'pitchCeiling': 400.0, 'formantCeiling': 5000, 'bandwidth': 8000,
                           'sourceSampleRate': 16000, **info} for key, info in speakers.items()},
        'utterances': entries
    }
    numpy.save(os.path.join(directory, 'tokens.npy'), numpy.concatenate(tokens_all))
    numpy.save(os.path.join(directory, 'durations.npy'), numpy.concatenate(durations_all))
    numpy.save(os.path.join(directory, 'seconds.npy'), numpy.concatenate(seconds_all))
    numpy.save(os.path.join(directory, 'audio.npy'), numpy.concatenate(audio_all))
    numpy.save(os.path.join(directory, 'frames.npy'), numpy.concatenate(frames_all))
    with open(os.path.join(directory, 'index.json'), 'w', encoding='utf-8') as file:
        json.dump(index, file)
    with open(os.path.join(directory, 'alignment.jsonl'), 'w', encoding='utf-8') as file:
        file.write(''.join(json.dumps({'id': entry['id']}) + '\n' for entry in entries))
    return index


# The emotional corpus's speakers and styles (cremad.py's) for write_corpus
CREMAD_SPEAKERS = {'cremad-1001': {'sex': 'male', 'index': 0, 'pitchRanges': {'angry': [80.0, 300.0]}},
                   'cremad-1002': {'sex': 'female', 'index': 1}}
CREMAD_EMOTIONS = [('angry', 'high'), ('neutral', 'unspecified'), ('happy', 'low'), ('sad', 'medium')]


# Capture stdout and stderr - a context manager yielding (stdout, stderr) StringIO objects
@contextlib.contextmanager
def captured():
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield stdout, stderr


# Wait (polling) until predicate() is true - True, or False at the timeout (s). Tests of threads wait with this, not
# their own loops, so their coverage doesn't depend on the timing.
def wait_until(predicate, timeout=120):
    deadline = time.time() + timeout
    while not predicate():
        if time.time() >= deadline:
            return False
        time.sleep(0.01)
    return True


# A JavaScript literal of JSON tokens and concatenated string literals (package.py's manifest) - its value
def js_literal(text):
    return json.loads(re.sub(r'(?<!\\)"\s*\+\s*"', '', text))


# A packaged model module's (package.py's) manifest and weights, as tools/unpack.js unpacks it
def unpack_module(path):
    with open(path, encoding='utf-8') as file:
        text = file.read()
    manifest = js_literal(re.search(r'^export const manifest = (.*?);$', text, re.M | re.S).group(1))
    encoding = re.search(r"^export const encoding = '([^']*)';$", text, re.M).group(1)
    weights = base64.b64decode(re.search(r'^export const weights = `(.*?)`;$', text, re.M | re.S).group(1).replace('\n', ''))
    return manifest, zlib.decompress(weights, -15) if encoding == 'deflate-raw' else weights
