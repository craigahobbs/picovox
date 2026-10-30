# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The data tests' fakes - an HTTP response, an in-process multiprocessing.Pool, fake modules, a fake acoustic analysis
(parselmouth's), and synthetic recordings with phone labels that match the front end's tokens
"""

import contextlib
import io
import sys
import types

import numpy

from picovox_train import frontend
from picovox_train.wav import read_wav



class FakeResponse(io.BytesIO):
    """A urlopen response - its body, status, and Content-Length header (if length)"""

    def __init__(self, body, status=200, length=True):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Length': str(len(body))} if length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class FakePool:
    """multiprocessing.Pool, in this process (so the workers' code is covered) - results, if given, replaces the work's
    results"""

    results = None

    def __init__(self, processes=None, initializer=None, initargs=()):
        self.processes = processes
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def imap(self, function, iterable, chunksize=1):
        assert chunksize >= 1
        return iter(FakePool.results) if FakePool.results is not None else map(function, iterable)

    def map(self, function, iterable, chunksize=1):
        return list(self.imap(function, iterable, chunksize))

    def imap_unordered(self, function, iterable, chunksize=1):
        return self.imap(function, iterable, chunksize)


# Fake modules in sys.modules (by name) for the context - only those names restored after (unittest.mock.patch.dict would
# drop every module imported meanwhile, and a compiled extension can't be imported twice)
@contextlib.contextmanager
def fake_modules(**modules):
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in saved:
            del sys.modules[name]
        sys.modules.update({name: module for name, module in saved.items() if module is not None})


class FakeSound:
    """A fake recording (a path or samples) and its analyses: pitch by a frame autocorrelation
    (voiced where the signal is loud and periodic), harmonicity from its peak, and fixed formants wherever the signal
    isn't silent"""

    def __init__(self, values, sampling_frequency=None):
        if isinstance(values, str):
            values, sampling_frequency = read_wav(values)
            values = values[:, 0]
        self.samples = numpy.asarray(values, dtype=numpy.float64)
        self.rate = sampling_frequency

    # The frames' center times, each frame's RMS, and its autocorrelation peak (lag, normalized) within the pitch range
    def frames(self, time_step, pitch_floor=75.0, pitch_ceiling=600.0):
        half = int(0.02 * self.rate)
        times = (numpy.arange(int(len(self.samples) / self.rate / time_step)) + 0.5) * time_step
        rms = []
        peaks = []
        for time in times:
            center = int(time * self.rate)
            window = self.samples[max(0, center - half):center + half]
            rms.append(float(numpy.sqrt(numpy.mean(window ** 2))) if len(window) else 0.0)
            lags = numpy.arange(int(self.rate / pitch_ceiling), min(int(self.rate / pitch_floor), len(window) - 1))
            energy = float(numpy.dot(window, window))
            correlations = [float(numpy.dot(window[:-lag], window[lag:])) / energy for lag in lags] if energy > 0 and len(lags) else [0.0]
            best = int(numpy.argmax(correlations))
            peaks.append((int(lags[best]) if len(lags) else 1, correlations[best]))
        return times, numpy.array(rms), peaks

    def to_pitch_ac(self, time_step, pitch_floor, pitch_ceiling):
        times, rms, peaks = self.frames(time_step, pitch_floor, pitch_ceiling)
        voiced = [level > 0.01 and peak > 0.5 for level, (_, peak) in zip(rms, peaks)]
        return FakePitch(times, [self.rate / lag if ok else 0.0 for ok, (lag, _) in zip(voiced, peaks)],
                         [peak if ok else 0.0 for ok, (_, peak) in zip(voiced, peaks)])

    def to_harmonicity_cc(self, time_step, minimum_pitch):
        times, rms, peaks = self.frames(time_step, minimum_pitch)
        hnr = [10 * numpy.log10(min(peak, 0.999) / (1 - min(peak, 0.999))) if level > 0.01 and peak > 0.5 else -200.0
               for level, (_, peak) in zip(rms, peaks)]
        return FakeTrack(times, numpy.array([hnr]))

    def to_formant_burg(self, time_step, max_number_of_formants, maximum_formant, window_length=0.025, pre_emphasis_from=50):
        assert max_number_of_formants == 5 and maximum_formant > 0 and window_length > 0 and pre_emphasis_from > 0
        times, rms, _ = self.frames(time_step)
        return FakeFormant(times, rms > 0.001)


class FakeTrack:
    """A fake analysis's frames - their times and values"""

    def __init__(self, times, values):
        self.times = times
        self.values = values

    def xs(self):
        return self.times

    # The frame nearest a time
    def frame(self, time):
        return int(numpy.clip(numpy.round(time / (self.times[1] - self.times[0]) - 0.5), 0, len(self.times) - 1))


class FakePitch(FakeTrack):
    """A fake pitch track - its selected candidates (frequency, strength; 0 Hz unvoiced) and the value at a time (NaN
    unvoiced)"""

    def __init__(self, times, frequency, strength):
        super().__init__(times, None)
        self.selected_array = numpy.array(list(zip(frequency, strength)), dtype=[('frequency', float), ('strength', float)])


class FakeFormant(FakeTrack):
    """A fake formant track - formant n at 1000n - 500 Hz with an 80n Hz bandwidth where the signal isn't silent (else NaN)"""

    def get_value_at_time(self, number, time):
        return 1000.0 * number - 500.0 if self.values[self.frame(time)] else float('nan')

    def get_bandwidth_at_time(self, number, time):
        return 80.0 * number if self.values[self.frame(time)] else float('nan')


# The fake acoustic analysis module, for fake_modules
ANALYSIS = types.SimpleNamespace(Sound=FakeSound)


# Seconds of each phone, and of the edge silences
PHONE_SECONDS = 0.08
EDGE_SECONDS = 0.25


# A voiced, harmonic signal (a buzz at f0 with a vowel-like spectral tilt) of a duration at a sample rate
def buzz(seconds, rate, f0=150.0, seed=0):
    times = numpy.arange(int(round(seconds * rate))) / rate
    signal = sum(numpy.sin(2 * numpy.pi * f0 * harmonic * times) / harmonic for harmonic in range(1, 12))
    return 0.2 * signal + 0.002 * numpy.random.default_rng(seed).standard_normal(len(times))


# A text's phone labels - [(label, start, end)] (our token names, PHONE_SECONDS each after EDGE_SECONDS of silence) and
# the recording's seconds (EDGE_SECONDS more after them)
def text_labels(text):
    phones = [token for token in frontend.tokenize(text)['tokens'] if token not in frontend.BOUNDARIES]
    labels = [(phone, EDGE_SECONDS + index * PHONE_SECONDS, EDGE_SECONDS + (index + 1) * PHONE_SECONDS)
              for index, phone in enumerate(phones)]
    return labels, EDGE_SECONDS + len(phones) * PHONE_SECONDS + EDGE_SECONDS


# The recording of labels - silence in the pauses, a buzz in the phones - at a sample rate (float64 samples)
def labels_audio(labels, seconds, rate, seed=0):
    samples = numpy.zeros(int(round(seconds * rate)))
    for label, start, end in labels:
        if label.lower() not in ('pau', 'sil', 'sp', ''):
            begin = int(round(start * rate))
            segment = buzz(end - start, rate, seed=seed)
            samples[begin:begin + len(segment)] = segment[:len(samples) - begin]
    return samples
