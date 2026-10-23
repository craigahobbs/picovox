# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The synthesizer parameter set - the interpretable, sample-rate-independent source-filter parameters that the network
predicts frame by frame, that dsynth.py (PyTorch, differentiable) and lib/render.js (JavaScript) render, and that
voices and styles transform. This module is the one definition: the JavaScript copy, lib/params.js, is generated
from it (python -m picovox_train.params --write-js; a unit test checks it is current).

Frames are FRAME_RATE (200) per second - one every 5 ms, frame t at time t / FRAME_RATE - and the synthesizers
interpolate them linearly between frames (amplitudes as linear amplitudes, so a dB value is converted first). A
sequence of T frames renders floor((T - 1) * sample_rate / FRAME_RATE) samples, as the former lib/klatt.js did.

Every parameter is in physical units, independent of the sample rate:

- Frequencies and bandwidths in Hz.
- Levels in dB. A level's linear amplitude is 10 ** (dB / 20); FLOOR_DB (-90 dB) and below is silence.
  - av - the voicing: 0 dB is an LF glottal flow derivative pulse train with a negative peak (Ee) of 1 - the same
    scale as the former lib/klatt.js's av of 1.
  - ah, af - the aspiration and frication noise: 0 dB is white Gaussian noise of unit variance at the reference rate
    (REF_RATE, 44.1 kHz) - so a power spectral density of 1 / 44100 per Hz, which the synthesizers keep at any rate.
  - ap1, ap2, ab - the frication spectrum's two resonance gains and flat bypass gain (relative to af).
- The filters are defined by their frequency responses: each is the former lib/klatt.js's digital filter designed at REF_RATE,
  evaluated at the frequency in Hz. At 44.1 kHz they are exactly the former lib/klatt.js's filters; at another rate the
  synthesizers apply the same responses (band-limited to the rate's Nyquist frequency).

The signal model:

    voicing     = av * LF(rd) * shelf(shelf_db) * tract           (harmonics of f0; jitter and shimmer per cycle)
    aspiration  = ah * noise * modulation * lowpass * tract       (modulation - pulsing with the glottal cycle)
    frication   = af * noise * |ab + ap1 * BP(fp1, bp1)^2 + ap2 * BP(fp2, bp2)^2| * HP(fhp)^2   (parallel - no tract)
    tract       = R(f1, b1) ... R(f6, b6) * R(fnp, bnp) / R(fnz, bnz) * higher poles
    output      = (voicing + aspiration + frication) * channel EQ   (optional, static - see EQ_BAND_COUNT)

The higher poles above F6 (a real tract keeps resonating every ~1 kHz, so the spectrum doesn't fall away) are
HIGHER_POLES poles at f6 + k * f6 / 5.5 (k = 1 ... HIGHER_POLES) with bandwidths 500 + 100 * (k - 1) Hz - the former lib/voices.js's
male higherFormants exactly - so they follow F6 and scale with the tract length.
"""

import argparse
import json
import math
import os
import sys

from . import ROOT


# The frame rate (frames per second) - the former lib/speak.js's 5 ms frames
FRAME_RATE = 200

# The reference sample rate - the filters' design rate and the noise level reference
REF_RATE = 44100

# Levels at or below this (dB) are silence
FLOOR_DB = -90.0

# The voicing's high-shelf corner frequency (Hz) - the former lib/voices.js's default shelfHz
SHELF_HZ = 2000.0

# The aspiration's glottal-cycle modulation depth (at full voicing) - the former lib/voices.js's default noiseModulation
NOISE_MODULATION = 0.7

# The number of fixed-spacing poles above F6, their spacing (a fraction of F6), and bandwidths (Hz)
HIGHER_POLES = 6
HIGHER_POLE_SPACING = 1 / 5.5
HIGHER_POLE_BANDWIDTH = 500.0
HIGHER_POLE_BANDWIDTH_STEP = 100.0

# The harmonic synthesis band edge - harmonics fade out from HARMONIC_TAPER_START to HARMONIC_TAPER_END of the
# sample rate (0.5 is the Nyquist frequency)
HARMONIC_TAPER_START = 0.44
HARMONIC_TAPER_END = 0.475

# The F0 floor (Hz) - lower values are clamped, as the former lib/klatt.js did
F0_MIN = 40.0

# The LF pulse harmonic table - Rd from LF_RD_MIN to LF_RD_MAX in steps of LF_RD_STEP, LF_HARMONICS harmonics, from
# LF_POINTS samples of one period
LF_RD_MIN = 0.3
LF_RD_MAX = 2.7
LF_RD_STEP = 0.01
LF_HARMONICS = 1024
LF_POINTS = 8192


# The channel equalizer - an optional static (per recording or per speaker, not per frame) gain curve applied to the
# whole output: EQ_BAND_COUNT gains (dB) at log-spaced frequencies from EQ_LOW_HZ to EQ_HIGH_HZ, interpolated linearly
# in dB over log frequency and held beyond the ends. It models what the source-filter model doesn't - the microphone
# and room, and a speaker's average spectral residual - so the per-frame parameters stay interpretable.
EQ_BAND_COUNT = 24
EQ_LOW_HZ = 60.0
EQ_HIGH_HZ = 16000.0
EQ_RANGE_DB = 30.0


# The parameters - (name, group, units, scale, low, high, default, doc). The scale is how a network should map a
# normalized output to the parameter: 'log' (geometric between low and high) or 'linear'. The range is the network's
# output range, generous enough for any human voice and our styles; the default is a neutral (silent) frame.
PARAMETERS = [
    # The source
    ('f0', 'source', 'Hz', 'log', 40.0, 600.0, 100.0, 'The fundamental frequency'),
    ('av', 'source', 'dB', 'linear', FLOOR_DB, 12.0, FLOOR_DB, 'The voicing amplitude (0 dB - an LF pulse train with Ee = 1)'),
    ('rd', 'source', 'Rd', 'linear', 0.3, 2.7, 1.0, 'The LF pulse shape - 0.3 pressed, ~1 modal, 2.7 lax and breathy'),
    ('shelf', 'source', 'dB', 'linear', -12.0, 24.0, 0.0,
     f'The voicing\'s high-shelf gain at {SHELF_HZ:.0f} Hz - spectral tilt correction (brightness, effort)'),
    ('jitter', 'source', 'fraction', 'linear', 0.0, 0.1, 0.0, 'Random cycle-to-cycle period variation (standard deviation)'),
    ('shimmer', 'source', 'fraction', 'linear', 0.0, 0.5, 0.0, 'Random cycle-to-cycle amplitude variation (standard deviation)'),
    ('ah', 'source', 'dB', 'linear', FLOOR_DB, 12.0, FLOOR_DB,
     'The aspiration noise amplitude (0 dB - unit-variance white noise at 44.1 kHz), through the tract'),
    ('af', 'source', 'dB', 'linear', FLOOR_DB, 12.0, FLOOR_DB,
     'The frication noise amplitude (0 dB - unit-variance white noise at 44.1 kHz), through the frication filter'),

    # The vocal tract - the cascade formants and the nasal pole-zero pair
    ('f1', 'tract', 'Hz', 'log', 120.0, 1400.0, 500.0, 'The first formant frequency'),
    ('f2', 'tract', 'Hz', 'log', 400.0, 3400.0, 1500.0, 'The second formant frequency'),
    ('f3', 'tract', 'Hz', 'log', 1000.0, 4500.0, 2500.0, 'The third formant frequency'),
    ('f4', 'tract', 'Hz', 'log', 2000.0, 6000.0, 3500.0, 'The fourth formant frequency'),
    ('f5', 'tract', 'Hz', 'log', 2500.0, 7000.0, 4500.0, 'The fifth formant frequency'),
    ('f6', 'tract', 'Hz', 'log', 3000.0, 8500.0, 5500.0, 'The sixth formant frequency (the higher poles follow it)'),
    ('b1', 'tract', 'Hz', 'log', 20.0, 1000.0, 80.0, 'The first formant bandwidth'),
    ('b2', 'tract', 'Hz', 'log', 20.0, 1500.0, 90.0, 'The second formant bandwidth'),
    ('b3', 'tract', 'Hz', 'log', 30.0, 2000.0, 150.0, 'The third formant bandwidth'),
    ('b4', 'tract', 'Hz', 'log', 40.0, 2500.0, 250.0, 'The fourth formant bandwidth'),
    ('b5', 'tract', 'Hz', 'log', 50.0, 3000.0, 300.0, 'The fifth formant bandwidth'),
    ('b6', 'tract', 'Hz', 'log', 60.0, 4000.0, 1000.0, 'The sixth formant bandwidth'),
    ('fnp', 'tract', 'Hz', 'log', 150.0, 4000.0, 270.0, 'The nasal (or lateral) pole frequency'),
    ('bnp', 'tract', 'Hz', 'log', 30.0, 2000.0, 100.0, 'The nasal pole bandwidth'),
    ('fnz', 'tract', 'Hz', 'log', 150.0, 4000.0, 270.0, 'The nasal zero frequency - equal to the pole, they cancel'),
    ('bnz', 'tract', 'Hz', 'log', 30.0, 2000.0, 100.0, 'The nasal zero bandwidth'),

    # The frication noise spectrum
    ('fp1', 'frication', 'Hz', 'log', 300.0, 12000.0, 3000.0, 'The frication noise\'s first resonance frequency'),
    ('bp1', 'frication', 'Hz', 'log', 50.0, 8000.0, 1000.0, 'The first resonance bandwidth'),
    ('ap1', 'frication', 'dB', 'linear', -60.0, 12.0, -60.0, 'The first resonance gain (fourth-order bandpass, 0 dB peak)'),
    ('fp2', 'frication', 'Hz', 'log', 300.0, 16000.0, 6000.0, 'The frication noise\'s second resonance frequency'),
    ('bp2', 'frication', 'Hz', 'log', 50.0, 10000.0, 2000.0, 'The second resonance bandwidth'),
    ('ap2', 'frication', 'dB', 'linear', -60.0, 12.0, -60.0, 'The second resonance gain'),
    ('ab', 'frication', 'dB', 'linear', -60.0, 12.0, -60.0, 'The flat (bypass) gain'),
    ('fhp', 'frication', 'Hz', 'log', 50.0, 10000.0, 500.0, 'The high-pass cutoff (24 dB/octave below it)')
]

NAMES = [parameter[0] for parameter in PARAMETERS]
INDEX = {name: ix for ix, name in enumerate(NAMES)}
COUNT = len(NAMES)
DEFAULTS = {parameter[0]: parameter[6] for parameter in PARAMETERS}

# The level parameters (dB) - converted to linear amplitudes before interpolation
DB_NAMES = [parameter[0] for parameter in PARAMETERS if parameter[2] == 'dB' and parameter[0] != 'shelf']

# The frequency parameters that the vocal tract length scales (the former lib/speak.js's tractScale list)
TRACT_FREQUENCIES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'fnp', 'fnz', 'fp1', 'fp2', 'fhp']


# Normalize parameter values (a torch tensor or numpy array [..., COUNT]) to 0-1 - the inverse of denormalize
def normalize(values):
    columns = []
    for ix, (_, _, _, scale, low, high, _, _) in enumerate(PARAMETERS):
        value = values[..., ix]
        if scale == 'log':
            columns.append((_log(value) - math.log(low)) / (math.log(high) - math.log(low)))
        else:
            columns.append((value - low) / (high - low))
    return _stack(columns, values)


# Map normalized (0-1) values [..., COUNT] to parameter values - a network's sigmoid outputs to parameters
def denormalize(unit):
    columns = []
    for ix, (_, _, _, scale, low, high, _, _) in enumerate(PARAMETERS):
        value = unit[..., ix]
        if scale == 'log':
            columns.append(_exp(math.log(low) + value * (math.log(high) - math.log(low))))
        else:
            columns.append(low + value * (high - low))
    return _stack(columns, unit)


def _log(value):
    return value.log() if hasattr(value, 'log') and callable(value.log) else __import__('numpy').log(value)


def _exp(value):
    return value.exp() if hasattr(value, 'exp') and callable(value.exp) else __import__('numpy').exp(value)


def _stack(columns, like):
    if type(like).__module__.startswith('torch'):
        return __import__('torch').stack(columns, dim=-1)
    return __import__('numpy').stack(columns, axis=-1)


#
# The bandwidth transform of parameter values ([..., COUNT] torch tensors or numpy arrays, returned as copies) - the
# inference default (the export's bandwidthScale); the voice and style transforms are lib/transforms.js's
#

def _copy(values):
    return values.clone() if hasattr(values, 'clone') else values.copy()


# Scale the formant bandwidths (b1-b5) - below 1 sharpens the resonances (a network trained by spectral regression
# over-smooths them), above 1 dulls them
def bandwidth_scale(values, scale, names=('b1', 'b2', 'b3', 'b4', 'b5')):
    out = _copy(values)
    for name in names:
        out[..., INDEX[name]] = out[..., INDEX[name]] * scale
    return out


#
# Parameter files - JSON {"frameRate": 200, "names": [...], "frames": [[value, ...], ...], "eq": [dB, ...] (optional)},
# shared with nn/js
#

# Write parameter frames (a [T, COUNT] array-like) to a JSON parameter file, with an optional channel equalizer (dB)
def write_params(path, frames, eq=None):
    rows = [[round(float(value), 4) for value in row[:COUNT]] for row in frames]
    data = {'frameRate': FRAME_RATE, 'names': NAMES, 'frames': rows}
    if eq is not None:
        data['eq'] = [round(float(value), 3) for value in eq]
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, separators=(',', ':'))


#
# The JavaScript copy - lib/params.js
#

JS_PATH = os.path.join(ROOT, 'lib', 'params.js')


# A JavaScript literal of a value (single-quoted strings)
def _js(value):
    if isinstance(value, str):
        return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_js(item) for item in value) + ']'
    return json.dumps(value)


def javascript():
    constants = {
        'FRAME_RATE': FRAME_RATE, 'REF_RATE': REF_RATE, 'FLOOR_DB': FLOOR_DB, 'SHELF_HZ': SHELF_HZ,
        'NOISE_MODULATION': NOISE_MODULATION, 'HIGHER_POLES': HIGHER_POLES, 'HIGHER_POLE_SPACING': HIGHER_POLE_SPACING,
        'HIGHER_POLE_BANDWIDTH': HIGHER_POLE_BANDWIDTH, 'HIGHER_POLE_BANDWIDTH_STEP': HIGHER_POLE_BANDWIDTH_STEP,
        'HARMONIC_TAPER_START': HARMONIC_TAPER_START, 'HARMONIC_TAPER_END': HARMONIC_TAPER_END, 'F0_MIN': F0_MIN,
        'LF_RD_MIN': LF_RD_MIN, 'LF_RD_MAX': LF_RD_MAX, 'LF_RD_STEP': LF_RD_STEP, 'LF_HARMONICS': LF_HARMONICS,
        'LF_POINTS': LF_POINTS, 'EQ_BAND_COUNT': EQ_BAND_COUNT, 'EQ_LOW_HZ': EQ_LOW_HZ, 'EQ_HIGH_HZ': EQ_HIGH_HZ,
        'EQ_RANGE_DB': EQ_RANGE_DB
    }
    lines = [
        '// The synthesizer parameter set - GENERATED from train/src/picovox_train/params.py (python -m picovox_train.params',
        '// --write-js); do not edit. See params.py there for the parameters\' meanings and the signal model.',
        ''
    ]
    for name, value in constants.items():
        lines.append(f'export const {name} = {_js(value)};')
    lines.append('')
    lines.append('// The parameters - {name, group, units, scale, low, high, default, doc}')
    lines.append('export const PARAMETERS = [')
    for name, group, units, scale, low, high, default, doc in PARAMETERS:
        item = {'name': name, 'group': group, 'units': units, 'scale': scale, 'low': low, 'high': high, 'default': default,
                'doc': doc}
        body = ', '.join(f"'{key}': {_js(value)}" for key, value in item.items() if key != 'doc')
        lines.append(f'    {{{body},')
        lines.append(f"        'doc': {_js(doc)}}},")
    lines[-1] = lines[-1].rstrip(',')
    lines.append('];')
    lines.append('')
    lines.append('// The parameter names, in order')
    lines.append('export const NAMES = PARAMETERS.map((parameter) => parameter.name);')
    lines.append('')
    lines.append('// The default (silent) frame')
    lines.append('export const DEFAULTS = Object.fromEntries(PARAMETERS.map((parameter) => [parameter.name, parameter.default]));')
    lines.append('')
    lines.append('// The level parameters (dB), converted to linear amplitudes before interpolation')
    lines.append(f'export const DB_NAMES = {_js(DB_NAMES)};')
    lines.append('')
    lines.append('// The frequency parameters the vocal tract length scales')
    lines.append(f'export const TRACT_FREQUENCIES = {_js(TRACT_FREQUENCIES)};')
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.params', description='The synthesizer parameter set')
    parser.add_argument('--write-js', action='store_true', required=True, help='write lib/params.js')
    parser.parse_args(argv)
    with open(JS_PATH, 'w', encoding='utf-8') as file:
        file.write(javascript())
    return 0


if __name__ == '__main__':
    sys.exit(main())
