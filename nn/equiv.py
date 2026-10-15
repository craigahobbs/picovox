# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The equivalence check of the two synthesizers - nn/dsynth.py (PyTorch) and nn/js/render.js (JavaScript) render the
same parameter sequences, random and realistic (the rule voice's, through nn/js/bridge.js), at several sample rates.

- With the shared counter-based noise (nn/rng.py, nn/js/rng.js) the renderings should match sample for sample: the
  SNR of their difference (float64 PyTorch on the CPU, and float32 on MPS, against the JavaScript float32 output).
- With independent noise (torch.randn - the training mode) they should match spectrally: the log-spectral distance,
  next to the distance between two JavaScript renderings with different seeds (the realization floor).
- One random sequence has a random channel equalizer (nn/params.py EQ_BAND_COUNT).
- With --correction, two more random sequences carry a random, smoothly varying spectral correction (nn/params.py
  CORRECTION_*; 20 and 16 bands, +-12 dB) - the second with the channel equalizer too.
- The JavaScript renderer streamed (frames pushed a few at a time, small blocks) should match it rendered at once.

    build/venv-nn/bin/python -m nn.equiv [--rates 16000 22050 44100]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy
import torch

from . import dsynth, params, spectral


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_CLI = os.path.join(ROOT, 'nn', 'js', 'rendercli.js')
SPEAK_CLI = os.path.join(ROOT, 'nn', 'js', 'speakparams.js')


# A random, smoothly varying parameter sequence - every parameter wandering through the middle of its range
def random_sequence(frame_count, seed):
    state = numpy.random.RandomState(seed)
    walk = numpy.cumsum(state.randn(frame_count, params.COUNT) * 0.04, axis=0) + state.rand(1, params.COUNT) * 2
    unit = 0.15 + 0.7 * numpy.abs((walk % 2) - 1)
    values = params.denormalize(unit)
    # Voicing that starts silent, plausible noise levels, rising and falling F0, and some jitter and shimmer
    values[:, params.INDEX['av']] = numpy.where(numpy.arange(frame_count) < frame_count // 5, params.FLOOR_DB,
                                                -12 + 10 * numpy.sin(numpy.arange(frame_count) / 15))
    values[:, params.INDEX['ah']] -= 35
    values[:, params.INDEX['af']] = numpy.where((numpy.arange(frame_count) // 40) % 2 == 1, -20.0, params.FLOOR_DB)
    values[:, params.INDEX['f0']] = 110 + 60 * numpy.sin(numpy.arange(frame_count) / 50)
    values[:, params.INDEX['jitter']] = 0.01
    values[:, params.INDEX['shimmer']] = 0.05
    return values


# A random, smoothly varying spectral correction [T, bands] (dB, within +-range_db)
def random_correction(frame_count, bands, seed, range_db=12.0):
    state = numpy.random.RandomState(seed + 100)
    walk = numpy.cumsum(state.randn(frame_count, bands) * 0.05, axis=0) + state.randn(1, bands)
    smooth = numpy.apply_along_axis(lambda row: numpy.convolve(row, [0.25, 0.5, 0.25], mode='same'), 1, walk)
    return range_db * numpy.tanh(smooth)


# Render a parameter file with nn/js/render.js - float32 samples and the render statistics
def render_js(path, rate, seed, push=0, block=4096):
    with tempfile.NamedTemporaryFile(suffix='.f32') as raw:
        command = ['node', RENDER_CLI, path, '-r', str(rate), '-s', str(seed), '--raw', raw.name, '--block', str(block)]
        if push:
            command += ['--push', str(push)]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        return numpy.fromfile(raw.name, dtype=numpy.float32), json.loads(process.stdout)


# The SNR (dB) of a rendering's difference from a reference
def snr(reference, other):
    error = numpy.asarray(other, dtype=numpy.float64) - numpy.asarray(reference, dtype=numpy.float64)
    return float(10 * numpy.log10((numpy.asarray(reference, dtype=numpy.float64) ** 2).mean() / max((error ** 2).mean(), 1e-30)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.equiv', description='The Python and JavaScript synthesizers, compared')
    parser.add_argument('--rates', type=int, nargs='+', default=[16000, 22050, 44100])
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--correction', action='store_true', help='also the sequences with a spectral correction')
    parser.add_argument('--sequences', nargs='+', default=None, help='only these sequences (names)')
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as directory:
        # The sequences - two random, and the rule voice (male and female)
        sequences = {}
        for seed in (1, 2):
            path = os.path.join(directory, f'random{seed}.json')
            # The second with a random channel equalizer
            eq = numpy.random.RandomState(seed).uniform(-12, 12, params.EQ_BAND_COUNT) if seed == 2 else None
            params.write_params(path, random_sequence(400, seed), eq=eq)
            sequences[f'random-{seed}'] = path
        if args.correction:
            for seed, bands in ((3, 20), (4, 16)):
                path = os.path.join(directory, f'random{seed}.json')
                eq = numpy.random.RandomState(seed).uniform(-12, 12, params.EQ_BAND_COUNT) if seed == 4 else None
                values = random_sequence(400, seed)
                params.write_params(path, numpy.concatenate([values, random_correction(len(values), bands, seed)], axis=1), eq=eq)
                sequences[f'random-{seed}-correction{bands}'] = path
        for voice in ('male', 'female'):
            path = os.path.join(directory, f'{voice}.json')
            subprocess.run(['node', SPEAK_CLI, '-v', voice, '--params', path,
                            'The birch canoe slid on the smooth planks. Is it heavy? Glue the sheet to the dark blue background!'],
                           capture_output=True, check=True)
            sequences[f'rule-{voice}'] = path

        rows = []
        for name, path in sequences.items():
            if args.sequences and name not in args.sequences:
                continue
            values = params.read_params(path)
            eq = params.read_eq(path)
            correction = params.read_correction(path)
            for rate in args.rates:
                js, stats = render_js(path, rate, seed=5)
                js_streamed, _ = render_js(path, rate, seed=5, push=7, block=333)
                js_other, _ = render_js(path, rate, seed=6)
                row = {'sequence': name, 'rate': rate, 'seconds': round(len(js) / rate, 2), 'jsRealtime': round(stats['realtime'], 1),
                       'streamedIdentical': bool(len(js_streamed) == len(js) and numpy.array_equal(js_streamed, js))}
                for label, dtype, device in [('cpu64', torch.float64, 'cpu'), ('mps32', torch.float32, args.device)]:
                    synth = dsynth.DSynth(rate, noise='counter').to(device)
                    with torch.no_grad():
                        eq_tensor = None if eq is None else torch.tensor(eq, dtype=dtype, device=device)
                        correction_tensor = None if correction is None else torch.tensor(correction, dtype=dtype, device=device)
                        python = synth(torch.tensor(values, dtype=dtype, device=device), seed=5, eq=eq_tensor,
                                       correction=correction_tensor).cpu().numpy()
                    row[f'snr_{label}'] = round(snr(python, js), 1) if len(python) == len(js) else None
                synth = dsynth.DSynth(rate, noise='torch').to(args.device)
                with torch.no_grad():
                    eq_tensor = None if eq is None else torch.tensor(eq, dtype=torch.float32, device=args.device)
                    correction_tensor = None if correction is None else torch.tensor(correction, dtype=torch.float32, device=args.device)
                    python = synth(torch.tensor(values, dtype=torch.float32, device=args.device), seed=11, eq=eq_tensor,
                                   correction=correction_tensor).cpu().numpy()
                row['lsdTorchNoise'] = round(spectral.log_spectral_distance(js, python, rate), 3)
                row['lsdJsSeeds'] = round(spectral.log_spectral_distance(js, js_other, rate), 3)
                rows.append(row)
                print(json.dumps(row), file=sys.stderr)
    json.dump(rows, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
