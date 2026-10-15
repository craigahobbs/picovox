# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Analysis by synthesis - fit parameter trajectories directly to recordings by gradient descent through nn/dsynth.py,
the ceiling a network predicting this synthesizer's parameters can reach.

    build/venv-nn/bin/python -m nn.fit [--speakers slt bdl] [--count 6] [--steps 1500] [--rate 16000]

Each recording is initialized from Praat (parselmouth): F0 (autocorrelation, unvoiced frames interpolated), F1-F5 and
their bandwidths (Burg), and the voicing and noise levels from the intensity and the voicing decision. Then every
parameter but F0, jitter, and shimmer (fixed: F0 from Praat, jitter 0.5%, shimmer 2%) is optimized with Adam in the
network's own parameterization - a sigmoid per parameter mapped to its range by nn/params.py's denormalize - against
the multi-resolution STFT loss (nn/spectral.py), with a light smoothness penalty and a formant-order penalty. Each
step draws new noise, so the fit can't chase one noise realization. With --eq, a channel equalizer (nn/params.py) is
fit per recording too.

The fitted re-synthesis is scored against the original recording: the spectral loss, the log-spectral distance, and the
listener panel (UTMOS, and ECAPA speaker similarity to the recording). Results go to build/nn/fit/ (fitted parameter
files, WAVs, results.json).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy
import parselmouth
import soundfile
import soxr
import torch

from . import dsynth, params, spectral
from .bridge import cosine, listen


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCTIC = os.path.join(ROOT, 'data', 'corpora', 'arctic')

# Per-speaker analysis settings - pitch range and formant ceiling
SPEAKERS = {
    'bdl': {'pitch': (60, 250), 'ceiling': 5000, 'f4': 3500, 'f5': 4500, 'f6': 5500},
    'rms': {'pitch': (55, 220), 'ceiling': 5000, 'f4': 3500, 'f5': 4500, 'f6': 5500},
    'slt': {'pitch': (100, 400), 'ceiling': 5500, 'f4': 4100, 'f5': 4900, 'f6': 6000},
    'clb': {'pitch': (100, 400), 'ceiling': 5500, 'f4': 4100, 'f5': 4900, 'f6': 6000}
}

# The parameters held fixed (not optimized)
FIXED = ['f0', 'jitter', 'shimmer']


# Read a recording at a sample rate (mono)
def read_audio(path, sample_rate):
    audio, rate = soundfile.read(path, dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        audio = soxr.resample(audio, rate, sample_rate)
    return audio


# The initial parameter frames [T, COUNT] for a recording, from Praat
def initialize(audio, sample_rate, speaker, very_accurate=True):
    settings = SPEAKERS[speaker]
    frame_count = int(len(audio) * params.FRAME_RATE / sample_rate) + 1
    times = numpy.arange(frame_count) / params.FRAME_RATE
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    pitch = sound.to_pitch_ac(time_step=0.005, pitch_floor=settings['pitch'][0], pitch_ceiling=settings['pitch'][1],
                              very_accurate=very_accurate)
    formant = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=settings['ceiling'],
                                    window_length=0.025)
    intensity = sound.to_intensity(minimum_pitch=settings['pitch'][0], time_step=0.005)

    values = numpy.tile(numpy.array([params.DEFAULTS[name] for name in params.NAMES]), (frame_count, 1))
    column = lambda name: values[:, params.INDEX[name]]  # pylint: disable=unnecessary-lambda-assignment

    # F0 - voiced frames from Praat, unvoiced interpolated (the median at the ends)
    f0 = numpy.array([pitch.get_value_at_time(time) for time in times])
    voiced = ~numpy.isnan(f0)
    if voiced.any():
        f0 = numpy.interp(times, times[voiced], f0[voiced])
    else:
        f0 = numpy.full(frame_count, float(numpy.mean(settings['pitch'])))
    values[:, params.INDEX['f0']] = f0

    # Formants and bandwidths - Praat's where defined, else the speaker defaults
    defaults = {'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': settings['f4'], 'f5': settings['f5']}
    for number in range(1, 6):
        track = numpy.array([formant.get_value_at_time(number, time) for time in times])
        bandwidth = numpy.array([formant.get_bandwidth_at_time(number, time) for time in times])
        good = ~numpy.isnan(track)
        track = numpy.interp(times, times[good], track[good]) if good.sum() > 1 else numpy.full(frame_count, defaults[f'f{number}'])
        good_bandwidth = ~numpy.isnan(bandwidth)
        bandwidth = numpy.interp(times, times[good_bandwidth], bandwidth[good_bandwidth]) if good_bandwidth.sum() > 1 else \
            numpy.full(frame_count, 100.0 * number)
        values[:, params.INDEX[f'f{number}']] = track
        values[:, params.INDEX[f'b{number}']] = numpy.clip(bandwidth, 40, 600)
    values[:, params.INDEX['f6']] = settings['f6']
    values[:, params.INDEX['b6']] = 500

    # Levels - the intensity (dB), voicing where Praat finds voicing, else noise
    level = numpy.array([intensity.get_value(time) for time in times], dtype=float)
    level = numpy.nan_to_num(level, nan=float(numpy.nanmin(level)) if not numpy.isnan(level).all() else 0)
    relative = level - level.max()
    voiced_now = numpy.array([not math.isnan(pitch.get_value_at_time(time)) for time in times])
    values[:, params.INDEX['av']] = numpy.where(voiced_now, relative - 8, -60)
    values[:, params.INDEX['ah']] = numpy.where(voiced_now, relative - 45, relative - 25)
    values[:, params.INDEX['af']] = numpy.where(voiced_now, relative - 40, relative - 15)
    values[:, params.INDEX['rd']] = 1.0
    values[:, params.INDEX['shelf']] = 6.0
    values[:, params.INDEX['jitter']] = 0.005
    values[:, params.INDEX['shimmer']] = 0.02

    # The frication shape - broad resonances, a modest bypass
    for name, value in [('fp1', 2500), ('bp1', 1500), ('ap1', 0), ('fp2', 5500), ('bp2', 2500), ('ap2', -6), ('ab', -20),
                        ('fhp', 800)]:
        values[:, params.INDEX[name]] = value
    for name in ['f1', 'f2', 'f3']:
        column(name)[:] = numpy.clip(column(name), params.PARAMETERS[params.INDEX[name]][4] * 1.05,
                                     params.PARAMETERS[params.INDEX[name]][5] * 0.95)
    return values


# Fit a batch of recordings (lists of audio arrays and initial frames) - returns the fitted frames and the loss history
def fit(audios, initials, sample_rate, device, steps=1500, learning_rate=0.02, smooth_weight=0.05, log_every=100, use_eq=False,
        use_residual=False):
    synth = dsynth.DSynth(sample_rate).to(device)
    frame_count = max(len(initial) for initial in initials)
    sample_total = dsynth.sample_count(frame_count, sample_rate)

    # Pad to a common length - silence after each recording
    targets = numpy.zeros((len(audios), sample_total))
    initial = numpy.zeros((len(audios), frame_count, params.COUNT))
    for ix, (audio, frames) in enumerate(zip(audios, initials)):
        length = min(len(audio), sample_total)
        targets[ix, :length] = audio[:length]
        initial[ix, :len(frames)] = frames
        initial[ix, len(frames):] = frames[-1]
        initial[ix, len(frames):, params.INDEX['av']] = params.FLOOR_DB
        initial[ix, len(frames):, params.INDEX['ah']] = params.FLOOR_DB
        initial[ix, len(frames):, params.INDEX['af']] = params.FLOOR_DB
    target = torch.tensor(targets, dtype=torch.float32, device=device)

    # The optimized logits (the network's parameterization) and the fixed parameters
    unit = numpy.clip(params.normalize(initial), 0.01, 0.99)
    logits = torch.tensor(numpy.log(unit / (1 - unit)), dtype=torch.float32, device=device, requires_grad=True)
    fixed_mask = torch.zeros(params.COUNT, device=device)
    for name in FIXED:
        fixed_mask[params.INDEX[name]] = 1
    fixed_values = torch.tensor(initial, dtype=torch.float32, device=device)
    # The channel equalizer (optional) - EQ_RANGE_DB * tanh of its free variables, one curve per recording
    eq_free = torch.zeros(len(audios), params.EQ_BAND_COUNT, device=device, requires_grad=True)
    # The residual noise's band levels (optional) - dB from FLOOR_DB to 12 by a sigmoid, starting at -70 dB
    start = (-70 - params.FLOOR_DB) / (12 - params.FLOOR_DB)
    residual_free = torch.full((len(audios), frame_count, params.RESIDUAL_BAND_COUNT), math.log(start / (1 - start)),
                               device=device, requires_grad=True)
    variables = [logits] + ([eq_free] if use_eq else []) + ([residual_free] if use_residual else [])
    optimizer = torch.optim.Adam(variables, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.05)
    formant_index = [params.INDEX[f'f{number}'] for number in range(1, 7)]

    def current():
        values = params.denormalize(torch.sigmoid(logits))
        return values * (1 - fixed_mask) + fixed_values * fixed_mask

    history = []
    for step in range(steps):
        optimizer.zero_grad()
        values = current()
        eq = params.EQ_RANGE_DB * torch.tanh(eq_free) if use_eq else None
        residual = params.FLOOR_DB + (12 - params.FLOOR_DB) * torch.sigmoid(residual_free) if use_residual else None
        audio = synth(values, seed=[step * 97 + ix for ix in range(len(audios))], eq=eq, residual=residual)
        loss = spectral.multi_resolution_stft_loss(audio, target)
        # Smoothness (of the normalized trajectories) and formant order
        unit_values = torch.sigmoid(logits)
        smooth = (unit_values[:, 1:] - unit_values[:, :-1]).pow(2).mean()
        formants = values[..., formant_index]
        order = torch.relu(formants[..., :-1] + 150 - formants[..., 1:]).mean() / 100
        total = loss + smooth_weight * smooth + order
        if use_eq:
            # A smooth curve - its second differences (dB) penalized
            total = total + 1e-4 * (eq[:, 2:] - 2 * eq[:, 1:-1] + eq[:, :-2]).pow(2).mean()
        total.backward()
        optimizer.step()
        scheduler.step()
        history.append(float(loss))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'  step {step:5d}  loss {float(loss):.4f}  smooth {float(smooth):.5f}  order {float(order):.4f}', file=sys.stderr)
    with torch.no_grad():
        fitted = current().cpu().numpy()
        eq = (params.EQ_RANGE_DB * torch.tanh(eq_free)).cpu().numpy() if use_eq else None
        residual = (params.FLOOR_DB + (12 - params.FLOOR_DB) * torch.sigmoid(residual_free)).cpu().numpy() if use_residual else None
    return fitted, eq, residual, history


# The spectral loss of a rendering against a target (numpy arrays)
def spectral_loss(audio, target):
    length = min(len(audio), len(target))
    with torch.no_grad():
        return float(spectral.multi_resolution_stft_loss(torch.tensor(audio[None, :length], dtype=torch.float32),
                                                         torch.tensor(target[None, :length], dtype=torch.float32)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.fit', description='Fit parameter trajectories to recordings')
    parser.add_argument('--speakers', nargs='+', default=['slt', 'bdl'])
    parser.add_argument('--count', type=int, default=6, help='recordings per speaker')
    parser.add_argument('--first', type=int, default=1, help='the first ARCTIC "a" prompt number')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--rate', type=int, default=16000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--smooth', type=float, default=0.05, help='the smoothness penalty weight')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'nn', 'fit'))
    parser.add_argument('--eq', action='store_true', help='also fit a channel equalizer per recording')
    parser.add_argument('--residual', action='store_true', help='also fit the residual noise band levels (experimental)')
    parser.add_argument('--no-listeners', action='store_true')
    args = parser.parse_args(argv)

    results = {'rate': args.rate, 'steps': args.steps, 'eq': args.eq, 'residual': args.residual, 'smooth': args.smooth, 'speakers': {}}
    groups = {}
    synth = dsynth.DSynth(args.rate).to(args.device)
    for speaker in args.speakers:
        directory = os.path.join(args.out, speaker)
        os.makedirs(directory, exist_ok=True)
        names = [f'arctic_a{number:04d}' for number in range(args.first, args.first + args.count)]
        audios = []
        initials = []
        for name in names:
            audio = read_audio(os.path.join(ARCTIC, f'cmu_us_{speaker}_arctic', 'wav', f'{name}.wav'), args.rate)
            audio = audio * 0.5 / numpy.abs(audio).max()
            audios.append(audio)
            initials.append(initialize(audio, args.rate, speaker))
        print(f'{speaker}: fitting {len(names)} recordings ({sum(len(audio) for audio in audios) / args.rate:.1f} s)', file=sys.stderr)
        start = time.time()
        fitted, eqs, residuals, history = fit(audios, initials, args.rate, args.device, steps=args.steps, learning_rate=args.lr,
                                              smooth_weight=args.smooth, use_eq=args.eq, use_residual=args.residual)
        fit_seconds = time.time() - start

        # Render the fits and the Praat initializations, and score them against the recordings
        rows = []
        group = {'original': [], 'fitted': [], 'initial': []}
        for ix, name in enumerate(names):
            frames = fitted[ix, :len(initials[ix])]
            eq = None if eqs is None else eqs[ix]
            params.write_params(os.path.join(directory, f'{name}.json'), frames, eq=eq)
            with torch.no_grad():
                eq_tensor = None if eq is None else torch.tensor(eq, dtype=torch.float32, device=args.device)
                residual = None if residuals is None else \
                    torch.tensor(residuals[ix, :len(initials[ix])], dtype=torch.float32, device=args.device)
                rendered = synth(torch.tensor(frames, dtype=torch.float32, device=args.device), seed=1, eq=eq_tensor,
                                 residual=residual).cpu().numpy()
                initial_audio = synth(torch.tensor(initials[ix], dtype=torch.float32, device=args.device), seed=1).cpu().numpy()
            original = audios[ix][:len(rendered)]
            paths = {}
            for key, audio in [('original', original), ('fitted', rendered), ('initial', initial_audio)]:
                paths[key] = os.path.join(directory, f'{name}-{key}.wav')
                soundfile.write(paths[key], audio * 0.89 / max(1e-9, numpy.abs(audio).max()), args.rate)
                group[key].append(paths[key])
            rows.append({
                'name': name, 'seconds': round(len(original) / args.rate, 2),
                'lossFitted': round(spectral_loss(rendered, original), 4), 'lossInitial': round(spectral_loss(initial_audio, original), 4),
                'lsdFitted': round(spectral.log_spectral_distance(original, rendered, args.rate), 3),
                'lsdInitial': round(spectral.log_spectral_distance(original, initial_audio, args.rate), 3)
            })
        groups[speaker] = group
        results['speakers'][speaker] = {
            'fitSeconds': round(fit_seconds, 1), 'finalTrainLoss': round(float(numpy.mean(history[-20:])), 4),
            'lossFitted': round(float(numpy.mean([row['lossFitted'] for row in rows])), 4),
            'lossInitial': round(float(numpy.mean([row['lossInitial'] for row in rows])), 4),
            'lsdFitted': round(float(numpy.mean([row['lsdFitted'] for row in rows])), 3),
            'lsdInitial': round(float(numpy.mean([row['lsdInitial'] for row in rows])), 3),
            'files': rows
        }

    if not args.no_listeners:
        paths = [path for group in groups.values() for files in group.values() for path in files]
        scores = listen(paths)
        for speaker, group in groups.items():
            summary = {}
            for key, files in group.items():
                summary[key] = {
                    'mos': round(float(numpy.mean([scores[path]['mos'] for path in files])), 3),
                    'similarity': round(float(numpy.mean([cosine(scores[path]['embedding'], scores[original]['embedding'])
                                                          for path, original in zip(files, group['original'])])), 3)
                }
            results['speakers'][speaker]['listeners'] = summary

    with open(os.path.join(args.out, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    summary = {speaker: {key: value for key, value in result.items() if key != 'files'} for speaker, result in results['speakers'].items()}
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
