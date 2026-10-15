# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Training - the network (nn/model.py) end to end through the differentiable synthesizer (nn/dsynth.py) against the
recordings (nn/data.py), DDSP-style: no hand-labeled formant targets.

    build/venv-nn/bin/python -B -m nn.train --speakers slt --name slt [--steps 30000] [--rate 16000]

Each step takes a batch of utterances, runs the model with the recorded durations (teacher forcing), renders a random
crop of each utterance's predicted parameter frames with the recorded F0 (F0 gets no useful gradient through the
synthesizer - nn/dsynth.py), and minimizes:

- the multi-resolution STFT loss (nn/spectral.py's spectral convergence and log magnitude, five FFT sizes) and a log-mel
  L1 loss, below --max-hz (ARCTIC has nothing above 8 kHz, and nn/dsynth.py's harmonics taper from 0.44 of the rate)
- the duration loss - log(d + 1) regression for the phones; for the boundary tokens a pause / no-pause classifier and the
  paused duration's regression; <bos> and the last token (the recording's edge silences) weighted lightly
- the F0 loss - L1 of log F0 against Praat's, voiced frames (unvoiced frames, interpolated, weighted lightly)
- weak guides that fade: L1 of log F1-F3 against Praat's (voiced, strong frames), the formant order, a smoothness
  penalty on the normalized trajectories, and the frication resonances kept below the data's band

Checkpoints and the log go to build/nn/runs/<name>/ (last.pt, best.pt - the lowest validation loss, log.jsonl).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy
import soxr
import torch

from . import discriminator, dsynth, params, spectral
from .data import ArcticData
from .model import BOS, FIRST_PHONE, PAD, Model, is_boundary, parameter_count


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get('NN_RUNS_DIR', os.path.join(ROOT, 'build', 'nn', 'runs'))

# Parameter priors per speaker (the analysis-by-synthesis fits' medians, nn/fit.py) - the head's starting point
PRIORS = {
    'slt': {'f0': 185, 'av': -30, 'rd': 1.0, 'shelf': 10, 'ah': -65, 'af': -50, 'f1': 600, 'f2': 1800, 'f3': 3000, 'f4': 4100,
            'f5': 5100, 'f6': 5900, 'b1': 150, 'b2': 180, 'b3': 200, 'b4': 220, 'b5': 300, 'b6': 320, 'fnp': 260, 'bnp': 100,
            'fnz': 290, 'bnz': 100, 'fp1': 2400, 'bp1': 600, 'ap1': -4, 'fp2': 5800, 'bp2': 700, 'ap2': -6, 'ab': -32,
            'fhp': 2000},
    'bdl': {'f0': 122, 'av': -30, 'rd': 1.0, 'shelf': 10, 'ah': -65, 'af': -45, 'f1': 500, 'f2': 1550, 'f3': 2700, 'f4': 3500,
            'f5': 4200, 'f6': 5300, 'b1': 90, 'b2': 110, 'b3': 150, 'b4': 160, 'b5': 200, 'b6': 500, 'fnp': 270, 'bnp': 100,
            'fnz': 270, 'bnz': 100, 'fp1': 2600, 'bp1': 450, 'ap1': -5, 'fp2': 5500, 'bp2': 550, 'ap2': -5, 'ab': -33,
            'fhp': 1800}
}

# Formant guide indexes (aux tracks f1-f3) and the parameter indexes
F_INDEX = [params.INDEX[f'f{number}'] for number in range(1, 7)]


#
# Data - every utterance of the speakers in memory, the audio at the training rate, and the 200 Hz F0 targets
#

class Utterances:
    """The utterances of a split, ready for training"""

    def __init__(self, data, split, speakers, rate, speaker_ids):
        self.items = []
        channels = data.index['channels']
        source_rate = data.index['sampleRate']
        for entry in data.entries(split, speakers):
            arrays = data.arrays(entry)
            audio = arrays['audio'].astype(numpy.float64)
            if rate != source_rate:
                audio = soxr.resample(audio, source_rate, rate)
            frame_count = entry['frameCount']
            audio = numpy.pad(audio, (0, max(0, frame_count * rate // 100 - len(audio))))[:frame_count * rate // 100]
            tracks = arrays['frames']
            settings = data.index['speakers'][entry['speaker']]
            self.items.append({
                'id': entry['id'], 'speaker': speaker_ids[entry['speaker']], 'tokens': arrays['tokens'],
                'durations': arrays['durations'], 'audio': audio.astype(numpy.float32), 'frameCount': frame_count,
                'f0': f0_track(tracks[:, channels.index('f0')], tracks[:, channels.index('voiced')] > 0.5, frame_count,
                               settings['pitchFloor'], settings['pitchCeiling']),
                'voiced': numpy.repeat(tracks[:, channels.index('voiced')] > 0.5, 2),
                'formants': numpy.stack([tracks[:, channels.index(f'f{number}')] for number in (1, 2, 3)], axis=-1),
                'formantWeight': ((tracks[:, channels.index('voiced')] > 0.5) & (tracks[:, channels.index('strength')] > 0.5) &
                                  (tracks[:, channels.index('f1')] > 0) & (tracks[:, channels.index('f3')] > 0)).astype(numpy.float32)
            })

    def __len__(self):
        return len(self.items)


# The 200 Hz F0 target (2F + 1 parameter frames, at t x 5 ms) of an utterance's 100 Hz Praat track (frame k centered at
# (k + 0.5) x 10 ms) - log-interpolated through the unvoiced frames, held at the ends, within the speaker's range
def f0_track(f0, voiced, frame_count, floor, ceiling):
    centers = (numpy.arange(frame_count) + 0.5) * 0.01
    times = numpy.arange(2 * frame_count + 1) * 0.005
    good = voiced & (f0 > 0)
    if good.sum() < 2:
        return numpy.full(len(times), math.sqrt(floor * ceiling), dtype=numpy.float32)
    log_f0 = numpy.interp(times, centers[good], numpy.log(f0[good]))
    return numpy.clip(numpy.exp(log_f0), floor, ceiling).astype(numpy.float32)


# Length-bucketed batches of indices (shuffled), at most batch_size utterances each
def make_batches(utterances, batch_size, rng):
    order = sorted(range(len(utterances)), key=lambda index: utterances.items[index]['frameCount'] + 40 * rng.random())
    batches = [order[ix:ix + batch_size] for ix in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


# Collate utterances into padded tensors on a device
def collate(utterances, indices, device):
    items = [utterances.items[index] for index in indices]
    token_count = max(len(item['tokens']) for item in items)
    frame_count = max(item['frameCount'] for item in items)
    batch = len(items)
    tokens = numpy.zeros((batch, token_count), dtype=numpy.int64)
    durations = numpy.zeros((batch, token_count), dtype=numpy.int64)
    f0 = numpy.zeros((batch, 2 * frame_count + 1), dtype=numpy.float32)
    voiced = numpy.zeros((batch, 2 * frame_count + 1), dtype=numpy.float32)
    formants = numpy.ones((batch, frame_count, 3), dtype=numpy.float32)
    formant_weight = numpy.zeros((batch, frame_count), dtype=numpy.float32)
    for ix, item in enumerate(items):
        count = len(item['tokens'])
        tokens[ix, :count] = item['tokens']
        durations[ix, :count] = item['durations']
        frames = item['frameCount']
        f0[ix, :2 * frames + 1] = item['f0']
        f0[ix, 2 * frames + 1:] = item['f0'][-1]
        voiced[ix, :2 * frames] = item['voiced']
        formants[ix, :frames] = numpy.maximum(item['formants'], 1)
        formant_weight[ix, :frames] = item['formantWeight']
    return {
        'tokens': torch.tensor(tokens, device=device), 'durations': torch.tensor(durations, device=device),
        'f0': torch.tensor(f0, device=device), 'voiced': torch.tensor(voiced, device=device),
        'formants': torch.tensor(formants, device=device), 'formantWeight': torch.tensor(formant_weight, device=device),
        'speaker': torch.tensor([item['speaker'] for item in items], device=device),
        'frameCounts': [item['frameCount'] for item in items], 'items': items
    }


#
# Losses
#

# The multi-resolution STFT loss below a frequency (spectral convergence + mean absolute log magnitude) plus a log-mel L1
# (each signal's STFT is taken once per size - the log-mel terms reuse the resolutions' spectra)
def spectral_loss(predicted, target, rate, max_hz, fft_sizes=(2048, 1024, 512, 256, 128), mel_weight=1.0, mel_eps=0.0):
    spectra = {}

    def spectra_of(fft_size):
        if fft_size not in spectra:
            spectra[fft_size] = (spectral.stft(predicted, fft_size, fft_size // 4), spectral.stft(target, fft_size, fft_size // 4))
        return spectra[fft_size]
    total = 0
    for fft_size in fft_sizes:
        bins = int(max_hz * fft_size / rate) + 1
        spectrum_predicted, spectrum_target = spectra_of(fft_size)
        magnitude_predicted = spectral.spectrum_magnitude(spectrum_predicted)[..., :bins]
        magnitude_target = spectral.spectrum_magnitude(spectrum_target)[..., :bins]
        convergence = torch.linalg.norm(magnitude_target - magnitude_predicted, dim=(1, 2)) / \
            (torch.linalg.norm(magnitude_target, dim=(1, 2)) + 1e-7)
        log_distance = (torch.log(magnitude_target) - torch.log(magnitude_predicted)).abs().mean(dim=(1, 2))
        total = total + convergence.mean() + log_distance.mean()
    total = total / len(fft_sizes)
    if mel_weight:
        mel_total = 0
        for fft_size in (1024, 256):
            bank = mel_bank(rate, fft_size, max_hz, predicted.device)
            spectrum_predicted, spectrum_target = spectra_of(fft_size)
            mel_predicted = torch.log(spectral.spectrum_magnitude(spectrum_predicted, eps=mel_eps) ** 2 @ bank.T + 1e-5)
            mel_target = torch.log(spectral.spectrum_magnitude(spectrum_target, eps=mel_eps) ** 2 @ bank.T + 1e-5)
            mel_total = mel_total + (mel_predicted - mel_target).abs().mean()
        total = total + mel_weight * mel_total / 2 * 0.25
    return total


_MEL_BANKS = {}


def mel_bank(rate, fft_size, max_hz, device):
    key = (rate, fft_size, max_hz, str(device))
    if key not in _MEL_BANKS:
        _MEL_BANKS[key] = torch.tensor(spectral.mel_filterbank(rate, fft_size, 64 if fft_size >= 512 else 32, high=max_hz),
                                       dtype=torch.float32, device=device)
    return _MEL_BANKS[key]


# The duration loss - phones: MSE of log(d + 1); boundary tokens: pause BCE and the paused duration's MSE; <bos> and the
# last token (the recording's edge silences) at edge_weight
def duration_loss(outputs, tokens, durations, edge_weight=0.1):
    log_duration, pause_logit, log_paused = outputs.unbind(-1)
    valid = tokens != PAD
    lengths = valid.sum(dim=1)
    target = torch.log1p(durations.float())
    weight = valid.float()
    weight = torch.where(tokens == BOS, weight * edge_weight, weight)
    last = torch.zeros_like(weight).scatter_(1, (lengths - 1)[:, None], 1.0)
    weight = torch.where(last > 0, weight * edge_weight, weight)
    phone = (tokens >= FIRST_PHONE).float() * weight
    boundary = (is_boundary(tokens) | (tokens == BOS)).float() * weight
    paused = (durations > 0).float()
    phone_loss = ((log_duration - target) ** 2 * phone).sum() / phone.sum().clamp(min=1)
    pause_loss = (torch.nn.functional.binary_cross_entropy_with_logits(pause_logit, paused, reduction='none') * boundary).sum() / \
        boundary.sum().clamp(min=1)
    paused_weight = boundary * paused
    paused_loss = ((log_paused - target) ** 2 * paused_weight).sum() / paused_weight.sum().clamp(min=1)
    return phone_loss, pause_loss, paused_loss


#
# Training
#

# A silent parameter frame (normalized and in parameter units)
def silent_values(device):
    values = torch.tensor([params.DEFAULTS[name] for name in params.NAMES], dtype=torch.float32, device=device)
    return values


# The silent frame and the mask of the levels that silence it (av, ah, af) on a device - made once per device
_SILENCE = {}


def _silence(device):
    key = str(device)
    if key not in _SILENCE:
        silence_mask = torch.zeros(params.COUNT, device=device)
        for name in ('av', 'ah', 'af'):
            silence_mask[params.INDEX[name]] = 1
        _SILENCE[key] = (silent_values(device), silence_mask)
    return _SILENCE[key]


# The model's parameter frames for a batch, ready to render - [B, 2F + 1, COUNT]: the end frame appended, frames past
# each utterance's end silent, and F0 replaced by the recorded F0 (teacher_f0) when given
def render_values(model, unit, batch, teacher_f0=True):
    values = model.parameters_from_unit(unit)
    values = torch.cat([values, values[:, -1:]], dim=1)
    total = values.shape[1]
    # (the batch's 'renderCounts' - 2 x its frame counts, on the device - if it has them: on MPS a blocking
    # host-to-device copy waits for the whole GPU queue)
    counts = batch.get('renderCounts')
    if counts is None:
        counts = torch.tensor([2 * count for count in batch['frameCounts']], device=values.device)
    past = (torch.arange(total, device=values.device)[None] > counts[:, None]).float()[..., None]
    silence, silence_mask = _silence(values.device)
    values = values * (1 - past * silence_mask) + silence * past * silence_mask
    if teacher_f0:
        f0 = batch['f0'][:, :total]
        values = torch.cat([values[..., :params.INDEX['f0']], f0[..., None], values[..., params.INDEX['f0'] + 1:]], dim=-1)
    return values


# Random crops of the batch's parameter frames and audio - (values [B, L + 1, COUNT], audio [B, L x rate / 200]); crops
# start on even frames (10 ms)
def crop(values, batch, crop_frames, rate, rng):
    per_frame = rate // params.FRAME_RATE
    crops_values = []
    crops_audio = []
    for ix, item in enumerate(batch['items']):
        available = 2 * item['frameCount']
        length = min(crop_frames, values.shape[1] - 1)
        start = 2 * int(rng.integers(0, max(0, available - length) // 2 + 1))
        start = min(start, values.shape[1] - 1 - length)
        crops_values.append(values[ix, start:start + length + 1])
        audio = item['audio'][start * per_frame:(start + length) * per_frame]
        crops_audio.append(numpy.pad(audio, (0, length * per_frame - len(audio))))
    target = torch.tensor(numpy.stack(crops_audio), device=values.device)
    return torch.stack(crops_values), target


# The loss terms of a batch - {name: tensor}
def batch_losses(model, synth, batch, args, rng, seed, guide_weight, train=True):
    outputs = model(batch['tokens'], batch['durations'], batch['speaker'])
    unit = outputs['unit']
    losses = {}
    losses['durPhone'], losses['durPause'], losses['durPaused'] = duration_loss(outputs['duration'], batch['tokens'], batch['durations'])

    # F0 regression (log), voiced frames weighted fully
    values = model.parameters_from_unit(unit)
    frames = values.shape[1]
    log_f0 = torch.log(values[..., params.INDEX['f0']])
    target_f0 = torch.log(batch['f0'][:, :frames])
    frame_mask = outputs['frameMask'].repeat_interleave(model.config['subframes'], dim=1)[..., 0]
    f0_weight = frame_mask * (batch['voiced'][:, :frames] + 0.1)
    losses['f0'] = ((log_f0 - target_f0).abs() * f0_weight).sum() / f0_weight.sum()

    # Formant guide - log F1-F3 at the decoder frames' centers (the odd subframes) against Praat's
    centers = values[:, 1::2]
    formant_weight = batch['formantWeight'][:, :centers.shape[1]]
    formant_error = (torch.log(centers[..., F_INDEX[:3]]) - torch.log(batch['formants'][:, :centers.shape[1]])).abs().sum(-1)
    losses['formant'] = (formant_error * formant_weight).sum() / formant_weight.sum().clamp(min=1)

    # Formant order, smoothness of the normalized trajectories, frication resonances below the band
    formants = values[..., F_INDEX]
    losses['order'] = (torch.relu(formants[..., :-1] + 150 - formants[..., 1:]).sum(-1) * frame_mask).sum() / frame_mask.sum() / 100
    smooth_mask = frame_mask[:, 1:] * frame_mask[:, :-1]
    losses['smooth'] = (((unit[:, 1:] - unit[:, :-1]) ** 2).mean(-1) * smooth_mask).sum() / smooth_mask.sum()
    band = sum(torch.relu(torch.log(values[..., params.INDEX[name]] / args.frication_max_hz)) for name in ('fp1', 'fp2'))
    losses['band'] = (band * frame_mask).sum() / frame_mask.sum()

    # The spectral loss of the rendered crops
    render = render_values(model, unit, batch, teacher_f0=True)
    crop_values, target = crop(render, batch, args.crop_frames, args.rate, rng)
    eq = model.eq(batch['speaker'])
    audio = synth(crop_values, seed=[seed + ix for ix in range(len(batch['items']))], eq=eq)
    losses['spectral'] = spectral_loss(audio, target[:, :audio.shape[1]], args.rate, args.max_hz)

    total = (losses['spectral'] + args.duration_weight * (losses['durPhone'] + losses['durPause'] + losses['durPaused']) +
             args.f0_weight * losses['f0'] + guide_weight * args.formant_weight * losses['formant'] + losses['order'] +
             args.smooth_weight * losses['smooth'] + losses['band'])
    losses['total'] = total
    return losses, audio, target[:, :audio.shape[1]]


# Validation - mean losses over the validation set (fixed crops and seeds), and frame-level metrics
@torch.no_grad()
def validate(model, synth, utterances, args, device):
    model.eval()
    rng = numpy.random.default_rng(1234)
    sums = {}
    count = 0
    f0_errors = []
    for indices in [list(range(ix, min(len(utterances), ix + args.batch))) for ix in range(0, len(utterances), args.batch)]:
        batch = collate(utterances, indices, device)
        losses, _, _ = batch_losses(model, synth, batch, args, rng, 99, 1.0, train=False)
        for key, value in losses.items():
            sums[key] = sums.get(key, 0) + float(value) * len(indices)
        count += len(indices)
        # F0 error in cents on voiced frames
        outputs = model(batch['tokens'], batch['durations'], batch['speaker'])
        values = model.parameters_from_unit(outputs['unit'])
        frames = values.shape[1]
        cents = 1200 * torch.log2(values[..., params.INDEX['f0']] / batch['f0'][:, :frames])
        voiced = batch['voiced'][:, :frames] > 0.5
        f0_errors.append(cents[voiced].abs().cpu().numpy())
    model.train()
    result = {key: value / count for key, value in sums.items()}
    result['f0Cents'] = float(numpy.mean(numpy.concatenate(f0_errors)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.train', description='Train the network through the synthesizer')
    parser.add_argument('--speakers', nargs='+', default=['slt'])
    parser.add_argument('--name', default=None)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--crop', type=float, default=2.0, help='the rendered crop (seconds)')
    parser.add_argument('--rate', type=int, default=16000)
    parser.add_argument('--max-hz', type=float, default=7000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--duration-weight', type=float, default=1.0)
    parser.add_argument('--f0-weight', type=float, default=2.0)
    parser.add_argument('--formant-weight', type=float, default=1.0)
    parser.add_argument('--guide-floor', type=float, default=0.1, help='the formant guide\'s final weight fraction')
    parser.add_argument('--guide-steps', type=float, default=0.3, help='the fraction of training over which the guide fades')
    parser.add_argument('--smooth-weight', type=float, default=0.05)
    parser.add_argument('--frication-max-hz', type=float, default=7500)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--config', default='{}', help='model config overrides (JSON)')
    parser.add_argument('--adversarial', type=float, default=0.0, help='the adversarial loss weight (0: no critic)')
    parser.add_argument('--feature-matching', type=float, default=2.0, help='the feature-matching loss weight (with a critic)')
    parser.add_argument('--adversarial-start', type=int, default=500, help='steps of the critic alone first')
    parser.add_argument('--critic-lr', type=float, default=2e-4)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--keep-every', type=int, default=10000, help='keep a checkpoint every this many steps')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', default=None, help='initialize the weights from a checkpoint')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(argv)
    args.crop_frames = int(round(args.crop * params.FRAME_RATE))
    name = args.name or '-'.join(args.speakers)
    directory = os.path.join(RUNS_DIR, name)
    os.makedirs(directory, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = numpy.random.default_rng(args.seed)
    device = torch.device(args.device)

    # Data
    data = ArcticData()
    speaker_ids = {speaker: ix for ix, speaker in enumerate(args.speakers)}
    start = time.time()
    train_set = Utterances(data, 'train', args.speakers, args.rate, speaker_ids)
    val_set = Utterances(data, 'val', args.speakers, args.rate, speaker_ids)
    print(f'data: {len(train_set)} train, {len(val_set)} val utterances ({time.time() - start:.1f} s)', file=sys.stderr)

    # The model, its priors (the first speaker's), and the optimizer
    config = {'speakers': len(args.speakers), **json.loads(args.config)}
    priors = PRIORS.get(args.speakers[0], {})
    model = Model(config, priors=priors, dropout=args.dropout).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device)
        model.load_state_dict(state['model'], strict=False)
    synth = dsynth.DSynth(args.rate).to(device)
    critic = discriminator.Critic(args.rate, args.max_hz).to(device) if args.adversarial > 0 else None
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, betas=(0.8, 0.99)) if critic else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    def learning_rate(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)

    step = 0
    best = math.inf
    if args.resume and os.path.exists(os.path.join(directory, 'last.pt')):
        state = torch.load(os.path.join(directory, 'last.pt'), map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        best = state.get('best', math.inf)
        if critic is not None and 'critic' in state:
            critic.load_state_dict(state['critic'])
            critic_optimizer.load_state_dict(state['criticOptimizer'])
        rng = numpy.random.default_rng(args.seed + step)
    print(f'model: {parameter_count(model)} parameters; {name}: {args.steps} steps from {step}', file=sys.stderr)

    def save(path, extra=None):
        if critic is not None:
            extra = {**(extra or {}), 'critic': critic.state_dict(), 'criticOptimizer': critic_optimizer.state_dict()}
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'step': step, 'best': best, 'config': model.config, 'priors': priors, 'speakers': args.speakers,
                    'args': vars(args), **(extra or {})}, path)

    log = open(os.path.join(directory, 'log.jsonl'), 'a', encoding='utf-8')  # pylint: disable=consider-using-with
    sums = {}
    sum_count = 0
    last_time = time.time()
    model.train()
    while step < args.steps:
        for indices in make_batches(train_set, args.batch, rng):
            if step >= args.steps:
                break
            batch = collate(train_set, indices, device)
            progress = step / args.steps
            guide = max(args.guide_floor, 1 - (1 - args.guide_floor) * progress / args.guide_steps)
            losses, audio, target = batch_losses(model, synth, batch, args, rng, int(rng.integers(1 << 30)), guide)
            if critic is not None:
                # The critic's step (on the rendering, detached), then the generator's adversarial and feature-matching
                # losses (after --adversarial-start steps of the critic alone)
                critic_loss = discriminator.discriminator_loss(critic(target), critic(audio.detach()))
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                losses['critic'] = critic_loss.detach()
                if step >= args.adversarial_start:
                    adversarial, matching = discriminator.generator_losses(critic(target), critic(audio))
                    losses['adversarial'] = adversarial.detach()
                    losses['matching'] = matching.detach()
                    losses['total'] = losses['total'] + args.adversarial * adversarial + args.feature_matching * matching
            optimizer.zero_grad()
            losses['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(norm):
                optimizer.step()
            scheduler.step()
            step += 1
            for key, value in losses.items():
                sums[key] = sums.get(key, 0) + float(value.detach())
            sum_count += 1
            if step % args.log_every == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                row = {'step': step, 'lr': scheduler.get_last_lr()[0], 'secondsPerStep': elapsed / args.log_every, 'guide': guide,
                       **{key: value / sum_count for key, value in sums.items()}}
                print(' '.join(f'{key} {value:.4g}' if isinstance(value, float) else f'{key} {value}' for key, value in row.items()),
                      file=sys.stderr, flush=True)
                log.write(json.dumps({'train': row}) + '\n')
                log.flush()
                sums, sum_count = {}, 0
            if step % args.val_every == 0 or step == args.steps:
                result = validate(model, synth, val_set, args, device)
                score = result['spectral'] + result['durPhone'] + result['f0']
                result['score'] = score
                print('  VAL ' + ' '.join(f'{key} {value:.4g}' for key, value in result.items()), file=sys.stderr, flush=True)
                log.write(json.dumps({'val': {'step': step, **result}}) + '\n')
                log.flush()
                if score < best:
                    best = score
                    save(os.path.join(directory, 'best.pt'), {'val': result})
                save(os.path.join(directory, 'last.pt'))
                if step % args.keep_every == 0:
                    save(os.path.join(directory, f'step-{step}.pt'))
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
