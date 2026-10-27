# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Training the multi-speaker, multi-style model (model.py's MultiModel) - end-to-end training through the
differentiable synthesizer (dsynth.py), over many speakers and the emotional styles:

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.trainmulti --name multi-a --recipe teacher.a
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.trainmulti --name multi-b --recipe teacher.b \
        --init build/nn/runs/multi-a/last.pt

--recipe takes the settings of a section of make train's recipe (data/nn/train.json - its keys are the flags), and
flags given override them.

The corpora (data.py's format, memory-mapped - the audio is read a crop at a time):

- the American reader pool, build/nn/data/multi/ (multi.py) - LibriTTS-R readers and CMU ARCTIC slt, bdl, clb,
  rms; each speaker's row in the speaker table is its data/nn/speakers.json id (index.json's speaker 'index')
- the emotional speech, build/nn/data/cremad/ (cremad.py) - CREMA-D's 91 actors, rows READERS + their index, and
  their six intended emotions: neutral is the zero style; the others are rows of the style table, scaled by the
  intensity (INTENSITY - low, medium, high; unspecified is medium)

Each batch is from one corpus (--cremad is the share of emotional batches). Training renders at --rate (24 kHz, the
corpora's rate) and compares below each utterance's own band: the recordings' bandwidths differ (ARCTIC and CREMA-D
stop at 8 kHz, LibriTTS-R at 12 kHz), so the rendering and the recording are both low-passed at 0.95 of the
recording's bandwidth before the losses, and the spectral losses stop at --max-hz (the synthesizer's harmonic taper).

The losses are losses.py's (the multi-resolution STFT and log-mel loss, durations, log F0, the fading formant guide,
formant order, smoothness, frication band) plus:

- the adversarial critic (discriminator.py, --adversarial and --feature-matching) on the reader batches only - the
  fix for the over-smoothing of a deterministic network trained by spectral regression (wide formant bandwidths, and
  noise filling the harmonics' valleys); the emotional recordings are noisy and echoey, so the critic never sees them
- the sonorant frication penalty (--sonorant-frication) - vowels, liquids, glides, and nasals have no frication source:
  in their frames the frication noise (af) above the voicing level less --sonorant-af-db is penalized, so the critic's
  push for aperiodic high-frequency energy goes to the aspiration (glottal-modulated, through the tract) rather than
  to flat static
- a smoothness penalty on each speaker's channel equalizer (its second differences), so blended curves stay smooth

Checkpoints and the log go to build/nn/runs/<name>/ (last.pt every --save-every steps, best.pt - the lowest reader
validation score, validated every --val-every steps from --val-from and at the end, step-<n>.pt every --keep-every steps,
log.jsonl).

The training step's options (optim.py; off by default - without them, training is unchanged):

- --lean-clip - the gradient clip and the AdamW step in a few kernel launches (the global norm of one concatenation,
  the clip as fused AdamW's grad_scale, one shared step counter) - the same up to rounding
- --optimizer muon - Muon for the hidden Linear matrices (batched, zero-padded Newton-Schulz), AdamW for the rest
- --ema DECAY - an exponential moving average of the weights, validated beside them (ema-best.pt, ema-last.pt)
- --schedule wsd --cooldown F --cooldown-shape linear|1-sqrt - warmup-stable-decay: constant, then the last F of the
  steps cooled down to the floor
- --crop-warmup STEPS:SECONDS - shorter rendered crops for the first steps
- --zero-init-proj - the residual output projections zero-initialized (a model from scratch)
"""

import argparse
import collections
import copy
import json
import math
import os
import queue
import sys
import threading
import time

import numpy
import soxr
import torch

from . import RECIPE_PATH, ROOT, discriminator, dsynth, optim, params
from .data import DATA_DIR, ArcticData
from .model import FIRST_PHONE, INTENSITY, STYLE_NAMES, MultiModel, core_parameter_count, parameter_count, save_checkpoint
from .losses import F_INDEX, PRIORS, duration_loss, f0_track, render_values, spectral_loss


# The data's sample rate
SOURCE_RATE = 24000

# The runs' directory (checkpoints and logs)
RUNS_DIR = os.environ.get('NN_RUNS_DIR', os.path.join(ROOT, 'build', 'nn', 'runs'))

# The most training steps the CPU runs ahead of the GPU (MPS)
IN_FLIGHT = 2


# The head's starting point - between the ARCTIC speakers' priors
MULTI_PRIORS = {name: (PRIORS['slt'][name] * PRIORS['bdl'][name]) ** 0.5 if PRIORS['slt'][name] > 0 else
                (PRIORS['slt'][name] + PRIORS['bdl'][name]) / 2 for name in PRIORS['slt']}

# The sonorant tokens - vowels, liquids, glides, nasals (vocab.json)
SONORANTS = ('L', 'M', 'N', 'NG', 'R', 'W', 'Y')


# The vocabulary's sonorant mask (a bool per token id)
def sonorant_mask(vocab):
    return numpy.array([index >= FIRST_PHONE and (token[-1:].isdigit() or token in SONORANTS) for index, token in enumerate(vocab)])


#
# Data
#

class Corpus:
    """A corpus's utterances of a split, ready for training - the audio stays memory-mapped"""

    def __init__(self, name, split, speaker_offset=0, max_items=None, seed=0):
        self.name = name
        self.data = ArcticData(os.path.join(DATA_DIR, name))
        speakers = self.data.index['speakers']
        channels = self.data.index['channels']
        entries = self.data.entries(split)
        if max_items and len(entries) > max_items:
            order = numpy.random.default_rng(seed).permutation(len(entries))[:max_items]
            entries = [entries[ix] for ix in sorted(order)]
        self.items = []
        for entry in entries:
            settings = speakers[entry['speaker']]
            tokens = slice(entry['tokenOffset'], entry['tokenOffset'] + entry['tokenCount'])
            frame_slice = slice(entry['frameOffset'], entry['frameOffset'] + entry['frameCount'])
            tracks = numpy.array(self.data.frames[frame_slice], dtype=numpy.float32)
            frame_count = entry['frameCount']
            floor, ceiling = settings['pitchFloor'], settings['pitchCeiling']
            emotion = entry.get('emotion')
            if emotion and emotion in settings.get('pitchRanges', {}):
                floor, ceiling = settings['pitchRanges'][emotion]
            voiced = tracks[:, channels.index('voiced')] > 0.5
            self.items.append({
                'id': entry['id'], 'speaker': speaker_offset + settings['index'],
                'emotion': STYLE_NAMES.index(emotion) if emotion in STYLE_NAMES else -1,
                'intensity': INTENSITY.get(entry.get('intensity'), 0.75) if emotion in STYLE_NAMES else 0.0,
                'tokens': numpy.array(self.data.tokens[tokens], dtype=numpy.int64),
                'durations': numpy.array(self.data.durations[tokens], dtype=numpy.int64),
                'frameCount': frame_count, 'sampleOffset': entry['sampleOffset'], 'sampleCount': entry['sampleCount'],
                'bandwidth': settings.get('bandwidth', 8000),
                'f0': f0_track(tracks[:, channels.index('f0')], voiced, frame_count, floor, ceiling),
                'voiced': numpy.repeat(voiced, 2),
                'formants': numpy.stack([tracks[:, channels.index(f'f{number}')] for number in (1, 2, 3)], axis=-1),
                'formantWeight': (voiced & (tracks[:, channels.index('strength')] > 0.5) & (tracks[:, channels.index('f1')] > 0) &
                                  (tracks[:, channels.index('f3')] > 0)).astype(numpy.float32)
            })

    def __len__(self):
        return len(self.items)

    # An item's audio samples [start, end) at the source rate (float32, zero-padded past its end)
    def audio(self, item, start, end):
        count = item['sampleCount']
        offset = item['sampleOffset']
        samples = numpy.array(self.data.audio[offset + min(start, count):offset + min(end, count)], dtype=numpy.float32) / 32768.0
        return numpy.pad(samples, (0, end - start - len(samples)))


# Length-bucketed batches of indices (shuffled), at most batch_size utterances and max_frames padded frames each
def make_batches(corpus, batch_size, max_frames, rng):
    order = sorted(range(len(corpus)), key=lambda index: corpus.items[index]['frameCount'] + 40 * rng.random())
    batches = []
    current = []
    for index in order:
        longest = max(corpus.items[ix]['frameCount'] for ix in current + [index])
        if current and (len(current) >= batch_size or longest * (len(current) + 1) > max_frames):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


# An endless stream of a corpus's batches, reshuffled each epoch
def batch_stream(corpus, batch_size, max_frames, rng):
    while True:
        yield from make_batches(corpus, batch_size, max_frames, rng)


# The smallest length of a geometric ladder (ratio, multiples of quantum) at least a length - the padded length of
# fixed-shape batches (ratio 0: the length itself)
def bucket_length(length, ratio, quantum=8):
    if not ratio:
        return length
    bucket = quantum
    while bucket < length:
        bucket = max(bucket + quantum, int(math.ceil(bucket * ratio / quantum)) * quantum)
    return bucket


# Collate utterances into padded host (CPU) tensors - a batch for upload(); with bucket_ratio, the token and frame
# counts are padded up to a few fixed lengths (bucket_length), so the GPU sees few distinct shapes (on MPS each new
# shape compiles its graphs) - the padding is masked, so only the dropout masks' draws change
def collate_host(corpus, indices, bucket_ratio=0):
    items = [corpus.items[index] for index in indices]
    token_count = bucket_length(max(len(item['tokens']) for item in items), bucket_ratio)
    frame_count = max(item['frameCount'] for item in items)
    if bucket_ratio:
        frame_count = bucket_length(max(frame_count, max(int(item['durations'].sum()) for item in items)), bucket_ratio)
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
    host = {
        'tokens': torch.from_numpy(tokens), 'durations': torch.from_numpy(durations), 'f0': torch.from_numpy(f0),
        'voiced': torch.from_numpy(voiced), 'formants': torch.from_numpy(formants), 'formantWeight': torch.from_numpy(formant_weight),
        'speaker': torch.tensor([item['speaker'] for item in items]), 'emotion': torch.tensor([item['emotion'] for item in items]),
        'intensity': torch.tensor([item['intensity'] for item in items], dtype=torch.float32),
        'renderCounts': torch.tensor([2 * item['frameCount'] for item in items])
    }
    # The model's frame count (model.py regulate - the longest utterance's total duration, at least 1; or the bucket)
    model_frames = frame_count if bucket_ratio else max(1, int(durations.sum(axis=1).max()))
    return {'corpus': corpus, 'host': host, 'frameCount': model_frames, 'frameCounts': [item['frameCount'] for item in items],
            'items': items}


# A host batch's tensors uploaded to a device - non-blocking (on MPS a blocking host-to-device copy waits for the whole
# GPU queue); the batch keeps its host tensors, which must live until the copies are done (a synchronization)
def upload(host_batch, device):
    batch = dict(host_batch)
    batch.update({key: value.to(device, non_blocking=True) for key, value in host_batch['host'].items()})
    return batch


# Collate utterances into padded tensors on a device
def collate(corpus, indices, device):
    return upload(collate_host(corpus, indices), device)


# The random crops' starts of a batch's items (parameter frames, even - 10 ms) and their length, for parameter frames
# [B, total, COUNT] - crop's draws from rng
def crop_starts(items, total, crop_frames, rng):
    length = min(crop_frames, total - 1)
    starts = []
    for item in items:
        available = 2 * item['frameCount']
        start = 2 * int(rng.integers(0, max(0, available - length) // 2 + 1))
        starts.append(min(start, total - 1 - length))
    return starts, length


# The recordings' crops [B, L x rate / 200] (numpy float32) at crop starts
def crop_audio(corpus, items, starts, length, rate):
    per_frame = SOURCE_RATE // params.FRAME_RATE
    crops_audio = []
    for item, start in zip(items, starts):
        audio = corpus.audio(item, start * per_frame, (start + length) * per_frame)
        if rate != SOURCE_RATE:
            audio = soxr.resample(audio, SOURCE_RATE, rate)
            audio = numpy.pad(audio, (0, max(0, length * rate // params.FRAME_RATE - len(audio))))[:length * rate // params.FRAME_RATE]
        crops_audio.append(audio)
    return numpy.stack(crops_audio)


# Random crops of the batch's parameter frames and recordings - (values [B, L + 1, COUNT], audio [B, L x rate / 200]);
# crops start on even frames (10 ms)
def crop(values, batch, crop_frames, rate, rng):
    starts, length = crop_starts(batch['items'], values.shape[1], crop_frames, rng)
    crops_values = torch.stack([values[ix, start:start + length + 1] for ix, start in enumerate(starts)])
    return crops_values, torch.tensor(crop_audio(batch['corpus'], batch['items'], starts, length, rate), device=values.device)


# The spectral loss compiled - on MPS its elementwise work fused into a few kernels (8-13% of a stage-A step; the
# gradient within rounding of the eager loss's, 143 dB)
compiled_spectral_loss = torch.compile(spectral_loss)


# The spectral loss of --crop-warmup's short crops (its own compiled function - one fixed crop length each)
def short_crop_spectral_loss(audio, target, rate, max_hz, mel_eps):
    return spectral_loss(audio, target, rate, max_hz, mel_eps=mel_eps)


compiled_short_crop_spectral_loss = torch.compile(short_crop_spectral_loss)


# The rendered crop's length (parameter frames - 200 a second) of a training step - --crop-warmup's short crop for its
# first steps, else --crop's
def step_crop_frames(args, step):
    warmup = getattr(args, 'crop_warmup', None)
    if warmup:
        steps, seconds = warmup.split(':')
        if step < int(steps):
            return int(round(float(seconds) * params.FRAME_RATE))
    return args.crop_frames


# Low-pass audio [B, N] at a cutoff per item (Hz) - zeroing the FFT bins above it (so the magnitude spectra of the
# losses and the critic take a small epsilon - a magnitude of exactly zero has an infinite gradient); cutoff_tensor is
# the cutoffs as a float32 tensor on the audio's device, if the caller has it
def band_limit(audio, cutoffs, rate, cutoff_tensor=None):
    if all(cutoff >= rate / 2 for cutoff in cutoffs):
        return audio
    spectrum = torch.fft.rfft(audio)
    frequency = torch.arange(spectrum.shape[-1], device=audio.device) * rate / audio.shape[-1]
    if cutoff_tensor is None:
        cutoff_tensor = torch.tensor(cutoffs, device=audio.device, dtype=torch.float32)
    mask = (frequency[None] <= cutoff_tensor[:, None]).float()
    return torch.fft.irfft(spectrum * mask, n=audio.shape[-1])


class Prefetcher:
    """The training steps' data made ahead on a thread, so a step's CPU work overlaps the GPU's: each step's corpus
    choice, batch, synthesis seed, and crop starts - the same draws from rng in the same order as the training loop
    made them - the collated batch, the recordings' crops, and the synthesizer's cycle plans of the recorded F0 (dsynth.py
    plan_cycles - the rendered frames' F0 is the recording's, and their jitter the model's constant). next() returns
    the steps' data in order."""

    def __init__(self, corpora, streams, args, rng, synth, jitter, depth=3, step=0):
        self.corpora = corpora
        self.step = step
        self.streams = streams
        self.args = args
        self.rng = rng
        self.synth = synth
        self.jitter = jitter
        self.queue = queue.Queue(maxsize=depth)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='prefetch', daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopping.is_set():
                prepared = self._prepare()
                while not self.stopping.is_set():
                    try:
                        self.queue.put(prepared, timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except BaseException as error:  # pylint: disable=broad-except
            self.queue.put(error)

    # One step's data (the training loop's draws from rng, in order: the corpus, the batch, the seed, the crops)
    def _prepare(self):
        args = self.args
        name = 'cremad' if 'cremad' in self.corpora and self.rng.random() < args.cremad else 'multi'
        batch = collate_host(self.corpora[name], next(self.streams[name]), getattr(args, 'bucket_ratio', 0))
        seed = int(self.rng.integers(1 << 30))
        total = 2 * batch['frameCount'] + 1
        starts, length = crop_starts(batch['items'], total, step_crop_frames(args, self.step), self.rng)
        self.step += 1
        host = batch['host']
        host['target'] = torch.from_numpy(crop_audio(batch['corpus'], batch['items'], starts, length, args.rate))
        cutoffs = [0.95 * item['bandwidth'] for item in batch['items']]
        host['cutoffs'] = torch.tensor(cutoffs, dtype=torch.float32)
        seeds = [seed + ix for ix in range(len(batch['items']))]
        plan = None
        f0 = host['f0'].numpy()[:, :total]
        if f0.shape[1] == total and dsynth.sample_count(length + 1, args.rate) > 0:
            f0_frames = numpy.stack([f0[ix, start:start + length + 1] for ix, start in enumerate(starts)]).astype(numpy.float64)
            plan = self.synth.plan_cycles(f0_frames, numpy.full(f0_frames.shape, self.jitter), seeds)
            plan['host'] = self.synth.plan_host_tensors(plan)
        return {'corpusName': name, 'batch': batch, 'seed': seed, 'starts': starts, 'length': length, 'cutoffs': cutoffs,
                'plan': plan}

    def next(self):
        prepared = self.queue.get()
        if isinstance(prepared, BaseException):
            raise prepared
        return prepared

    def close(self):
        self.stopping.set()
        while self.thread.is_alive():
            try:
                self.queue.get(timeout=0.1)
            except queue.Empty:
                pass
        self.thread.join()


#
# Losses
#

# Voice-quality statistics of parameter frames in strongly voiced frames (within 15 dB of the utterance's peak voicing) -
# median b1-b3 (Hz), ah and af relative to av (dB) - for monitoring the over-smoothing (0-dim tensors on the device, not
# read - no synchronization; NaN if there are no strong frames)
@torch.no_grad()
def quality_stats(values, frame_mask):
    av = values[..., params.INDEX['av']]
    peak = torch.where(frame_mask > 0, av, torch.full_like(av, params.FLOOR_DB)).max(dim=1, keepdim=True).values
    strong = (frame_mask > 0) & (av > peak - 15)

    # The medians of the strong frames' values - the NaN-ignoring medians of the values with the others NaN, the same
    # (lower) medians as of the selected values, without the boolean selection's synchronization (the five at once)
    names = ('b1', 'b2', 'b3', 'ahRel', 'afRel')
    columns = torch.stack([values[..., params.INDEX['b1']], values[..., params.INDEX['b2']], values[..., params.INDEX['b3']],
                           values[..., params.INDEX['ah']] - av, values[..., params.INDEX['af']] - av])
    medians = torch.where(strong, columns, torch.full_like(columns, math.nan)).reshape(len(names), -1).nanmedian(dim=-1).values
    return dict(zip(names, medians.unbind()))


# The loss terms of a batch - ({name: tensor}, the rendered crops, the recorded crops), both band-limited; prepared is
# the step's data made ahead (Prefetcher) - the crops, the recordings', and the cycle plan - else they're made here
def batch_losses(model, synth, batch, args, rng, seed, guide_weight, sonorants, prepared=None, with_values=False):
    outputs = model(batch['tokens'], batch['durations'], batch['speaker'], batch['emotion'], batch['intensity'],
                    speaker_vector=batch.get('speakerVector'), frame_count=batch.get('frameCount'))
    unit = outputs['unit']
    losses = {}
    losses['durPhone'], losses['durPause'], losses['durPaused'] = duration_loss(outputs['duration'], batch['tokens'], batch['durations'])

    # F0 regression (log), voiced frames weighted fully
    values = model.parameters_from_unit(unit)
    frames = values.shape[1]
    frame_mask = outputs['frameMask'].repeat_interleave(model.config['subframes'], dim=1)[..., 0]
    log_f0 = torch.log(values[..., params.INDEX['f0']])
    target_f0 = torch.log(batch['f0'][:, :frames])
    f0_weight = frame_mask * (batch['voiced'][:, :frames] + 0.1)
    losses['f0'] = ((log_f0 - target_f0).abs() * f0_weight).sum() / f0_weight.sum()

    # Formant guide - log F1-F3 at the decoder frames' centers against Praat's
    centers = values[:, 1::2]
    formant_weight = batch['formantWeight'][:, :centers.shape[1]]
    formant_error = (torch.log(centers[..., F_INDEX[:3]]) - torch.log(batch['formants'][:, :centers.shape[1]])).abs().sum(-1)
    losses['formant'] = (formant_error * formant_weight).sum() / formant_weight.sum().clamp(min=1)

    # Formant order, smoothness, frication resonances below the band
    formants = values[..., F_INDEX]
    losses['order'] = (torch.relu(formants[..., :-1] + 150 - formants[..., 1:]).sum(-1) * frame_mask).sum() / frame_mask.sum() / 100
    smooth_mask = frame_mask[:, 1:] * frame_mask[:, :-1]
    losses['smooth'] = (((unit[:, 1:] - unit[:, :-1]) ** 2).mean(-1) * smooth_mask).sum() / smooth_mask.sum()
    band = sum(torch.relu(torch.log(values[..., params.INDEX[name]] / args.frication_max_hz)) for name in ('fp1', 'fp2'))
    losses['band'] = (band * frame_mask).sum() / frame_mask.sum()

    # The sonorant frication penalty - af above av - sonorant_af_db (dB / 10) in the sonorant tokens' middle frames
    token_index, position = frame_tokens(batch['durations'], frames // model.config['subframes'])
    token_at = torch.gather(batch['tokens'], 1, token_index)
    sonorant = sonorants[token_at] & (position > 0.2) & (position < 0.8)
    sonorant = sonorant.repeat_interleave(model.config['subframes'], dim=1).float() * frame_mask
    excess = torch.relu(values[..., params.INDEX['af']] - values[..., params.INDEX['av']] + args.sonorant_af_db) / 10
    losses['sonorantAf'] = (excess * sonorant).sum() / sonorant.sum().clamp(min=1)

    # The channel equalizers' smoothness
    eq = model.eq(batch['speaker'])
    losses['eqSmooth'] = ((eq[:, 2:] - 2 * eq[:, 1:-1] + eq[:, :-2]) ** 2).mean() / 100

    # The spectral loss of the rendered crops, both band-limited to the recording's band
    render = render_values(model, unit, batch, teacher_f0=True)
    if prepared is None:
        crop_values, target = crop(render, batch, args.crop_frames, args.rate, rng)
        plan = None
    else:
        starts, length = prepared['starts'], prepared['length']
        target = batch['target']
        plan = prepared['plan']
        crop_values = torch.stack([render[ix, start:start + length + 1] for ix, start in enumerate(starts)])
    # (the rendered F0 is the recording's - render_values's teacher F0 - so the synthesizer needn't differentiate in it)
    audio = synth(crop_values, seed=[seed + ix for ix in range(len(batch['items']))], eq=eq, plan=plan, constant_f0=True)
    target = target[:, :audio.shape[1]]
    cutoffs = [0.95 * item['bandwidth'] for item in batch['items']]
    audio = band_limit(audio, cutoffs, args.rate, batch.get('cutoffs'))
    target = band_limit(target, cutoffs, args.rate, batch.get('cutoffs'))
    full_crop = audio.shape[1] == round(args.crop * args.rate)
    short_crop = bool(getattr(args, 'crop_warmup', None)) and audio.shape[1] == round(float(args.crop_warmup.split(':')[1]) * args.rate)
    if audio.device.type == 'mps' and (full_crop or short_crop):  # pragma: no cover - the Apple GPU only
        # (the full-length crops - nearly every step - through the compiled loss, the batch size a dynamic dimension;
        # --crop-warmup's short crops through their own compiled loss)
        # (torch's API for a dynamic dimension is torch._dynamo's)
        torch._dynamo.mark_dynamic(audio, 0)  # pylint: disable=protected-access
        torch._dynamo.mark_dynamic(target, 0)  # pylint: disable=protected-access
        if full_crop:
            losses['spectral'] = compiled_spectral_loss(audio, target, args.rate, args.max_hz, mel_eps=1e-9)
        else:
            losses['spectral'] = compiled_short_crop_spectral_loss(audio, target, args.rate, args.max_hz, 1e-9)
    else:
        losses['spectral'] = spectral_loss(audio, target, args.rate, args.max_hz, mel_eps=1e-9)

    total = (losses['spectral'] + args.duration_weight * (losses['durPhone'] + losses['durPause'] + losses['durPaused']) +
             args.f0_weight * losses['f0'] + guide_weight * args.formant_weight * losses['formant'] + losses['order'] +
             args.smooth_weight * losses['smooth'] + losses['band'] + args.sonorant_frication * losses['sonorantAf'] +
             losses['eqSmooth'])
    losses['total'] = total
    if with_values:
        return losses, audio, target, quality_stats(values.detach(), frame_mask), values
    return losses, audio, target, quality_stats(values.detach(), frame_mask)


# Each 10 ms frame's token index [B, F] and its relative position within the token [B, F]
def frame_tokens(durations, frame_count):
    ends = torch.cumsum(durations, dim=1)
    frames = torch.arange(frame_count, device=durations.device)
    token_index = torch.searchsorted(ends, frames[None].expand(durations.shape[0], -1).contiguous(), right=True)
    token_index = token_index.clamp(max=durations.shape[1] - 1)
    duration = torch.gather(durations, 1, token_index).float().clamp(min=1)
    start = torch.gather(ends - durations, 1, token_index).float()
    return token_index, (frames[None].float() - start + 0.5) / duration


# The validation losses and statistics of a corpus (no gradients; the model in eval mode) - mean losses and quality
# statistics (fixed crops and seeds), and the F0 error
@torch.no_grad()
def validate(model, synth, corpus, args, device, sonorants):
    model.eval()
    rng = numpy.random.default_rng(1234)
    sums = {}
    count = 0
    f0_errors = []
    order = sorted(range(len(corpus)), key=lambda index: corpus.items[index]['frameCount'])
    for start in range(0, len(order), args.batch):
        indices = order[start:start + args.batch]
        batch = collate(corpus, indices, device)
        losses, _, _, stats, values = batch_losses(model, synth, batch, args, rng, 99, 1.0, sonorants, with_values=True)
        stats = {key: value for key, value in ((key, float(value)) for key, value in stats.items()) if not math.isnan(value)}
        for key, value in {**losses, **stats}.items():
            sums[key] = sums.get(key, 0) + float(value) * len(indices)
        count += len(indices)
        frames = values.shape[1]
        cents = 1200 * torch.log2(values[..., params.INDEX['f0']] / batch['f0'][:, :frames])
        voiced = batch['voiced'][:, :frames] > 0.5
        f0_errors.append(cents[voiced].abs().cpu().numpy())
    model.train()
    result = {key: value / count for key, value in sums.items()}
    result['f0Cents'] = float(numpy.mean(numpy.concatenate(f0_errors)))
    result['score'] = result['spectral'] + result['durPhone'] + result['f0']
    return result


# The settings of a section of the recipe (data/nn/train.json; "teacher.a" - a dotted path) as argparse defaults
def recipe_defaults(section, path=RECIPE_PATH):
    with open(path, encoding='utf-8') as file:
        settings = json.load(file)
    for key in section.split('.'):
        settings = settings[key]
    return {key.replace('-', '_'): value for key, value in settings.items() if key != 'description'}


# The command line's parser
def argument_parser():
    parser = argparse.ArgumentParser(prog='python -m picovox_train.trainmulti', description='Train the multi-speaker, multi-style model')
    # (the training settings' defaults are the recipe's teacher.defaults)
    parser.add_argument('--name', required=True)
    parser.add_argument('--recipe', help='the settings of a section of the recipe (teacher.a), flags overriding them')
    parser.add_argument('--recipe-file', default=RECIPE_PATH, help='the recipe (default data/nn/train.json)')
    parser.add_argument('--steps', type=int, help='the learning rate schedule\'s steps')
    parser.add_argument('--stop', type=int, default=None, help='stop after this many steps (default: --steps)')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--max-frames', type=int, help='the most padded 10 ms frames in a batch')
    parser.add_argument('--crop', type=float, help='the rendered crop (seconds)')
    parser.add_argument('--bucket-ratio', type=float,
                        help='pad token and frame counts up to a geometric ladder of this ratio (0: no fixed shapes)')
    parser.add_argument('--rate', type=int)
    parser.add_argument('--max-hz', type=float)
    parser.add_argument('--cremad', type=float, help='the share of batches from the emotional speech')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--duration-weight', type=float)
    parser.add_argument('--f0-weight', type=float)
    parser.add_argument('--formant-weight', type=float)
    parser.add_argument('--guide-floor', type=float)
    parser.add_argument('--guide-steps', type=float)
    parser.add_argument('--smooth-weight', type=float)
    parser.add_argument('--frication-max-hz', type=float)
    parser.add_argument('--sonorant-frication', type=float, help='the sonorant frication penalty weight')
    parser.add_argument('--sonorant-af-db', type=float, help='frication allowed this far below the voicing')
    parser.add_argument('--speaker-noise', type=float, help='Gaussian noise added to the speaker embeddings')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--config', default='{}', help='model config overrides (JSON)')
    parser.add_argument('--adversarial', type=float, help='the adversarial loss weight (0: no critic)')
    parser.add_argument('--feature-matching', type=float)
    parser.add_argument('--adversarial-start', type=int, help='steps of the critic alone first')
    parser.add_argument('--critic-lr', type=float)
    parser.add_argument('--critic-precision', choices=['float32', 'bfloat16', 'float16'],
                        help="the critic's convolutions' precision")
    parser.add_argument('--critic-channels', type=int, default=32, help="the critic's convolutions' width")
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--val-from', type=int, default=0, help='validate (every --val-every steps) from this step on')
    parser.add_argument('--save-every', type=int, default=1000, help='save last.pt every this many steps (and at validations)')
    parser.add_argument('--val-count', type=int, help='reader validation utterances')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--keep-every', type=int, default=5000, help='keep step-<n>.pt every this many steps (0: none)')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--init', default=None, help='initialize the weights from a checkpoint')
    parser.add_argument('--init-critic', default=None, help='initialize the critic from a checkpoint\'s (with --adversarial)')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lean-clip', action='store_true', help='the lean clip and AdamW step (optim.py)')
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw', help='muon: Muon for the hidden matrices')
    parser.add_argument('--ema', type=float, default=0.0, help='an EMA of the weights at this decay, validated beside them')
    parser.add_argument('--schedule', choices=['cosine', 'wsd'], default='cosine', help='the learning rate schedule')
    parser.add_argument('--cooldown', type=float, default=0.0, help="wsd: the cooldown's share of --steps")
    parser.add_argument('--cooldown-shape', choices=['linear', '1-sqrt'], default='linear')
    parser.add_argument('--crop-warmup', default=None, help='STEPS:SECONDS - shorter rendered crops for the first steps')
    parser.add_argument('--zero-init-proj', action='store_true', help='zero-initialize the residual output projections')
    parser.set_defaults(**recipe_defaults('teacher.defaults'))
    return parser


# The default settings (the flags' defaults) with overrides - an argparse namespace, as main's args
def default_settings(**overrides):
    settings = argument_parser().parse_args(['--name', 'defaults'])
    for key, value in overrides.items():
        setattr(settings, key, value)
    settings.crop_frames = int(round(settings.crop * params.FRAME_RATE))
    return settings


def main(argv=None):
    parser = argument_parser()
    known = parser.parse_known_args(argv)[0]
    parser.set_defaults(**recipe_defaults('teacher.defaults', known.recipe_file))
    if known.recipe:
        parser.set_defaults(**recipe_defaults(known.recipe, known.recipe_file))
    args = parser.parse_args(argv)
    args.crop_frames = int(round(args.crop * params.FRAME_RATE))
    stop = min(args.stop or args.steps, args.steps)
    directory = os.path.join(RUNS_DIR, args.name)
    os.makedirs(directory, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = numpy.random.default_rng(args.seed)
    device = torch.device(args.device)

    # Data - the readers, and the emotional speech after them in the speaker table
    start = time.time()
    readers = Corpus('multi', 'train')
    reader_count = 1 + max(settings['index'] for settings in readers.data.index['speakers'].values())
    val_sets = {'multi': Corpus('multi', 'val', max_items=args.val_count)}
    cremad = None
    if args.cremad > 0:
        cremad = Corpus('cremad', 'train', speaker_offset=reader_count)
        val_sets['cremad'] = Corpus('cremad', 'val', speaker_offset=reader_count)
    cremad_index = ArcticData(os.path.join(DATA_DIR, 'cremad')).index
    speaker_keys = [None] * (reader_count + len(cremad_index['speakers']))
    for key, settings in readers.data.index['speakers'].items():
        speaker_keys[settings['index']] = key
    for key, settings in cremad_index['speakers'].items():
        speaker_keys[reader_count + settings['index']] = key
    print(f'data: {len(readers)} reader utterances, {len(cremad) if cremad else 0} emotional, val ' +
          ', '.join(f'{name} {len(corpus)}' for name, corpus in val_sets.items()) + f' ({time.time() - start:.1f} s)',
          file=sys.stderr, flush=True)
    sonorants = torch.tensor(sonorant_mask(readers.data.index['vocab']), device=device)

    # The model, the critic, and the optimizers
    config = {'speakers': len(speaker_keys), 'vocab': len(readers.data.index['vocab']), **json.loads(args.config)}
    model = MultiModel(config, priors=MULTI_PRIORS, dropout=args.dropout).to(device)
    if args.zero_init_proj:
        optim.zero_init_projections(model)
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state['model'], strict=False)
        print(f'init from {args.init}: missing {missing}, unexpected {unexpected}', file=sys.stderr)
    synth = dsynth.DSynth(args.rate).to(device)
    critic = None
    if args.adversarial > 0:
        critic_dtype = None if args.critic_precision == 'float32' else getattr(torch, args.critic_precision)
        critic = discriminator.Critic(args.rate, args.max_hz, eps=1e-9, dtype=critic_dtype, channels=args.critic_channels).to(device)
        if args.init_critic:
            critic.load_state_dict(torch.load(args.init_critic, map_location=device, weights_only=False)['critic'])
            print(f'critic init from {args.init_critic}', file=sys.stderr)
    # (on the GPU the fused AdamW - one kernel, not a loop over the parameters - which also skips a step with a
    # non-finite gradient norm on the device, so the step needn't read the norm)
    fused = device.type in ('mps', 'cuda')
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, betas=(0.8, 0.99),
                                         fused=fused or None) if critic else None
    # (the lean step - and Muon, which takes its clip - optim.py)
    lean = args.lean_clip or args.optimizer == 'muon'
    if lean:
        optimizer = optim.LeanOptimizer(optim.param_groups(model, muon=args.optimizer == 'muon'), lr=args.lr,
                                        weight_decay=args.weight_decay, betas=(0.9, 0.98))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98),
                                      fused=fused or None)

    def learning_rate(step):
        return optim.lr_factor(step, args.steps, args.warmup, args.schedule, args.cooldown, args.cooldown_shape)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    # (the EMA of the weights, and a model to validate it in)
    ema = optim.FlatEMA(model.parameters(), args.ema) if args.ema else None
    ema_model = copy.deepcopy(model) if ema else None
    ema_best = math.inf

    step = 0
    best = math.inf
    if args.resume and os.path.exists(os.path.join(directory, 'last.pt')):
        state = torch.load(os.path.join(directory, 'last.pt'), map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        # (a checkpoint saved by the unfused AdamW resumes fused: its param groups say fused, so loading moves the
        # step counts to the device as the fused step needs)
        if not lean:
            for group in state['optimizer']['param_groups']:
                group['fused'] = fused or None
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        best = state.get('best', math.inf)
        if ema is not None and 'ema' in state:
            ema.load_state_dict(state['ema'])
            ema_best = state.get('emaBest', math.inf)
        elif ema is not None:
            # (a run without the EMA resumed with it - the EMA starts from the resumed weights)
            ema = optim.FlatEMA(model.parameters(), args.ema)
        if critic is not None and 'critic' in state:
            critic.load_state_dict(state['critic'])
            for group in state['criticOptimizer']['param_groups']:
                group['fused'] = fused or None
            critic_optimizer.load_state_dict(state['criticOptimizer'])
        rng = numpy.random.default_rng(args.seed + step)
    core = core_parameter_count(model)
    print(f'model: {parameter_count(model)} parameters ({core} without the speaker and equalizer tables); {args.name}: '
          f'{args.steps} steps from {step}', file=sys.stderr, flush=True)

    def save(path, extra=None, weights=None):
        if critic is not None:
            extra = {**(extra or {}), 'critic': critic.state_dict(), 'criticOptimizer': critic_optimizer.state_dict()}
        if ema is not None:
            extra = {**(extra or {}), 'ema': ema.state_dict(), 'emaBest': ema_best}
        save_checkpoint({'model': (weights or model).state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'step': step, 'best': best, 'config': model.config,
                         'priors': MULTI_PRIORS, 'speakers': speaker_keys, 'readerCount': reader_count,
                         'styles': list(STYLE_NAMES), 'args': vars(args), **(extra or {})}, path)

    streams = {'multi': batch_stream(readers, args.batch, args.max_frames, rng)}
    corpora = {'multi': readers}
    if cremad is not None:
        streams['cremad'] = batch_stream(cremad, args.batch, args.max_frames, rng)
        corpora['cremad'] = cremad
    log = open(os.path.join(directory, 'log.jsonl'), 'a', encoding='utf-8')  # pylint: disable=consider-using-with
    sums = {}
    sum_counts = {}
    # The steps' loss and statistics tensors since the last log row - read only when it's written (each read
    # synchronizes): (key prefix, loss keys, statistics keys, the values [losses + statistics])
    pending = []
    last_time = time.time()
    model.train()
    # The steps' data, made ahead on a thread - all the draws from rng are its, in the order the steps made them
    jitter = float(model.fixed[params.INDEX['jitter']].cpu())
    prefetcher = Prefetcher(corpora, streams, args, rng, synth, jitter, step=step)
    # The recent steps' data - their host tensors are the sources of non-blocking uploads, so each is kept until the
    # GPU is past its step (an event after it): on MPS the steps don't read the device (the synthesizer finishes its
    # plans made ahead there), so the CPU runs up to IN_FLIGHT steps ahead of the GPU
    in_flight = collections.deque()
    while step < stop:
        prepared = prefetcher.next()
        corpus = prepared['corpusName']
        batch = upload(prepared['batch'], device)
        if args.speaker_noise > 0:
            vectors = model.speaker_vector(batch['speaker'])
            batch['speakerVector'] = vectors + args.speaker_noise * torch.randn_like(vectors)
        progress = step / args.steps
        guide = max(args.guide_floor, 1 - (1 - args.guide_floor) * progress / args.guide_steps)
        losses, audio, target, stats = batch_losses(model, synth, batch, args, None, prepared['seed'], guide, sonorants, prepared)
        if critic is not None and corpus == 'multi':
            # The critic's step (on the rendering, detached), then the generator's adversarial and feature-matching
            # losses (after --adversarial-start steps of the critic alone). The discriminators' input spectrograms don't
            # depend on their weights, so each signal's are computed once for both passes.
            generator_pass = step >= args.adversarial_start
            real_spectrograms = critic.spectrograms(target)
            with torch.set_grad_enabled(generator_pass):
                fake_spectrograms = critic.spectrograms(audio)
            critic_loss = discriminator.discriminator_loss(critic.score(real_spectrograms),
                                                           critic.score([value.detach() for value in fake_spectrograms]))
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            losses['critic'] = critic_loss.detach()
            if generator_pass:
                # (the real features are targets only - no graph; and the generator's backward needs the gradient
                # of the rendering's scores and features with respect to the audio only, not to the critic's weights)
                with torch.no_grad():
                    real_outputs = critic.score(real_spectrograms)
                critic.requires_grad_(False)
                fake_outputs = critic.score(fake_spectrograms)
                critic.requires_grad_(True)
                adversarial, matching = discriminator.generator_losses(real_outputs, fake_outputs)
                losses['adversarial'] = adversarial.detach()
                losses['matching'] = matching.detach()
                losses['total'] = losses['total'] + args.adversarial * adversarial + args.feature_matching * matching
        optimizer.zero_grad()
        losses['total'].backward()
        if lean:
            # (the clip inside the step, which skips a non-finite one on the device)
            optimizer.step()
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if fused:  # pragma: no cover - the GPU's fused optimizer
                # (the fused step skips itself - parameters, moments, and step count - when found_inf is 1)
                optimizer.found_inf = (~torch.isfinite(norm)).float()
                optimizer.step()
            elif torch.isfinite(norm):
                optimizer.step()
        if ema is not None:
            ema.update()
        scheduler.step()
        step += 1
        if device.type == 'mps':  # pragma: no cover - the Apple GPU only
            event = torch.mps.Event()
            event.record()
            in_flight.append((event, prepared))
            while len(in_flight) > IN_FLIGHT:
                in_flight.popleft()[0].synchronize()
        prefix = '' if corpus == 'multi' else 'cremad.'
        pending.append((prefix, list(losses), list(stats),
                        torch.stack([value.detach().float() for value in losses.values()] + list(stats.values()))))
        if step % args.log_every == 0 or len(pending) >= 1000:
            # The pending steps' values, read at once, summed in step order (as floats, as they were read each step)
            read = torch.cat([values for _, _, _, values in pending]).cpu().tolist()
            offset = 0
            for prefix_keys, loss_keys, stat_keys, _ in pending:
                step_values = read[offset:offset + len(loss_keys) + len(stat_keys)]
                offset += len(step_values)
                for key, value in zip(loss_keys + stat_keys, step_values):
                    # (statistics are NaN without strong frames - left out)
                    if key in stat_keys and math.isnan(value):
                        continue
                    sums[prefix_keys + key] = sums.get(prefix_keys + key, 0) + value
                    sum_counts[prefix_keys + key] = sum_counts.get(prefix_keys + key, 0) + 1
            pending = []
            # (a plan made ahead that didn't match its frames - one finished on the device can't be made again)
            if synth.deferred_plan_mismatches():  # pragma: no cover - plans are finished on the device only on the Apple GPU
                raise RuntimeError('a synthesis cycle plan made ahead did not match its frames')
        if step % args.log_every == 0:
            elapsed = time.time() - last_time
            last_time = time.time()
            row = {'step': step, 'lr': scheduler.get_last_lr()[0], 'secondsPerStep': elapsed / args.log_every, 'guide': guide,
                   **{key: value / sum_counts[key] for key, value in sums.items()}}
            print(' '.join(f'{key} {value:.4g}' if isinstance(value, float) else f'{key} {value}' for key, value in row.items()),
                  file=sys.stderr, flush=True)
            log.write(json.dumps({'train': row}) + '\n')
            log.flush()
            sums, sum_counts = {}, {}
        # Validation (every --val-every steps from --val-from, and the last step) - the best checkpoint is the lowest
        # reader score; last.pt every --save-every steps and at each validation; step-<n>.pt every --keep-every steps.
        # (Validation draws from no training random state - its own rng, eval mode, the synthesizer's seeded noise - so
        # the schedule doesn't change the training.)
        validating = step == stop or (step % args.val_every == 0 and step >= args.val_from)
        extra = None
        if ema is not None and validating:
            # The EMA weights' validation (an emaVal row) and checkpoints - ema-best.pt (the lowest reader score), ema-last.pt
            ema.copy_to(ema_model.parameters())
            ema_results = {name: validate(ema_model, synth, corpus_set, args, device, sonorants) for name, corpus_set in val_sets.items()}
            ema_result = dict(ema_results['multi'])
            for name, other in ema_results.items():
                if name != 'multi':
                    ema_result.update({f'{name}.{key}': value for key, value in other.items()})
            print('  EMA VAL ' + ' '.join(f'{key} {value:.4g}' for key, value in ema_result.items()), file=sys.stderr, flush=True)
            log.write(json.dumps({'emaVal': {'step': step, **ema_result}}) + '\n')
            log.flush()
            if ema_result['score'] < ema_best:
                ema_best = ema_result['score']
                save(os.path.join(directory, 'ema-best.pt'), {'val': ema_result}, ema_model)
            save(os.path.join(directory, 'ema-last.pt'), {'val': ema_result}, ema_model)
        if validating:
            results = {name: validate(model, synth, corpus_set, args, device, sonorants) for name, corpus_set in val_sets.items()}
            result = dict(results['multi'])
            for name, other in results.items():
                if name != 'multi':
                    result.update({f'{name}.{key}': value for key, value in other.items()})
            print('  VAL ' + ' '.join(f'{key} {value:.4g}' for key, value in result.items()), file=sys.stderr, flush=True)
            log.write(json.dumps({'val': {'step': step, **result}}) + '\n')
            log.flush()
            extra = {'val': result}
            if result['score'] < best:
                best = result['score']
                save(os.path.join(directory, 'best.pt'), extra)
        if validating or (args.save_every and step % args.save_every == 0):
            save(os.path.join(directory, 'last.pt'), extra)
        if args.keep_every and step % args.keep_every == 0:
            save(os.path.join(directory, f'step-{step}.pt'), extra)
    prefetcher.close()
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
