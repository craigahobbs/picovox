# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Knowledge distillation of the multi-speaker acoustic model (nn/model.py's MultiModel) - small students trained to
imitate a big teacher, the way the project shrinks its voice model at each retrain: train a big teacher, then

    build/venv-nn/bin/python -B -m nn.distill targets [--teacher build/nn/runs/multi-b/best.pt]    # 1. the distillation set
    build/venv-nn/bin/python -B -m nn.distill train --student s64                                 # 2. a student (each one)
    build/venv-nn/bin/python -B -m nn.distill polish --student s64                                # 3. optional - real audio
    build/venv-nn/bin/python -B -m nn.distill eval teacher s96 s64 s48 s32 s64-polish             # 4. the browser path
    build/venv-nn/bin/python -B -m nn.distill pick                                                # 5. the smallest in tolerance
    build/venv-nn/bin/python -B -m nn.distill ladder                                              # 1-5, all of the ladder

The settings - the teacher, the distillation set's sources, the training settings, the student ladder, the polish, the
evaluation speakers, and the tolerance - are data/nn/distill.json (--config another file). The outputs go to
NN_DISTILL_DIR (default build/nn/distill/): targets/ (the distillation set), eval/<model>/ (exports, renderings,
result.json), and summary.json (pick); progress lines go to its code.log, targets.log, students.log, and eval.log. Runs
go to build/nn/runs/distill-<student>/ (train.log, log.jsonl, best.pt, last.pt), and listening samples to
build/nn/distill-samples/.

The method:

- Targets - sequence-level distillation in parameter space. The teacher speaks (tokens, speaker, style) to its
  durations (pauses included) and its normalized parameter frames (nn/params.py, 5 ms), exactly as inference does
  (model.infer's duration rounding and pause threshold) but without rendering audio. The inputs are (a) the training
  utterances' tokens with their own speakers and styles (the readers, and the emotional corpus's actors with their
  emotions and intensities) and (b) sentences of real book text (Project Gutenberg books, tokenized by the Python front
  end - the training-time label source - and split into sentences as inference splits them), each spoken by a random
  reader (or actor, or a blend of two readers) in a neutral or random emotional style at a random intensity. Tokens
  are remapped as inference remaps them (nn/export.py token_remap). Frames are stored as float16.
- Students - MultiModel with smaller widths and depths. They share the teacher's conditioning space: the speaker,
  style, and channel-equalizer tables are copied from the teacher and frozen, so voices designed as blends of the
  teacher's table are the same voices in a student. A student learns, with the teacher's durations (teacher forcing on
  its length regulator), L1 on the normalized parameter frames (all but the constant jitter and shimmer) plus the
  duration losses against the teacher's duration outputs: phones' log durations (MSE), the pause decision (the KL
  divergence from the teacher's pause probability), and the paused durations (MSE weighted by the teacher's pause
  probability). Options: --init-teacher (copy every teacher tensor of the same name and shape) and --feature-weight
  (intermediate-feature matching: the student's encoder output and last decoder features, projected to the teacher's
  widths, against the teacher's).
- Polish - an optional short fine-tune with the real-audio losses (nn/trainmulti.py: the multi-resolution spectral
  loss through the differentiable synthesizer on the recordings, and the adversarial critic and feature matching as
  stage B used them, the critic starting from the teacher's).
- Evaluation - through the browser path: each model is exported (nn/export.py, 8-bit), packaged (nn/package.py - the
  generated JavaScript module's bytes and gzip bytes), and spoken by nn/js/speak.js on the listener panel's texts
  (tools/panel-texts.json) for six speakers, then scored: UTMOS (tools/listeners.py), Whisper's WER (whisper.cpp
  base.en), and the ECAPA speaker similarity to the teacher's rendering of the same speaker and texts. The speed is
  nn/js/speak.js's real-time factor at 44.1 kHz. pick chooses the smallest student within the tolerance.
"""

import argparse
import concurrent.futures
import datetime
import json
import math
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy
import torch

from .data import DATA_DIR, ArcticData
from .export import export, load_model, token_remap, training_token_counts
from .frontend import tokenize, vocabulary
from .model import BOS, FIRST_PHONE, FIXED_NAMES, PAD, STYLE_NAMES, MultiModel, is_boundary, parameter_count
from .synthesize import sentences
from . import params


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, 'data', 'nn', 'distill.json')
DISTILL_DIR = os.environ.get('NN_DISTILL_DIR', os.path.join(ROOT, 'build', 'nn', 'distill'))
RUNS_DIR = os.path.join(ROOT, 'build', 'nn', 'runs')
SAMPLES_DIR = os.environ.get('NN_DISTILL_SAMPLES', os.path.join(ROOT, 'build', 'nn', 'distill-samples'))
LISTENERS_PYTHON = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')
WHISPER_MODEL = os.path.join(ROOT, 'data', 'whisper', 'ggml-base.en.bin')

# (NNPACK's depthwise 1-D convolution on the CPU is about 8x slower than PyTorch's own)
torch.backends.nnpack.set_flags(False)

# The distillation set's sources
SOURCES = ('readers', 'actors', 'books')

# The style intensities of CREMA-D's intensity labels (nn/trainmulti.py INTENSITY)
INTENSITY = {'low': 0.5, 'medium': 0.75, 'high': 1.0, 'unspecified': 0.75}

# The distillation set's per-item record
ITEM_DTYPE = numpy.dtype([
    ('tokenOffset', '<i8'), ('tokenCount', '<i4'), ('frameOffset', '<i8'), ('frameCount', '<i4'),
    ('speakerA', '<i2'), ('speakerB', '<i2'), ('blend', '<f4'), ('emotion', '<i1'), ('intensity', '<f4'),
    ('source', '<i1'), ('split', '<i1')
])


#
# Settings and logs
#

def load_config(path=CONFIG_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# A path of the settings (relative to the repository root) as an absolute path
def root_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# Append a progress line to a log in DISTILL_DIR (and stderr)
def log_line(log, text):
    os.makedirs(DISTILL_DIR, exist_ok=True)
    line = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {text}'
    with open(os.path.join(DISTILL_DIR, f'{log}.log'), 'a', encoding='utf-8') as file:
        file.write(line + '\n')
    print(line, file=sys.stderr, flush=True)


# The device - mps when available, else cpu
def default_device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


#
# Book text
#

GUTENBERG_START = re.compile(r'^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$', re.MULTILINE)
GUTENBERG_END = re.compile(r'^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$', re.MULTILINE)


# A Gutenberg book's paragraphs (the text between the license header and footer, line-wrapped paragraphs joined)
def book_paragraphs(path):
    with open(path, encoding='utf-8', errors='replace') as file:
        text = file.read().replace('\r\n', '\n')
    start = GUTENBERG_START.search(text)
    end = GUTENBERG_END.search(text)
    text = text[start.end() if start else 0:end.start() if end else len(text)]
    paragraphs = []
    for block in re.split(r'\n\s*\n', text):
        paragraph = ' '.join(line.strip() for line in block.split('\n')).strip()
        if paragraph and re.search('[a-zA-Z]', paragraph):
            paragraphs.append(paragraph.replace('_', ''))
    return paragraphs


# A book's sentences as token id lists (the front end's tokens, split as inference splits them) - (path, sentences)
def book_sentences(path):
    vocab = vocabulary()
    out = []
    for paragraph in book_paragraphs(path):
        out.extend(sentences(tokenize(paragraph)['ids'], vocab))
    return path, out


#
# The teacher's targets
#

def teacher_outputs(model, tokens, speaker_vector, emotion, intensity):
    """The teacher's inference outputs for a batch - (duration outputs [B, N, 3], integer durations [B, N] (10 ms
    frames, as model.infer), normalized parameter frames [B, 2F, COUNT])"""
    condition = model.condition(None, emotion, intensity, speaker_vector)
    features, mask = model.encode(tokens, condition)
    duration_outputs = model.durations(features, mask, condition)
    durations = model.predicted_durations(duration_outputs, tokens)
    regulated, positions, frame_mask = model.regulate(features, durations)
    unit = model.decode(regulated, positions, frame_mask, condition)
    return duration_outputs, durations, unit


# The speaker vectors [B, speakerDim] of a batch's speakers - blends (1 - blend) A + blend B of the speaker table's rows
def blend_vectors(table, speaker_a, speaker_b, blend):
    return table[speaker_a] * (1 - blend)[:, None] + table[speaker_b] * blend[:, None]


# The distillation set's inputs - (token lists, item records without offsets), the source counts, and the book report
def target_inputs(state, settings, remap, log):
    rng = numpy.random.default_rng(settings['seed'])
    reader_count = state['readerCount']
    readers = [index for index in range(reader_count) if state['speakers'][index]]
    actors = list(range(reader_count, len(state['speakers'])))
    token_lists = []
    records = []

    def add(tokens, source, speaker_a, speaker_b=None, blend=0.0, emotion=-1, intensity=0.0, split=0):
        token_lists.append(numpy.array([remap.get(int(token), int(token)) for token in tokens], dtype=numpy.int16))
        records.append((0, len(tokens), 0, 0, speaker_a, speaker_a if speaker_b is None else speaker_b, blend, emotion,
                        intensity, source, split))

    # (a) the training utterances with their own speakers and styles
    for corpus, source, offset in (('multi', 0, 0), ('cremad', 1, reader_count)):
        data = ArcticData(os.path.join(DATA_DIR, corpus))
        speakers = data.index['speakers']
        for entry in data.entries('train'):
            emotion = entry.get('emotion')
            style = STYLE_NAMES.index(emotion) if emotion in STYLE_NAMES else -1
            tokens = data.tokens[entry['tokenOffset']:entry['tokenOffset'] + entry['tokenCount']]
            add(tokens, source, offset + speakers[entry['speaker']]['index'], emotion=style,
                intensity=INTENSITY.get(entry.get('intensity'), 0.75) if style >= 0 else 0.0)
    log(f'training utterances: {len(records)} ({sum(1 for record in records if record[9] == 0)} readers, '
        f'{sum(1 for record in records if record[9] == 1)} actors)')

    # (b) the books' sentences, each with a random speaker and style
    books_dir = root_path(settings['books'])
    paths = sorted(os.path.join(books_dir, name) for name in os.listdir(books_dir) if name.endswith('.txt'))
    book_report = {}
    seen = set()
    sentence_list = []
    with multiprocessing.Pool(min(8, os.cpu_count() or 1)) as pool:
        for number, (path, book) in enumerate(pool.imap(book_sentences, paths), 1):
            kept = 0
            for sentence in book:
                key = tuple(sentence)
                if settings['minTokens'] <= len(sentence) <= settings['maxTokens'] and key not in seen:
                    seen.add(key)
                    sentence_list.append(sentence)
                    kept += 1
            book_report[os.path.basename(path)] = {'sentences': len(book), 'kept': kept}
            log(f'books: {number}/{len(paths)} {os.path.basename(path)}: {len(book)} sentences, {kept} kept')
    order = rng.permutation(len(sentence_list))
    val_count = settings['valItems']
    for copy in range(settings['bookCopies']):
        for rank, index in enumerate(order):
            split = 1 if rank < val_count else 0
            if split and copy:
                continue
            draw = rng.random()
            blend = 0.0
            if draw < settings['actorShare']:
                speaker_a = speaker_b = int(rng.choice(actors))
            else:
                speaker_a = int(rng.choice(readers))
                speaker_b = speaker_a
                if rng.random() < settings['blendShare']:
                    speaker_b = int(rng.choice(readers))
                    blend = float(rng.uniform(0.2, 0.8))
            emotion, intensity = -1, 0.0
            if rng.random() < settings['emotionShare']:
                emotion = int(rng.integers(len(STYLE_NAMES)))
                intensity = float(rng.uniform(*settings['intensity']))
            add(sentence_list[index], 2, speaker_a, speaker_b, blend, emotion, intensity, split)
    return token_lists, records, book_report


def make_targets(args):
    """The targets subcommand - the distillation set"""
    config = load_config(args.config)
    settings = {**config['targets'], **({'bookCopies': args.book_copies} if args.book_copies else {})}
    teacher_path = root_path(args.teacher or config['teacher'])
    out_dir = args.out or os.path.join(DISTILL_DIR, 'targets')
    device = torch.device(args.device or default_device())

    def log(text):
        log_line('targets', text)
    start = time.time()
    log(f'teacher {teacher_path}; out {out_dir}; device {device}')
    model, state = load_model(teacher_path, device)
    counts, vocab = training_token_counts(state)
    remap_names = token_remap(vocab, counts)
    remap = {vocab.index(source): vocab.index(target) for source, target in remap_names.items()}
    token_lists, records, book_report = target_inputs(state, settings, remap, log)
    if args.limit:
        keep = numpy.random.default_rng(0).permutation(len(records))[:args.limit]
        keep = sorted(keep, key=lambda index: (records[index][10] == 0, index))
        token_lists = [token_lists[index] for index in keep]
        records = [records[index] for index in keep]
    items = numpy.array(records, dtype=ITEM_DTYPE)
    prepare_seconds = time.time() - start
    log(f'inputs: {len(items)} items ({", ".join(f"{name} {int((items["source"] == index).sum())}" for index, name in enumerate(SOURCES))}; '
        f'{int((items["split"] == 1).sum())} validation) in {prepare_seconds:.0f} s')

    # The teacher's outputs, in length-sorted batches, appended to the arrays' files
    os.makedirs(out_dir, exist_ok=True)
    files = {name: open(os.path.join(out_dir, f'{name}.bin'), 'wb')  # pylint: disable=consider-using-with
             for name in ('tokens', 'durations', 'durationOutputs', 'frames')}
    order = numpy.argsort(items['tokenCount'], kind='stable')
    token_total = frame_total = 0
    batches = []
    current = []
    for index in order:
        if current and (len(current) >= args.batch or (len(current) + 1) * items['tokenCount'][index] > args.batch_tokens):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    table = model.speaker.weight.detach()
    generate_start = time.time()
    with torch.no_grad():
        for number, batch in enumerate(batches, 1):
            length = int(items['tokenCount'][batch].max())
            tokens = numpy.zeros((len(batch), length), dtype=numpy.int64)
            for row, index in enumerate(batch):
                tokens[row, :items['tokenCount'][index]] = token_lists[index]
            record = items[batch]
            speaker_vector = blend_vectors(table, torch.tensor(record['speakerA'].astype(numpy.int64), device=device),
                                           torch.tensor(record['speakerB'].astype(numpy.int64), device=device),
                                           torch.tensor(record['blend'].astype(numpy.float32), device=device))
            duration_outputs, durations, unit = teacher_outputs(
                model, torch.tensor(tokens, device=device), speaker_vector, torch.tensor(record['emotion'].astype(numpy.int64), device=device),
                torch.tensor(record['intensity'].astype(numpy.float32), device=device))
            duration_outputs = duration_outputs.float().cpu().numpy().astype(numpy.float16)
            durations = durations.cpu().numpy()
            unit = unit.float().cpu().numpy().astype(numpy.float16)
            for row, index in enumerate(batch):
                count = int(items['tokenCount'][index])
                frame_count = int(durations[row, :count].sum())
                items['tokenOffset'][index] = token_total
                items['frameOffset'][index] = frame_total
                items['frameCount'][index] = frame_count
                files['tokens'].write(token_lists[index].astype('<i2').tobytes())
                files['durations'].write(durations[row, :count].astype('<i2').tobytes())
                files['durationOutputs'].write(duration_outputs[row, :count].astype('<f2').tobytes())
                files['frames'].write(unit[row, :2 * frame_count].astype('<f2').tobytes())
                token_total += count
                frame_total += 2 * frame_count
            if number % 200 == 0 or number == len(batches):
                elapsed = time.time() - generate_start
                log(f'teacher: {number}/{len(batches)} batches, {frame_total / 2 / 100 / 3600:.1f} hours of speech, {elapsed:.0f} s')
    for file in files.values():
        file.close()
    generate_seconds = time.time() - generate_start
    numpy.save(os.path.join(out_dir, 'items.npy'), items)
    sizes = {name: os.path.getsize(os.path.join(out_dir, f'{name}.bin')) for name in files}
    hours = {name: round(float(items['frameCount'][items['source'] == index].sum()) / 100 / 3600, 2) for index, name in enumerate(SOURCES)}
    index = {
        'format': 'voice-synth-distill', 'version': 1, 'teacher': teacher_path, 'teacherStep': state.get('step'),
        'teacherConfig': state['config'], 'speakers': state['speakers'], 'readerCount': state['readerCount'],
        'styles': state['styles'], 'remap': remap_names, 'vocab': vocab, 'settings': settings,
        'count': len(items), 'sources': {name: int((items['source'] == ix).sum()) for ix, name in enumerate(SOURCES)},
        'validation': int((items['split'] == 1).sum()), 'tokens': token_total, 'frames': frame_total, 'hours': hours,
        'arrays': {'tokens': ['<i2', [token_total]], 'durations': ['<i2', [token_total]],
                   'durationOutputs': ['<f2', [token_total, 3]], 'frames': ['<f2', [frame_total, params.COUNT]]},
        'bytes': sizes, 'books': book_report, 'prepareSeconds': round(prepare_seconds, 1),
        'generateSeconds': round(generate_seconds, 1), 'seconds': round(time.time() - start, 1), 'device': str(device)
    }
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=1)
    log(f'done: {len(items)} sentence-speaker pairs ({", ".join(f"{name} {count}" for name, count in index["sources"].items())}), '
        f'{sum(hours.values()):.0f} hours of teacher speech ({", ".join(f"{name} {value:.0f} h" for name, value in hours.items())}), '
        f'{sum(sizes.values()) / 1e9:.2f} GB, generated in {generate_seconds / 60:.1f} min ({index["seconds"] / 60:.1f} min in all)')
    return 0


class Targets:
    """A distillation set (memory-mapped)"""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, 'index.json'), encoding='utf-8') as file:
            self.index = json.load(file)
        self.items = numpy.load(os.path.join(directory, 'items.npy'))
        self.arrays = {name: numpy.memmap(os.path.join(directory, f'{name}.bin'), dtype=dtype, mode='r', shape=tuple(shape))
                       for name, (dtype, shape) in self.index['arrays'].items()}

    # The item indexes of a split (0 train, 1 validation)
    def split(self, split):
        return numpy.nonzero(self.items['split'] == split)[0]


# Round up to a multiple (fewer distinct shapes - the GPU's kernels are cached per shape)
def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def collate_targets(targets, indices):
    """A batch of distillation items as host tensors (padded - tokens to a multiple of 8, frames to a multiple of 16)"""
    items = targets.items[indices]
    arrays = targets.arrays
    token_count = round_up(int(items['tokenCount'].max()), 8)
    frame_count = round_up(int(items['frameCount'].max()), 16)
    batch = len(indices)
    tokens = numpy.zeros((batch, token_count), dtype=numpy.int64)
    durations = numpy.zeros((batch, token_count), dtype=numpy.int64)
    duration_outputs = numpy.zeros((batch, token_count, 3), dtype=numpy.float32)
    unit = numpy.zeros((batch, 2 * frame_count, params.COUNT), dtype=numpy.float32)
    for row, item in enumerate(items):
        token_slice = slice(item['tokenOffset'], item['tokenOffset'] + item['tokenCount'])
        tokens[row, :item['tokenCount']] = arrays['tokens'][token_slice]
        durations[row, :item['tokenCount']] = arrays['durations'][token_slice]
        duration_outputs[row, :item['tokenCount']] = arrays['durationOutputs'][token_slice]
        unit[row, :2 * item['frameCount']] = arrays['frames'][item['frameOffset']:item['frameOffset'] + 2 * item['frameCount']]
    return {
        'tokens': torch.from_numpy(tokens), 'durations': torch.from_numpy(durations),
        'durationOutputs': torch.from_numpy(duration_outputs), 'unit': torch.from_numpy(unit),
        'frameCounts': torch.from_numpy(items['frameCount'].astype(numpy.int64)),
        'speakerA': torch.from_numpy(items['speakerA'].astype(numpy.int64)), 'speakerB': torch.from_numpy(items['speakerB'].astype(numpy.int64)),
        'blend': torch.from_numpy(items['blend'].astype(numpy.float32)), 'emotion': torch.from_numpy(items['emotion'].astype(numpy.int64)),
        'intensity': torch.from_numpy(items['intensity'].astype(numpy.float32)), 'frameCount': frame_count
    }


# Length-bucketed batches of item indexes (shuffled) - at most batch_size items and max_frames padded frames each
def target_batches(targets, indices, batch_size, max_frames, rng):
    frames = targets.items['frameCount'][indices].astype(numpy.float64)
    order = indices[numpy.argsort(frames + 40 * rng.random(len(indices)))]
    batches = []
    current = []
    longest = 0
    for index in order:
        count = int(targets.items['frameCount'][index])
        if current and (len(current) >= batch_size or max(longest, count) * (len(current) + 1) > max_frames):
            batches.append(numpy.array(current))
            current, longest = [], 0
        current.append(index)
        longest = max(longest, count)
    if current:
        batches.append(numpy.array(current))
    rng.shuffle(batches)
    return batches


class BatchPrefetcher:
    """Training batches collated ahead on a thread"""

    def __init__(self, targets, indices, batch_size, max_frames, rng, depth=4):
        self.targets = targets
        self.indices = indices
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.rng = rng
        self.queue = queue.Queue(maxsize=depth)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopping.is_set():
                for batch in target_batches(self.targets, self.indices, self.batch_size, self.max_frames, self.rng):
                    collated = collate_targets(self.targets, batch)
                    while not self.stopping.is_set():
                        try:
                            self.queue.put(collated, timeout=0.1)
                            break
                        except queue.Full:
                            pass
                    if self.stopping.is_set():
                        return
        except BaseException as error:  # pylint: disable=broad-except
            self.queue.put(error)

    def next(self):
        batch = self.queue.get()
        if isinstance(batch, BaseException):
            raise batch
        return batch

    def close(self):
        self.stopping.set()
        while self.thread.is_alive():
            try:
                self.queue.get(timeout=0.1)
            except queue.Empty:
                pass
        self.thread.join()


#
# Students
#

# The decoder's features before its norm and head, and the normalized frames - (features [B, F, D], unit [B, 2F, COUNT])
def decode_features(model, regulated, positions, mask, condition):
    x = (model.decoder_in(regulated) + model.position_in(positions) + model.condition_decoder(condition)) * mask
    for layer in model.decoder:
        x = layer(x, mask, condition)
    logits = model.head(model.decoder_norm(x))
    batch, frames, _ = logits.shape
    return x, torch.sigmoid(logits.reshape(batch, frames * model.config['subframes'], params.COUNT))


def run_model(model, batch, durations=None):
    """A model's forward pass on a distillation batch with the teacher's durations - {duration [B, N, 3], unit [B, 2F,
    COUNT], encoded [B, N, D], decoded [B, F, D], frameMask [B, F, 1]}"""
    table = model.speaker.weight
    speaker_vector = blend_vectors(table, batch['speakerA'], batch['speakerB'], batch['blend'])
    condition = model.condition(None, batch['emotion'], batch['intensity'], speaker_vector)
    features, mask = model.encode(batch['tokens'], condition)
    duration_outputs = model.durations(features, mask, condition)
    durations = batch['durations'] if durations is None else durations
    regulated, positions, frame_mask = model.regulate(features, durations, batch['frameCount'])
    decoded, unit = decode_features(model, regulated, positions, frame_mask, condition)
    return {'duration': duration_outputs, 'unit': unit, 'encoded': features, 'decoded': decoded, 'frameMask': frame_mask}


def distill_losses(outputs, batch, column_mask):
    """The distillation losses of a student's outputs against the teacher's - {frames, durPhone, durPause, durPaused}"""
    subframes = outputs['unit'].shape[1] // outputs['frameMask'].shape[1]
    frame_mask = outputs['frameMask'].repeat_interleave(subframes, dim=1)
    weight = frame_mask * column_mask
    losses = {'frames': ((outputs['unit'] - batch['unit']).abs() * weight).sum() / weight.sum().clamp(min=1)}
    tokens = batch['tokens']
    log_duration, pause_logit, log_paused = outputs['duration'].unbind(-1)
    teacher_duration, teacher_logit, teacher_paused = batch['durationOutputs'].unbind(-1)
    phone = (tokens >= FIRST_PHONE).float()
    boundary = (is_boundary(tokens) | (tokens == BOS)).float()
    losses['durPhone'] = ((log_duration - teacher_duration) ** 2 * phone).sum() / phone.sum().clamp(min=1)
    probability = torch.sigmoid(teacher_logit)
    # (the KL divergence from the teacher's pause probability - the cross entropy less the teacher's entropy)
    cross = torch.nn.functional.binary_cross_entropy_with_logits(pause_logit, probability, reduction='none')
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(teacher_logit, probability, reduction='none')
    losses['durPause'] = ((cross - entropy) * boundary).sum() / boundary.sum().clamp(min=1)
    paused_weight = boundary * probability
    losses['durPaused'] = ((log_paused - teacher_paused) ** 2 * paused_weight).sum() / paused_weight.sum().clamp(min=1)
    return losses


def student_config(config, name):
    """A student's model config (the ladder's entry by name, or a JSON config)"""
    for student in config['students']:
        if student['name'] == name:
            return dict(student['config'])
    raise ValueError(f'unknown student {name} (the ladder: {", ".join(student["name"] for student in config["students"])})')


def build_student(student_cfg, teacher, teacher_state, dropout, init_teacher=False):
    """A student - the teacher's config with the student's widths and depths, the teacher's speaker, style, and
    equalizer tables (copied, frozen), and with init_teacher every teacher tensor of the same name and shape"""
    config = {key: value for key, value in teacher_state['config'].items()}
    config.update(student_cfg)
    model = MultiModel(config, priors=teacher_state.get('priors'), dropout=dropout)
    teacher_tensors = teacher.state_dict()
    state = model.state_dict()
    copied = []
    for name, tensor in state.items():
        shared = name in ('speaker.weight', 'style', 'eq_free')
        if name in teacher_tensors and teacher_tensors[name].shape == tensor.shape and (shared or init_teacher):
            state[name] = teacher_tensors[name].detach().clone().to(tensor.dtype)
            copied.append(name)
    model.load_state_dict(state)
    for parameter in (model.speaker.weight, model.style, model.eq_free):
        parameter.requires_grad_(False)
    return model, copied


class FeatureProjections(torch.nn.Module):
    """Linear maps from a student's encoder and decoder features to the teacher's widths (intermediate-feature matching)"""

    def __init__(self, student, teacher):
        super().__init__()
        self.encoder = torch.nn.Linear(student.config['dim'], teacher.config['dim'])
        self.decoder = torch.nn.Linear(student.config['decoderDim'], teacher.config['decoderDim'])


# The feature-matching loss - the projected student features against the teacher's (layer-normed, so scale-free), MSE
def feature_loss(projections, outputs, teacher_outputs_, batch):
    token_mask = (batch['tokens'] != PAD).float()[..., None]
    frame_mask = outputs['frameMask']
    norm = torch.nn.functional.layer_norm
    encoder_target = norm(teacher_outputs_['encoded'], teacher_outputs_['encoded'].shape[-1:])
    decoder_target = norm(teacher_outputs_['decoded'], teacher_outputs_['decoded'].shape[-1:])
    encoder = norm(projections.encoder(outputs['encoded']), encoder_target.shape[-1:])
    decoder = norm(projections.decoder(outputs['decoded']), decoder_target.shape[-1:])
    encoder_loss = (((encoder - encoder_target) ** 2).mean(-1, keepdim=True) * token_mask).sum() / token_mask.sum()
    decoder_loss = (((decoder - decoder_target) ** 2).mean(-1, keepdim=True) * frame_mask).sum() / frame_mask.sum()
    return encoder_loss + decoder_loss


@torch.no_grad()
def validate_student(model, targets, indices, device, column_mask, batch_size=48, max_frames=24000):
    """The distillation losses on the validation items, and the predicted durations' error against the teacher's"""
    model.eval()
    sums = {}
    count = 0
    duration_error = []
    for batch_indices in target_batches(targets, indices, batch_size, max_frames, numpy.random.default_rng(0)):
        host = collate_targets(targets, batch_indices)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in host.items()}
        outputs = run_model(model, batch)
        for key, value in distill_losses(outputs, batch, column_mask).items():
            sums[key] = sums.get(key, 0) + float(value) * len(batch_indices)
        predicted = model.predicted_durations(outputs['duration'], batch['tokens'])
        valid = batch['tokens'] != PAD
        duration_error.append((predicted - batch['durations']).abs()[valid].float().cpu().numpy())
        count += len(batch_indices)
    model.train()
    result = {key: value / count for key, value in sums.items()}
    errors = numpy.concatenate(duration_error)
    result['durationErrorFrames'] = float(errors.mean())
    return result


def checkpoint_state(model, teacher_state, step, args_dict, val=None, extra=None):
    """A student checkpoint - nn/export.py's and nn/trainmulti.py's format (the teacher's speaker list, priors, and
    style names; args with the teacher's cremad share, so the export's token remap is the teacher's)"""
    return {'model': model.state_dict(), 'config': model.config, 'priors': teacher_state.get('priors'),
            'speakers': teacher_state['speakers'], 'readerCount': teacher_state['readerCount'], 'styles': teacher_state['styles'],
            'step': step, 'val': val, 'args': {**args_dict, 'cremad': teacher_state['args'].get('cremad', 0)},
            **(extra or {})}


def train_student(args):
    """The train subcommand - a student distilled from the teacher's targets"""
    config = load_config(args.config)
    settings = {**config['train']}
    for key in ('steps', 'batch', 'lr', 'warmup'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    name = args.name or args.student
    directory = os.path.join(RUNS_DIR, f'distill-{name}')
    os.makedirs(directory, exist_ok=True)
    device = torch.device(args.device or default_device())
    torch.manual_seed(args.seed)
    rng = numpy.random.default_rng(args.seed)
    targets = Targets(args.targets or os.path.join(DISTILL_DIR, 'targets'))
    teacher, teacher_state = load_model(targets.index['teacher'] if not args.teacher else root_path(args.teacher), device)
    student_cfg = json.loads(args.student_config) if args.student_config else student_config(config, args.student)
    model, copied = build_student(student_cfg, teacher.cpu(), teacher_state, settings['dropout'], args.init_teacher)
    model = model.to(device)
    teacher = teacher.to(device)
    projections = FeatureProjections(model, teacher).to(device) if args.feature_weight > 0 else None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if projections is not None:
        trainable += list(projections.parameters())
    fused = device.type in ('mps', 'cuda')
    optimizer = torch.optim.AdamW(trainable, lr=settings['lr'], weight_decay=settings['weightDecay'], betas=(0.9, 0.98),
                                  fused=fused or None)
    steps = settings['steps']

    def learning_rate(step):
        if step < settings['warmup']:
            return (step + 1) / settings['warmup']
        progress = (step - settings['warmup']) / max(1, steps - settings['warmup'])
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    column_mask = torch.ones(params.COUNT, device=device)
    for fixed in FIXED_NAMES:
        column_mask[params.INDEX[fixed]] = 0
    args_dict = {key: value for key, value in vars(args).items() if key != 'func'}
    args_dict['settings'] = settings

    train_log = open(os.path.join(directory, 'train.log'), 'a', encoding='utf-8')  # pylint: disable=consider-using-with

    def say(text):
        print(text, file=sys.stderr, flush=True)
        train_log.write(text + '\n')
        train_log.flush()
    count = parameter_count(model)
    core = count - model.speaker.weight.numel() - model.eq_free.numel()
    say(f'model: {count} parameters ({core} without the speaker and equalizer tables); distill-{name}: {steps} steps from 0')
    say(f'student {json.dumps(student_cfg)}; teacher {targets.index["teacher"]} (step {targets.index["teacherStep"]}); targets '
        f'{targets.directory} ({targets.index["count"]} items); copied from the teacher: {", ".join(copied)}; '
        f'feature weight {args.feature_weight}; settings {json.dumps(settings)}')
    log_line('students', f'{name}: training - {count} parameters, {json.dumps(student_cfg)}, {steps} steps')

    train_indices = targets.split(0)
    val_indices = targets.split(1)
    prefetcher = BatchPrefetcher(targets, train_indices, settings['batch'], settings['maxFrames'], rng)
    log = open(os.path.join(directory, 'log.jsonl'), 'a', encoding='utf-8')  # pylint: disable=consider-using-with
    best = math.inf
    step = 0
    pending = []
    last_time = start_time = time.time()
    model.train()
    if projections is not None:
        teacher.eval()
    while step < steps:
        host = prefetcher.next()
        batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in host.items()}
        outputs = run_model(model, batch)
        losses = distill_losses(outputs, batch, column_mask)
        total = losses['frames'] + settings['durationWeight'] * (losses['durPhone'] + losses['durPause'] + losses['durPaused'])
        if projections is not None:
            with torch.no_grad():
                teacher_result = run_model(teacher, batch)
            losses['features'] = feature_loss(projections, outputs, teacher_result, batch)
            total = total + args.feature_weight * losses['features']
        losses['total'] = total
        optimizer.zero_grad()
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if fused:
            optimizer.found_inf = (~torch.isfinite(norm)).float()
            optimizer.step()
        elif torch.isfinite(norm):
            optimizer.step()
        scheduler.step()
        step += 1
        pending.append(torch.stack([value.detach().float() for value in losses.values()]))
        if step % args.log_every == 0:
            means = torch.stack(pending).mean(0).cpu().tolist()
            pending = []
            elapsed = time.time() - last_time
            last_time = time.time()
            row = {'step': step, 'lr': scheduler.get_last_lr()[0], 'secondsPerStep': elapsed / args.log_every,
                   **dict(zip(losses.keys(), means))}
            say(' '.join(f'{key} {value:.4g}' if isinstance(value, float) else f'{key} {value}' for key, value in row.items()))
            log.write(json.dumps({'train': row}) + '\n')
            log.flush()
        if step % settings['valEvery'] == 0 or step == steps:
            result = validate_student(model, targets, val_indices, device, column_mask)
            result['total'] = result['frames'] + settings['durationWeight'] * (result['durPhone'] + result['durPause'] + result['durPaused'])
            say('  VAL ' + ' '.join(f'{key} {value:.4g}' for key, value in result.items()))
            log.write(json.dumps({'val': {'step': step, **result}}) + '\n')
            log.flush()
            state = checkpoint_state(model, teacher_state, step, args_dict, result, {'distill': {'student': student_cfg, 'targets': targets.directory}})
            if result['total'] < best:
                best = result['total']
                torch.save(state, os.path.join(directory, 'best.pt'))
            torch.save(state, os.path.join(directory, 'last.pt'))
            log_line('students', f'{name}: {step}/{steps} steps, val frames {result["frames"]:.4f} durPhone {result["durPhone"]:.4f} '
                     f'durations off by {result["durationErrorFrames"]:.2f} frames ({(time.time() - start_time) / 60:.1f} min)')
            if args.keep_every and step % args.keep_every == 0:
                torch.save(state, os.path.join(directory, f'ckpt-{step}.pt'))
    prefetcher.close()
    log.close()
    minutes = (time.time() - start_time) / 60
    say(f'done: {steps} steps in {minutes:.1f} min; best val total {best:.4f}')
    log_line('students', f'{name}: trained - {count} parameters, {steps} steps in {minutes:.1f} min, best val total {best:.4f}')
    train_log.close()
    return 0


#
# Polish - a short fine-tune with the real-audio losses (nn/trainmulti.py)
#

class Tee:
    """A stream that writes to two streams"""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def write(self, text):
        self.first.write(text)
        self.second.write(text)
        return len(text)

    def flush(self):
        self.first.flush()
        self.second.flush()


def polish(args):
    """The polish subcommand - nn/trainmulti.py from a distilled student, its critic from the teacher's"""
    from . import trainmulti  # pylint: disable=import-outside-toplevel
    config = load_config(args.config)
    settings = {**config['polish'], **({'steps': args.steps} if args.steps else {})}
    init = args.init or os.path.join(RUNS_DIR, f'distill-{args.student}', 'best.pt')
    state = torch.load(init, map_location='cpu', weights_only=False)
    teacher = root_path(args.teacher) if args.teacher else None
    if teacher is None:
        with open(os.path.join(state['distill']['targets'], 'index.json'), encoding='utf-8') as file:
            teacher = json.load(file)['teacher']
    name = args.name or f'{args.student}-polish'
    directory = os.path.join(RUNS_DIR, f'distill-{name}')
    os.makedirs(directory, exist_ok=True)
    student_cfg = {key: value for key, value in state['config'].items() if key != 'kind'}
    argv = ['--name', f'distill-{name}', '--init', init, '--init-critic', teacher, '--config', json.dumps(student_cfg),
            '--steps', str(settings['steps']), '--lr', str(settings['lr']), '--warmup', str(settings['warmup']),
            '--cremad', str(settings['cremad']), '--adversarial', str(settings['adversarial']),
            '--feature-matching', str(settings['featureMatching']), '--adversarial-start', str(settings['adversarialStart']),
            '--guide-steps', str(settings['guideSteps']), '--val-every', str(settings['valEvery']), '--keep-every', '0']
    if args.device:
        argv += ['--device', args.device]
    log_line('students', f'{name}: polishing {init} - {settings["steps"]} steps of the real-audio losses')
    with open(os.path.join(directory, 'train.log'), 'a', encoding='utf-8') as train_log:
        stderr = sys.stderr
        sys.stderr = Tee(stderr, train_log)
        try:
            start = time.time()
            trainmulti.main(argv)
        finally:
            sys.stderr = stderr
    # (the polished checkpoints keep the distillation's record)
    for checkpoint in ('best.pt', 'last.pt'):
        path = os.path.join(directory, checkpoint)
        polished = torch.load(path, map_location='cpu', weights_only=False)
        polished['distill'] = {**state.get('distill', {}), 'polishedFrom': init}
        torch.save(polished, path)
    log_line('students', f'{name}: polished - {settings["steps"]} steps in {(time.time() - start) / 60:.1f} min')
    return 0


#
# Evaluation
#

# Score WAV files with the listener panel on the CPU (UTMOS and the ECAPA embedding) - {path: {mos, embedding}}
def listen(paths):
    process = subprocess.run([LISTENERS_PYTHON, '-I', os.path.join(ROOT, 'tools', 'listeners.py'), '--device', 'cpu', '--no-emotion',
                              '--files-from', '-'], input='\n'.join(paths), capture_output=True, text=True, check=True)
    return {item['file']: item for item in (json.loads(line) for line in process.stdout.splitlines() if line.strip())}


# Transcribe a WAV file with whisper.cpp base.en (cached beside it as <name>.json) - the text
def transcribe(path):
    base = path[:-4]
    if not os.path.exists(base + '.json') or os.path.getmtime(base + '.json') < os.path.getmtime(path):
        subprocess.run(['whisper-cli', '-m', WHISPER_MODEL, '-t', '2', '-f', path, '-np', '-ojf', '-of', base], check=True,
                       capture_output=True, cwd=ROOT)
    with open(base + '.json', encoding='utf-8') as file:
        result = json.load(file)
    return ''.join(segment['text'] for segment in result['transcription']).strip()


# Words for scoring (tools/eval.js words)
def words(text):
    return [word for word in re.sub(r'[^a-z0-9\' ]+', ' ', text.lower()).replace("'", '').split() if word]


# Word-level edit distance
def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for ix, ref_word in enumerate(reference, 1):
        current = [ix] + [0] * len(hypothesis)
        for jx, hyp_word in enumerate(hypothesis, 1):
            current[jx] = min(previous[jx] + 1, current[jx - 1] + 1, previous[jx - 1] + (ref_word != hyp_word))
        previous = current
    return previous[-1]


def cosine(first, second):
    first, second = numpy.asarray(first), numpy.asarray(second)
    return float(first @ second / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))


# The panel texts - [{id, text}]
def panel_texts():
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        return json.load(file)['texts']


# Speak a batch with an export (nn/js/speak.js) - its speed report
def speak(model_path, batch, batch_path, rate, frames_only=False):
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    command = ['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', model_path, '--batch', batch_path, '-r', str(rate)]
    if frames_only:
        command.append('--frames-only')
    process = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(process.stdout.strip().splitlines()[-1])


# A model's checkpoint by name - teacher (the targets' teacher), a ladder student (its run's best.pt), or a path
def model_checkpoint(name, config):
    if name == 'teacher':
        targets_index = os.path.join(DISTILL_DIR, 'targets', 'index.json')
        if os.path.exists(targets_index):
            with open(targets_index, encoding='utf-8') as file:
                return json.load(file)['teacher']
        return root_path(config['teacher'])
    if os.path.exists(name):
        return name
    return os.path.join(RUNS_DIR, f'distill-{name}', 'best.pt')


def evaluate_model(name, checkpoint, config, reference=None, listen_panel=True):
    """Export, package, speak, and score a model - its result (also <eval>/<name>/result.json); reference is the
    teacher's result (the panel renderings' embeddings for the similarity)"""
    from .package import package  # pylint: disable=import-outside-toplevel
    settings = config['eval']
    out_dir = os.path.join(DISTILL_DIR, 'eval', name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    start = time.time()
    model, state = load_model(checkpoint)
    result = {'name': name, 'checkpoint': checkpoint, 'step': state.get('step'),
              'config': {key: model.config[key] for key in ('dim', 'decoderDim', 'encoderLayers', 'decoderLayers', 'heads', 'ffnMultiple')},
              'parameters': parameter_count(model),
              'parametersCore': parameter_count(model) - model.speaker.weight.numel() - model.eq_free.numel()}
    manifest = export(model, state, 'model', out_dir, speaker=state['speakers'].index('arctic-bdl'))
    _, sizes = package(os.path.join(out_dir, 'model'))
    result['size'] = {'binBytes': sizes['binBytes'], 'manifestBytes': sizes['manifestBytes'], 'encoding': sizes['chosen'],
                      'moduleBytes': sizes[sizes['chosen']]['bytes'], 'moduleGzipBytes': sizes[sizes['chosen']]['gzipBytes']}
    model_path = os.path.join(out_dir, 'model')
    keys = [speaker['key'] for speaker in manifest['speakers']]
    texts = panel_texts()
    if listen_panel:
        items = []
        for speaker, sex in settings['speakers']:
            for number, text in enumerate(texts):
                items.append({'speaker': speaker, 'sex': sex, 'textId': text['id'], 'text': text['text'],
                              'wav': os.path.join(out_dir, f'{speaker}-{text["id"]}.wav'), 'seed': number + 1})
        batch = [{'text': item['text'], 'wav': item['wav'], 'seed': item['seed'], 'speaker': keys.index(item['speaker'])} for item in items]
        result['panelSpeed'] = speak(model_path, batch, os.path.join(out_dir, 'batch.json'), settings['rate'])
        paths = [item['wav'] for item in items]
        scores = listen(paths)
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            transcripts = dict(zip(paths, pool.map(transcribe, paths)))
        speakers = {}
        for speaker, sex in settings['speakers']:
            group = [item for item in items if item['speaker'] == speaker]
            mos = [scores[item['wav']]['mos'] for item in group]
            errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in group)
            count = sum(len(words(item['text'])) for item in group)
            embedding = numpy.mean([scores[item['wav']]['embedding'] for item in group], axis=0)
            row = {'sex': sex, 'mos': float(numpy.mean(mos)), 'mosMin': float(numpy.min(mos)), 'wer': 100 * errors / max(1, count),
                   'embedding': embedding.tolist()}
            if reference is not None:
                teacher_row = reference['speakers'][speaker]
                row['similarity'] = cosine(embedding, teacher_row['embedding'])
                row['pairedSimilarity'] = float(numpy.mean([
                    cosine(scores[item['wav']]['embedding'], reference['utterances'][os.path.basename(item['wav'])]['embedding'])
                    for item in group]))
            speakers[speaker] = row
        result['speakers'] = speakers
        result['utterances'] = {os.path.basename(item['wav']): {'mos': scores[item['wav']]['mos'], 'embedding': scores[item['wav']]['embedding'],
                                                                'transcript': transcripts[item['wav']]} for item in items}
        result['panel'] = {key: float(numpy.mean([row[key] for row in speakers.values()]))
                           for key in ('mos', 'wer', 'similarity', 'pairedSimilarity') if all(key in row for row in speakers.values())}
        if reference is not None:
            result['panel']['similarityMin'] = float(min(row['similarity'] for row in speakers.values()))
        for item in items:
            os.remove(item['wav'])
    result['seconds'] = round(time.time() - start, 1)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    return result


def measure_speed(names, config, repeats=3):
    """The real-time factors of the exported models (nn/js/speak.js at the samples' rate on the panel texts for the
    app's male voice) - the best of repeats, interleaved - {name: {realTimeNetwork, realTimeTotal, ...}}"""
    rate = config['samples']['rate']
    texts = panel_texts()
    best = {}
    for _ in range(repeats):
        for name in names:
            out_dir = os.path.join(DISTILL_DIR, 'eval', name)
            with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as file:
                keys = [speaker['key'] for speaker in json.load(file)['speakers']]
            batch = [{'text': text['text'], 'wav': os.path.join(out_dir, f'speed-{number:02d}.wav'), 'seed': number + 1,
                      'speaker': keys.index('arctic-bdl')} for number, text in enumerate(texts)]
            report = speak(os.path.join(out_dir, 'model'), batch, os.path.join(out_dir, 'speed-batch.json'), rate)
            for item in batch:
                os.remove(item['wav'])
            previous = best.get(name)
            if previous is None or report['networkSeconds'] < previous['networkSeconds']:
                best[name] = {**report, 'renderSeconds': min(report['renderSeconds'], previous['renderSeconds']) if previous else report['renderSeconds']}
            elif report['renderSeconds'] < previous['renderSeconds']:
                previous['renderSeconds'] = report['renderSeconds']
    for report in best.values():
        report['realTimeNetwork'] = report['audioSeconds'] / report['networkSeconds']
        report['realTimeTotal'] = report['audioSeconds'] / (report['networkSeconds'] + report['renderSeconds'])
    return best


def render_samples(name, config):
    """The listening samples of an evaluated model (its export) - flat WAV files in SAMPLES_DIR"""
    settings = config['samples']
    out_dir = os.path.join(DISTILL_DIR, 'eval', name)
    with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as file:
        keys = [speaker['key'] for speaker in json.load(file)['speakers']]
    texts = {text['id']: text['text'] for text in panel_texts()}
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    batch = [{'text': texts[text_id], 'wav': os.path.join(SAMPLES_DIR, f'distill-{name}-{speaker}-{text_id}.wav'), 'seed': 1,
              'speaker': keys.index(speaker)} for speaker in settings['speakers'] for text_id in settings['texts']]
    speak(os.path.join(out_dir, 'model'), batch, os.path.join(out_dir, 'samples-batch.json'), settings['rate'])
    return [item['wav'] for item in batch]


# A result without its bulky parts (embeddings, utterances)
def brief(result):
    out = {key: value for key, value in result.items() if key not in ('utterances', 'speakers')}
    if 'speakers' in result:
        out['speakers'] = {speaker: {key: value for key, value in row.items() if key != 'embedding'} for speaker, row in result['speakers'].items()}
    return out


def evaluate(args):
    """The eval subcommand - each model through the browser path, against the teacher"""
    config = load_config(args.config)
    names = args.models or ['teacher'] + [student['name'] for student in config['students']]
    reference_path = os.path.join(DISTILL_DIR, 'eval', 'teacher', 'result.json')
    reference = None
    if 'teacher' in names:
        names = ['teacher'] + [name for name in names if name != 'teacher']
    for number, name in enumerate(names, 1):
        if name != 'teacher' and reference is None:
            with open(reference_path, encoding='utf-8') as file:
                reference = json.load(file)
        checkpoint = model_checkpoint(name, config)
        log_line('eval', f'{number}/{len(names)} {name}: evaluating {checkpoint}')
        result = evaluate_model(name, checkpoint, config, None if name == 'teacher' else reference)
        if name == 'teacher':
            reference = result
        panel = result.get('panel', {})
        log_line('eval', f'{number}/{len(names)} {name}: {result["parameters"]} parameters, module {result["size"]["moduleBytes"]} bytes '
                 f'({result["size"]["moduleGzipBytes"]} gzip); UTMOS {panel.get("mos", 0):.3f}, WER {panel.get("wer", 0):.2f}%, '
                 f'similarity {panel.get("similarity", 1):.3f} ({result["seconds"]:.0f} s)')
    return 0


def pick(args):
    """The pick subcommand - the speed of every evaluated model, the samples, and the smallest student within the
    tolerance - DISTILL_DIR/summary.json"""
    config = load_config(args.config)
    tolerance = config['tolerance']
    eval_dir = os.path.join(DISTILL_DIR, 'eval')
    results = {}
    for name in sorted(os.listdir(eval_dir)):
        path = os.path.join(eval_dir, name, 'result.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as file:
                results[name] = json.load(file)
    teacher = results['teacher']
    if not args.no_speed:
        log_line('eval', f'speed: {len(results)} models, best of {args.speed_repeats}')
        for name, report in measure_speed(list(results), config, args.speed_repeats).items():
            results[name]['speed'] = report
    rows = []
    for name, result in results.items():
        panel = result['panel']
        row = {'name': name, 'parameters': result['parameters'], 'moduleBytes': result['size']['moduleBytes'],
               'moduleGzipBytes': result['size']['moduleGzipBytes'], 'mos': panel['mos'], 'wer': panel['wer'],
               'similarity': panel.get('similarity', 1.0), 'pairedSimilarity': panel.get('pairedSimilarity', 1.0),
               'realTimeNetwork': result.get('speed', {}).get('realTimeNetwork'), 'realTimeTotal': result.get('speed', {}).get('realTimeTotal')}
        if name != 'teacher':
            row['within'] = {'mos': panel['mos'] >= teacher['panel']['mos'] - tolerance['mos'],
                             'wer': panel['wer'] <= teacher['panel']['wer'] + tolerance['wer'],
                             'similarity': panel['similarity'] >= tolerance['similarity']}
            row['withinTolerance'] = all(row['within'].values())
        rows.append(row)
    rows.sort(key=lambda row: row['parameters'], reverse=True)
    passing = [row for row in rows if row.get('withinTolerance')]
    chosen = min(passing, key=lambda row: (row['moduleBytes'], row['parameters'])) if passing else None
    summary = {'teacher': teacher['checkpoint'], 'tolerance': tolerance, 'rows': rows, 'chosen': chosen and chosen['name'],
               'results': {name: brief(result) for name, result in results.items()}}
    samples = []
    if not args.no_samples:
        sample_names = ['teacher'] + (args.samples or [row['name'] for row in sorted(
            (row for row in rows if row['name'] != 'teacher'), key=lambda row: -row['mos'])[:3]])
        for name in sample_names:
            samples.extend(render_samples(name, config))
        summary['samples'] = samples
    with open(os.path.join(DISTILL_DIR, 'summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=1)
    for row in rows:
        log_line('eval', f'{row["name"]}: {row["parameters"]} parameters, {row["moduleBytes"]} bytes ({row["moduleGzipBytes"]} gzip), '
                 f'UTMOS {row["mos"]:.3f}, WER {row["wer"]:.2f}%, similarity {row["similarity"]:.3f}, real time '
                 f'{row["realTimeNetwork"] or 0:.0f}x network / {row["realTimeTotal"] or 0:.1f}x total'
                 + ('' if row['name'] == 'teacher' else f', within tolerance: {"yes" if row["withinTolerance"] else "no"}'))
    log_line('eval', f'done: chosen {chosen["name"] if chosen else "none"}' + (
        f' - {chosen["parameters"]} parameters, {chosen["moduleBytes"]} bytes ({chosen["moduleGzipBytes"]} gzip), UTMOS {chosen["mos"]:.3f} '
        f'(teacher {teacher["panel"]["mos"]:.3f}), WER {chosen["wer"]:.2f}% (teacher {teacher["panel"]["wer"]:.2f}%), similarity '
        f'{chosen["similarity"]:.3f}' if chosen else ' - no student is within the tolerance'))
    return 0


def ladder(args):
    """The ladder subcommand - targets (unless present), each student, the evaluation, and the pick"""
    config = load_config(args.config)
    common = ['--config', args.config] + (['--device', args.device] if args.device else [])
    if not os.path.exists(os.path.join(DISTILL_DIR, 'targets', 'index.json')):
        main(['targets'] + common)
    names = []
    for student in config['students']:
        if not os.path.exists(os.path.join(RUNS_DIR, f'distill-{student["name"]}', 'best.pt')):
            main(['train', '--student', student['name']] + common)
        names.append(student['name'])
    for name in args.polish or []:
        main(['polish', '--student', name] + common)
        names.append(f'{name}-polish')
    main(['eval', '--config', args.config, 'teacher'] + names)
    return main(['pick', '--config', args.config])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.distill', description='Distill small students from the multi-speaker model')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common(sub):
        sub.add_argument('--config', default=CONFIG_PATH, help='the settings (default data/nn/distill.json)')
        sub.add_argument('--device', default=None)

    sub = subparsers.add_parser('targets', help='the distillation set')
    add_common(sub)
    sub.add_argument('--teacher', default=None, help='the teacher checkpoint (default: the settings\')')
    sub.add_argument('--out', default=None, help='the output directory (default <distill>/targets)')
    sub.add_argument('--book-copies', type=int, default=None, help='each book sentence this many times (different speakers)')
    sub.add_argument('--batch', type=int, default=64)
    sub.add_argument('--batch-tokens', type=int, default=8192)
    sub.add_argument('--limit', type=int, default=None, help='only this many items (a smoke test)')
    sub.set_defaults(func=make_targets)

    sub = subparsers.add_parser('train', help='a student')
    add_common(sub)
    sub.add_argument('--student', required=True, help='the ladder student (data/nn/distill.json students)')
    sub.add_argument('--student-config', default=None, help='a model config instead of the ladder\'s (JSON)')
    sub.add_argument('--name', default=None, help='the run name (distill-<name>; default the student)')
    sub.add_argument('--targets', default=None)
    sub.add_argument('--teacher', default=None, help='the teacher (default: the targets\')')
    sub.add_argument('--steps', type=int, default=None)
    sub.add_argument('--batch', type=int, default=None)
    sub.add_argument('--lr', type=float, default=None)
    sub.add_argument('--warmup', type=int, default=None)
    sub.add_argument('--init-teacher', action='store_true', help='copy every teacher tensor of the same name and shape')
    sub.add_argument('--feature-weight', type=float, default=0.0, help='intermediate-feature matching weight (0: none)')
    sub.add_argument('--log-every', type=int, default=50)
    sub.add_argument('--keep-every', type=int, default=0, help='keep ckpt-<step>.pt every this many steps (0: none)')
    sub.add_argument('--seed', type=int, default=1)
    sub.set_defaults(func=train_student)

    sub = subparsers.add_parser('polish', help='a short fine-tune of a student with the real-audio losses')
    add_common(sub)
    sub.add_argument('--student', required=True)
    sub.add_argument('--init', default=None, help='the student checkpoint (default its run\'s best.pt)')
    sub.add_argument('--teacher', default=None, help='the critic\'s source (default the targets\' teacher)')
    sub.add_argument('--name', default=None, help='the run name (distill-<name>; default <student>-polish)')
    sub.add_argument('--steps', type=int, default=None)
    sub.set_defaults(func=polish)

    sub = subparsers.add_parser('eval', help='evaluate models through the browser path')
    add_common(sub)
    sub.add_argument('models', nargs='*', help='teacher, ladder students or runs (distill-<name>/best.pt), or checkpoints')
    sub.set_defaults(func=evaluate)

    sub = subparsers.add_parser('pick', help='speed, samples, and the smallest student within the tolerance')
    add_common(sub)
    sub.add_argument('--no-speed', action='store_true')
    sub.add_argument('--speed-repeats', type=int, default=3)
    sub.add_argument('--no-samples', action='store_true')
    sub.add_argument('--samples', nargs='*', default=None, help='the students to render samples of (default the best three)')
    sub.set_defaults(func=pick)

    sub = subparsers.add_parser('ladder', help='targets, the students, the evaluation, and the pick')
    add_common(sub)
    sub.add_argument('--polish', nargs='*', default=None, help='students to polish too')
    sub.set_defaults(func=ladder)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
