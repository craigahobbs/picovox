# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The heteronym models - the G2P network's reading input and ambiguity head, and the context network that chooses a
reading from the neighboring words (lib/g2p.js runs both from lib/models/g2pModel.js; g2pmodel.py is the port). No
dictionary, word list, or rule at runtime: a word whose ambiguity head fires takes the reading slot the context network
chooses, and the G2P pronounces the word in that reading. The commands (python is
PYTHONPATH=train/src build/venv-train/bin/python -B):

    python -m picovox_train.context train-g2p --labels DIR/labels.json --init G2P.pt --out DIR/g2p.pt
    python -m picovox_train.context train-context --labels DIR/labels.json --out DIR/context.pt --folds DIR/folds [-j N]
    python -m picovox_train.context folds DIR/folds --labels DIR/labels.json       # the gate - the folds' held-out results
    python -m picovox_train.context export --g2p DIR/g2p.pt --context DIR/context.pt [--out lib/models/g2pModel.js]
    python -m picovox_train.context check NEW.js     # make train's G2P gate: identical to lib/models/g2pModel.js,
                                                     # else the quality gate (make train-install installs it - trained.py)

The settings are train/data/train.json's "context" section (g2p, network, export, gate).

The G2P additions (g2p.py's model):

- the reading input - a learned embedding per alternate reading (1 to READINGS - 1), added to the decoder's initial state
  (reading 0, the default, adds nothing - the network's default pronunciations are unchanged by the addition)
- the ambiguity head - one logit from the encoder's final states: positive for a word with 2+ readings that the
  neighboring words decide (the labels' words, contextlabels.py)
- the spelled head - two more logits from the encoder's final states: positive for a word that, written in capitals, is
  spelled - its letters' names: alone among lowercase words, and beside a word in capitals (spelled.py's labels)

Their training continues a trained G2P: its own data (each word's reading 0) plus each ambiguous word's alternate readings
("readingShare" of the samples), the other words with an alternate reading index and their default target
("invariantShare" - so a reading index on a word the head wrongly fires for changes nothing), and the ambiguity head's
binary cross entropy ("ambiguityWeight"). The spelled head trains beside them on its own batches (the "spelled"
settings - "batch" strings a step, drawn by its own generator; its loss times "weight", its gradient clipped alone): the
encoder is frozen, so its loss reaches nothing but the head - the rest trains exactly as without it.

The context network: each item of the window (WINDOW words and phrase-end marks each side of the word - frontend.py
context_sequence) is read by a character GRU (its last MAX_CHARS characters) to a vector (zero beyond the text); a
bidirectional GRU runs over the window's vectors; a tanh layer of its outputs at the middle, gated (multiplied) by a tanh
layer of the word's own vector - so a word can flip a shared context feature (a noun's position means reading 0 for one
word and 1 for another) - gives the reading slots' logits. Slot k is the G2P's reading k, so the network is
word-independent - the word's spelling is part of its input. Trained with cross entropy on the labels (words balanced -
a word's share of the samples grows as its examples ** "balance"; dropout, weight decay, and "itemDropout" - context
items dropped as beyond the text), then "quantSteps" of quantization-aware training at the export's bits.

Distilled from an ensemble, for steadiness: a single network's choices in contexts its labels don't cover depend on its
seed (five seeds of the same data differed on 14% of unlabeled book windows and 9 of the regression set's cases a
pair). So "teachers" networks are trained first (each its own seed - 100 * seed + 1, 2, ...; quantized as exported),
and the network (the same size - the only one exported) is trained toward their mean probabilities ("distill" of the
loss, the rest the labels' cross entropy) on its examples and on the labels' transfer windows (unlabeled - the ambiguous
words' other windows in the books, contextlabels.py), "transferShare" of a batch's size added to each batch. The
cross-validation's other folds (the gate's held-out accuracy - context.py folds) train one plain network each.
"""

import argparse
import base64
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
import time
import zlib

import numpy

from . import RECIPE_PATH, ROOT, g2p, g2pmodel, read_json
from . import spelled as spelled_module
from .corpora import file_sha256
from .trained import module_body
from .g2p import (EOS, LETTERS, MAX_WORD, MODULE_LINE_LENGTH, PHONES, UPSAMPLE, build_model, learning_rate, log_line, make_batch,
                  model_config, predict, quantize_array, quantize_state, weight_bytes)
from .readings import sz_only


# The recipe's settings - train/data/train.json's "context" (labels, g2p, network, export, gate)
RECIPE = read_json(RECIPE_PATH)['context']

# The reading slots (the default and two alternates), the ambiguity head's hidden layer, and the spelled head's settings
# (its hidden layer, its loss's weight, its batch, and its labels' and their sampling's - spelled.py)
READINGS = RECIPE['g2p']['readings']
AMBIGUITY_HIDDEN = RECIPE['g2p']['ambiguityHidden']
SPELLED = RECIPE['g2p']['spelled']

# The context network's input - its characters (id 0 is padding), the characters of an item it reads, and the window
CONTEXT_CHARS = '\'abcdefghijklmnopqrstuvwxyz,.?!'
CONTEXT_CHAR_IDS = {char: index + 1 for index, char in enumerate(CONTEXT_CHARS)}
MAX_CHARS = RECIPE['network']['maxChars']
WINDOW = RECIPE['network']['window']

# The export - each model's matrices' bits (per-row bfloat16 scales) and its vectors' bits
EXPORT = RECIPE['export']

# The context network's settings, and the G2P reading stage's
CONTEXT_SETTINGS = {key: value for key, value in RECIPE['network'].items() if key not in ('window', 'maxChars')}
G2P_SETTINGS = {key: value for key, value in RECIPE['g2p'].items() if key not in ('readings', 'ambiguityHidden', 'spelled')}

# The reading stage's new tensors (their learning rate is the stage's times headRate), and the spelled head's
SPELLED_HEAD = ('spelled_layer', 'spelled')
HEADS = ('reading', 'ambiguity_layer', 'ambiguity', *SPELLED_HEAD)


#
# The G2P's reading input and ambiguity head
#

def add_reading_heads(model, readings=READINGS):
    """Add the reading embedding, the ambiguity head, and the spelled head to a G2P model (g2p.py build_model)"""
    from torch import nn  # pylint: disable=import-outside-toplevel
    model.reading = nn.Embedding(readings - 1, model.config['decoder'])
    nn.init.normal_(model.reading.weight, std=1.0)
    model.ambiguity_layer = nn.Linear(2 * model.config['hidden'], AMBIGUITY_HIDDEN)
    model.ambiguity = nn.Linear(AMBIGUITY_HIDDEN, 1)
    nn.init.constant_(model.ambiguity.bias, -4.0)
    model.spelled_layer = nn.Linear(2 * model.config['hidden'], int(SPELLED['hidden']))
    model.spelled = nn.Linear(int(SPELLED['hidden']), 2)
    model.config = {**model.config, 'readings': readings}
    return model


# A G2P model from a checkpoint (with the reading heads if it has them)
def load_g2p(path):
    import torch  # pylint: disable=import-outside-toplevel
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['config'])
    if 'reading.weight' in checkpoint['state']:
        add_reading_heads(model, checkpoint['config']['readings'])
    model.load_state_dict(checkpoint['state'])
    return model


# The ambiguity head's logits from the encoder's final states
def ambiguity_logit(model, ends):
    import torch  # pylint: disable=import-outside-toplevel
    return model.ambiguity(torch.tanh(model.ambiguity_layer(ends))).squeeze(-1)


# The spelled head's logits from the encoder's final states - [batch, 2]: alone, and beside a word in capitals
def spelled_logit(model, ends):
    import torch  # pylint: disable=import-outside-toplevel
    return model.spelled(torch.tanh(model.spelled_layer(ends)))


# The encoder's final states of words (no gradient - the heads' input)
def encoder_ends(model, words):
    import torch  # pylint: disable=import-outside-toplevel
    with torch.no_grad():
        return model.encode(make_batch(words, [[] for _ in words])[0])[3]


# The spelled head's decisions of words - whether each, in capitals, is spelled: [alone, beside a word in capitals]
def spelled_decisions(model, words, batch_size=2048):
    model.eval()
    decisions = []
    for start in range(0, len(words), batch_size):
        decisions.extend((spelled_logit(model, encoder_ends(model, words[start:start + batch_size])) > 0).tolist())
    return decisions


# The loss of a batch - teacher-forced cross entropy of the readings' targets (g2p.py's), plus the ambiguity head's
def reading_loss(model, letters, targets, readings, ambiguous, ambiguity_weight):
    from torch import nn  # pylint: disable=import-outside-toplevel
    phone_loss, ends = model.loss(letters, targets, readings)
    ambiguity_loss = nn.functional.binary_cross_entropy_with_logits(ambiguity_logit(model, ends), ambiguous)
    return phone_loss + ambiguity_weight * ambiguity_loss, phone_loss, ambiguity_loss


# The ambiguity head's logits of words
def ambiguity_logits(model, words, batch_size=2048):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            chunk = words[start:start + batch_size]
            letters = make_batch(chunk, [[] for _ in chunk])[0]
            _, _, _, ends = model.encode(letters)
            logits.extend(ambiguity_logit(model, ends).tolist())
    return logits


# The reading targets of the ambiguous words - [(word, phones, reading)] for each reading (its first variant; the default
# reading's is the G2P's own target, given as defaults {word: phones})
def reading_targets(labels, defaults=None):
    targets = []
    for word, entry in sorted(labels['words'].items()):
        for reading, variants in enumerate(entry['readings'][:READINGS]):
            if reading == 0 and defaults is not None:
                targets.append((word, defaults.get(word, variants[0].split()), 0))
            elif reading > 0 or defaults is not None:
                targets.append((word, variants[0].split(), reading))
    return targets


def train_g2p(model, items, probabilities, labels, spelled, settings, log):
    """Train the reading input, the ambiguity head, and the spelled head (and the rest) from a trained G2P - items [(word,
    phones)] and their sampling probabilities are the G2P's own data (reading 0), spelled the spelled head's labels
    (spelled.py spelled_labels); returns the model"""
    import torch  # pylint: disable=import-outside-toplevel
    settings = {**G2P_SETTINGS, **settings}
    torch.set_num_threads(int(settings['threads']))
    torch.manual_seed(int(settings['seed']))
    rng = numpy.random.default_rng(int(settings['seed']))
    spelled_rng = numpy.random.default_rng([int(settings['seed']), 1])
    spelled_sample = spelled_module.batch_sampler(spelled)
    if not hasattr(model, 'reading'):
        add_reading_heads(model)
    ambiguous = set(labels['words'])
    targets = reading_targets(labels, {word: phones for word, phones in items if word in ambiguous})
    invariant = numpy.array([0.0 if word in ambiguous else probability for (word, _), probability in zip(items, probabilities)])
    invariant /= invariant.sum()
    reading_share, invariant_share = float(settings['readingShare']), float(settings['invariantShare'])
    steps, batch_size, peak, warmup = int(settings['steps']), int(settings['batch']), float(settings['lr']), int(settings['warmup'])
    trained = [parameter for name, parameter in model.named_parameters() if name.split('.')[0] in settings['train'].split(',')]
    for parameter in model.parameters():
        parameter.requires_grad_(any(parameter is other for other in trained))
    head_rate = float(settings['headRate'])
    heads = [parameter for name, parameter in model.named_parameters() if name.split('.')[0] in HEADS]
    spelled_head = [parameter for name, parameter in model.named_parameters() if name.split('.')[0] in SPELLED_HEAD]
    others = [parameter for parameter in trained if not any(parameter is head for head in spelled_head)]
    optimizer = torch.optim.AdamW([
        {'params': [parameter for parameter in trained if not any(parameter is head for head in heads)], 'scale': 1.0},
        {'params': [parameter for parameter in trained if any(parameter is head for head in heads)], 'scale': head_rate}
    ], lr=peak, weight_decay=0.0)
    log(f'train-g2p: {sum(parameter.numel() for parameter in trained)} parameters trained ({settings["train"]}), {len(items)} '
        f'items, {len(ambiguous)} ambiguous words, {len(targets)} reading targets, {steps} steps')
    window = collections.defaultdict(float)
    window_start = time.time()
    batches = []
    model.train()
    for step in range(1, steps + 1):
        if not batches:
            # The next pool of samples - (word, phones, reading), sorted by length and shuffled as batches
            count = batch_size * 32
            kinds = rng.choice(3, size=count, p=[1 - reading_share - invariant_share, reading_share, invariant_share])
            pool = []
            base = rng.choice(len(items), size=count, p=probabilities)
            invariants = rng.choice(len(items), size=count, p=invariant)
            alternates = rng.integers(len(targets), size=count)
            for index in range(count):
                if kinds[index] == 0:
                    word, phones = items[base[index]]
                    pool.append((word, phones, 0))
                elif kinds[index] == 1:
                    pool.append(targets[alternates[index]])
                else:
                    word, phones = items[invariants[index]]
                    pool.append((word, phones, int(rng.integers(1, READINGS))))
            pool.sort(key=lambda sample: (len(sample[0]), len(sample[1])))
            batches = [pool[start:start + batch_size] for start in range(0, len(pool), batch_size)]
            rng.shuffle(batches)
        batch = batches.pop()
        letters, phone_targets = make_batch([word for word, _, _ in batch], [phones for _, phones, _ in batch])
        readings = torch.tensor([reading for _, _, reading in batch], dtype=torch.long)
        labels_tensor = torch.tensor([1.0 if word in ambiguous else 0.0 for word, _, _ in batch])
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, steps, peak, warmup) * group['scale']
        loss, phone_loss, ambiguity_loss = reading_loss(model, letters, phone_targets, readings, labels_tensor,
                                                        float(settings['ambiguityWeight']))

        # The spelled head's batch - its own strings, the encoder's final states of them fixed
        spelled_words, spelled_targets = spelled_sample(spelled_rng, int(SPELLED['batch']))
        spelled_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            spelled_logit(model, encoder_ends(model, spelled_words)), torch.tensor(spelled_targets))
        optimizer.zero_grad()
        (loss + float(SPELLED['weight']) * spelled_loss).backward()
        torch.nn.utils.clip_grad_norm_(others, 1.0)
        torch.nn.utils.clip_grad_norm_(spelled_head, 1.0)
        optimizer.step()
        window['phone'] += phone_loss.item()
        window['ambiguity'] += ambiguity_loss.item()
        window['spelled'] += spelled_loss.item()
        if step % 500 == 0 or step == steps:
            count = step % 500 or 500
            log(f'train-g2p: step {step}/{steps} phone loss {window["phone"] / count:.4f}, ambiguity loss '
                f'{window["ambiguity"] / count:.5f}, spelled loss {window["spelled"] / count:.5f} '
                f'({(time.time() - window_start) / count * 1000:.0f} ms/step)')
            window.clear()
            window_start = time.time()
    model.eval()
    return model


def evaluate_g2p(model, labels, words, spelled):
    """The reading measures of a G2P - the alternate readings right, the ambiguity head's errors over words (the
    dictionary and labeled words), and the spelled head's errors over its labels (spelled.py spelled_errors) -
    {readingsRight, readings, ambiguousMissed, falsePositives, spelled}"""
    targets = reading_targets(labels, {})
    predictions = predict(model, [word for word, _, _ in targets], [reading for _, _, reading in targets])
    right = sum(' '.join(predicted) in labels['words'][word]['readings'][reading]
                for (word, _, reading), predicted in zip(targets, predictions))
    ambiguous = set(labels['words'])
    words = sorted(set(words) | ambiguous)
    logits = ambiguity_logits(model, words)
    missed = [word for word, logit in zip(words, logits) if word in ambiguous and logit <= 0]
    false_positives = sum(1 for word, logit in zip(words, logits) if word not in ambiguous and logit > 0)
    return {'readingsRight': right, 'readings': len(targets), 'ambiguousMissed': missed, 'falsePositives': false_positives,
            'spelled': spelled_module.spelled_errors(spelled, lambda spelled_words: spelled_decisions(model, spelled_words))}


#
# The context network
#

# An item's character ids - its last MAX_CHARS characters
def item_chars(item):
    return [CONTEXT_CHAR_IDS[char] for char in item if char in CONTEXT_CHAR_IDS][-MAX_CHARS:]


def build_context(config, slots=READINGS):
    """The context network (see the module's description)"""
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn  # pylint: disable=import-outside-toplevel

    class ContextModel(nn.Module):
        """The context network"""

        def __init__(self):
            super().__init__()
            self.config = {'embed': int(config['embed']), 'hidden': int(config['hidden']), 'layer': int(config['layer']),
                           'slots': slots}
            embed, hidden, layer = self.config['embed'], self.config['hidden'], self.config['layer']
            self.char_embed = nn.Embedding(len(CONTEXT_CHARS) + 1, embed, padding_idx=0)
            self.char_gru = nn.GRU(embed, hidden, batch_first=True)
            self.window_gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            self.layer = nn.Linear(2 * hidden, layer)
            self.word_layer = nn.Linear(hidden, layer)
            self.output = nn.Linear(layer, slots)
            self.dropout = nn.Dropout(float(config.get('dropout', 0.0)))

        # The items' vectors - chars [items, MAX_CHARS] (padded), lengths [items] - [items, hidden] (zero for length 0)
        def item_vectors(self, chars, lengths):
            vectors = torch.zeros(chars.shape[0], self.config['hidden'])
            present = lengths > 0
            if present.any():
                packed = nn.utils.rnn.pack_padded_sequence(self.char_embed(chars[present]), lengths[present].cpu(), batch_first=True,
                                                           enforce_sorted=False)
                _, final = self.char_gru(packed)
                vectors[present] = final[0]
            return vectors

        # The slots' logits of windows - chars [batch, window, MAX_CHARS], lengths [batch, window]
        def forward(self, chars, lengths):
            batch, width = lengths.shape
            vectors = self.item_vectors(chars.reshape(batch * width, -1), lengths.reshape(-1)).reshape(batch, width, -1)
            outputs, _ = self.window_gru(vectors)
            middle = width // 2
            gated = torch.tanh(self.layer(outputs[:, middle])) * torch.tanh(self.word_layer(vectors[:, middle]))
            return self.output(self.dropout(gated))

    return ContextModel()


# The windows' character arrays - (chars [count, window, MAX_CHARS], lengths [count, window]) as int64 arrays
def window_arrays(windows):
    width = 2 * WINDOW + 1
    chars = numpy.zeros((len(windows), width, MAX_CHARS), dtype=numpy.int64)
    lengths = numpy.zeros((len(windows), width), dtype=numpy.int64)
    for index, window in enumerate(windows):
        for position, item in enumerate(window):
            ids = item_chars(item)
            chars[index, position, :len(ids)] = ids
            lengths[index, position] = len(ids)
    return chars, lengths


# The context network's slots for windows
def context_predict(model, windows, batch_size=1024):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    chars, lengths = window_arrays(windows)
    slots = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            logits = model(torch.from_numpy(chars[start:start + batch_size]), torch.from_numpy(lengths[start:start + batch_size]))
            slots.extend(logits.argmax(-1).tolist())
    return slots


class WeightQuantizer:
    """Quantization-aware training - quantize() replaces the matrices with their quantized values (EXPORT's bits, the
    export's rounding), restore() puts the float weights back (the gradient passes straight through)"""

    def __init__(self, model, spec):
        self.parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.dim() == 2]
        self.spec = spec
        self.saved = None

    def quantize(self):
        import torch  # pylint: disable=import-outside-toplevel
        self.saved = [parameter.data.clone() for _, parameter in self.parameters]
        for _, parameter in self.parameters:
            values = parameter.data.double().numpy()
            _, _, dequantized = quantize_array(values, self.spec['bits'])
            parameter.data.copy_(torch.from_numpy(dequantized).float())

    def restore(self):
        for (_, parameter), saved in zip(self.parameters, self.saved):
            parameter.data.copy_(saved)


# The cross-validation folds - by book (fold 0 is the labels' held-out books - contextlabels.py - so its network is the
# context network itself; a book of the others is in fold 1 + its hash's remainder by FOLDS - 1)
FOLDS = RECIPE['gate']['folds']


def fold_labels(labels, fold):
    """The labels with one fold's books' examples (every source's) held out, the others trained on - fold 0 the labels'
    own split; in the others, the held-out books' and the seeds' examples train"""
    def split(example):
        if fold == 0:
            return example['split']
        if example['split'] == 'heldout' or example['source'] == 'seed':
            return 'train'
        return 'heldout' if 1 + int(hashlib.sha1(example['group'].encode('utf-8')).hexdigest(), 16) % (FOLDS - 1) == fold else 'train'
    return {**labels, 'examples': [{**example, 'split': split(example)} for example in labels['examples']]}


# The teachers' mean reading probabilities of windows (chars, lengths - window_arrays) - [count, slots] float64
def teacher_probabilities(teachers, chars, lengths, batch_size=1024):
    import torch  # pylint: disable=import-outside-toplevel
    total = numpy.zeros((chars.shape[0], teachers[0].config['slots']))
    with torch.no_grad():
        for teacher in teachers:
            teacher.eval()
            for start in range(0, chars.shape[0], batch_size):
                logits = teacher(torch.from_numpy(chars[start:start + batch_size]), torch.from_numpy(lengths[start:start + batch_size]))
                total[start:start + batch_size] += torch.softmax(logits.double(), -1).numpy()
    return total / len(teachers)


def train_context(labels, settings, log, teachers=None):
    """Train the context network on the labels' training examples - the model (with teachers, trained toward their mean
    probabilities too - "distill" of the loss - on the examples and the labels' transfer windows)"""
    import torch  # pylint: disable=import-outside-toplevel
    settings = {**CONTEXT_SETTINGS, **settings}
    torch.set_num_threads(int(settings['threads']))
    torch.manual_seed(int(settings['seed']))
    rng = numpy.random.default_rng(int(settings['seed']))
    examples = [example for example in labels['examples'] if example['split'] == 'train']
    examples += [example for example in examples if example['source'] == 'seed'] * (int(settings['seedRepeat']) - 1)
    # Each word's share of the samples grows as its examples ** balance; within a word, the readings' shares are their
    # examples' (seeds repeated seedRepeat times)
    counts = collections.Counter(example['word'] for example in examples)
    weights = numpy.array([counts[example['word']] ** (float(settings['balance']) - 1) for example in examples])
    probabilities = weights / weights.sum()
    chars, lengths = window_arrays([example['window'] for example in examples])
    slots = numpy.array([example['slot'] for example in examples])
    soft = teacher_probabilities(teachers, chars, lengths) if teachers else None
    # The transfer windows (unlabeled - the teachers' probabilities only), "transferShare" of a batch's size added to it
    transfer = labels.get('transfer', []) if teachers else []
    transfer_count = int(int(settings['batch']) * float(settings['transferShare'])) if transfer else 0
    if transfer_count:
        transfer_chars, transfer_lengths = window_arrays(transfer)
        transfer_soft = teacher_probabilities(teachers, transfer_chars, transfer_lengths)
    distill = float(settings['distill'])
    item_dropout = float(settings['itemDropout'])
    model = build_context(settings)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    log(f'train-context: {len(examples)} examples ({len(counts)} words), {parameters} parameters'
        f'{f", {len(teachers)} teachers" if teachers else ""}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings['lr']), weight_decay=float(settings['weightDecay']))
    steps, quant_steps, batch_size = int(settings['steps']), int(settings['quantSteps']), int(settings['batch'])
    quantizer = WeightQuantizer(model, EXPORT)
    window_loss, window_start = 0.0, time.time()
    model.train()
    for step in range(1, steps + quant_steps + 1):
        quantizing = step > steps
        if quantizing:
            rate = learning_rate(step - steps, quant_steps, float(settings['quantLr']), 0)
        else:
            rate = learning_rate(step, steps, float(settings['lr']), int(settings['warmup']))
        for group in optimizer.param_groups:
            group['lr'] = rate
        batch = rng.choice(len(examples), size=batch_size, p=probabilities)
        batch_chars, batch_lengths = chars[batch], lengths[batch]
        if transfer_count:
            extra = rng.integers(len(transfer), size=transfer_count)
            batch_chars = numpy.concatenate([batch_chars, transfer_chars[extra]])
            batch_lengths = numpy.concatenate([batch_lengths, transfer_lengths[extra]])
        if item_dropout:
            # The context items dropped (as beyond the text), never the word itself
            dropped = rng.random(batch_lengths.shape) < item_dropout
            dropped[:, WINDOW] = False
            batch_lengths = numpy.where(dropped, 0, batch_lengths)
        if quantizing:
            quantizer.quantize()
        logits = model(torch.from_numpy(batch_chars), torch.from_numpy(batch_lengths))
        loss = torch.nn.functional.cross_entropy(logits[:batch_size], torch.from_numpy(slots[batch]))
        if soft is not None:
            targets = soft[batch] if not transfer_count else numpy.concatenate([soft[batch], transfer_soft[extra]])
            soft_loss = -(torch.from_numpy(targets).float() * torch.log_softmax(logits, -1)).sum(-1).mean()
            loss = (1 - distill) * loss + distill * soft_loss
        optimizer.zero_grad()
        loss.backward()
        if quantizing:
            quantizer.restore()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        window_loss += loss.item()
        if step % 1000 == 0 or step == steps + quant_steps:
            count = step % 1000 or 1000
            log(f'train-context: step {step}/{steps + quant_steps}{" (quantized)" if quantizing else ""} loss {window_loss / count:.4f} '
                f'({(time.time() - window_start) / count * 1000:.1f} ms/step)')
            window_loss, window_start = 0.0, time.time()
    model.eval()
    return model


# The labels of a fold (fold_labels)
def labels_of(labels_path, fold):
    return fold_labels(read_json(labels_path), fold)


# A fold's log lines' prefix (none for fold 0 - the context network itself)
def fold_prefix(fold):
    return f'fold {fold}: ' if fold else ''


# The teachers' seeds - "teachers" of them, 100 * seed + 1, 2, ...
def teacher_seeds(settings):
    settings = {**CONTEXT_SETTINGS, **settings}
    return [100 * int(settings['seed']) + index + 1 for index in range(int(settings['teachers']))]


# One teacher (a worker task) - (labels path, fold, seed, settings, log path): the context network trained on the fold's
# labels with that seed (no teachers of its own) - its state, quantized as exported
def teacher_task(task):
    labels_path, fold, seed, settings, log = task
    teacher = train_context(labels_of(labels_path, fold), {**settings, 'seed': seed},
                            lambda text: log_line(log, f'{fold_prefix(fold)}teacher {seed}: {text}'))
    WeightQuantizer(teacher, EXPORT).quantize()
    return teacher.state_dict()


# Worker tasks' results in order - in jobs worker processes when more than one (the same results either way)
def run_tasks(function, tasks, jobs):
    if jobs > 1 and len(tasks) > 1:
        with multiprocessing.Pool(min(jobs, len(tasks))) as pool:
            return pool.map(function, tasks)
    return [function(task) for task in tasks]


def train_student(labels, fold, settings, log, states):
    """The context network trained on a fold's labels toward the teachers' mean probabilities (teachers' states - none,
    the plain training) - the model"""
    settings = {**CONTEXT_SETTINGS, **settings}
    teachers = []
    for state in states:
        teacher = build_context(settings)
        teacher.load_state_dict(state)
        teachers.append(teacher)
    return train_context(labels, settings, lambda text: log_line(log, f'{fold_prefix(fold)}{text}'), teachers)


def evaluate_context(slot_of, labels):
    """The context choices' accuracy per word and source on the held-out examples, against each word's majority slot (in
    the training examples) - slot_of(windows) gives the chosen slots - {words: {source/word: {right, total, majority}}}"""
    examples = [example for example in labels['examples'] if example['split'] == 'heldout']
    majority = {}
    training = collections.defaultdict(collections.Counter)
    for example in labels['examples']:
        if example['split'] == 'train':
            training[example['word']][example['slot']] += 1
    for word, counter in training.items():
        majority[word] = counter.most_common(1)[0][0]
    chosen = slot_of([example['window'] for example in examples])
    words = collections.defaultdict(lambda: {'right': 0, 'total': 0, 'majority': 0})
    for example, slot in zip(examples, chosen):
        counts = words[(example['source'], example['word'])]
        counts['total'] += 1
        counts['right'] += slot == example['slot']
        counts['majority'] += majority.get(example['word'], 0) == example['slot']
    return {'words': {f'{source}/{word}': counts for (source, word), counts in sorted(words.items())}}


#
# The export - the G2P module with the reading heads and the context network
#

# The G2P's quantization calibration - {words, phones, readings, spelled}: EXPORT's "calibration" words drawn by the
# G2P's own training probabilities (a fixed seed; reading 0) and every alternate reading target, and as many of the
# spelled head's strings, drawn as its batches (the reading stage's checkpoint keeps it, so the export is the
# checkpoint's alone)
def g2p_calibration(items, probabilities, labels, spelled):
    chosen = numpy.random.default_rng(0).choice(len(items), size=int(EXPORT['calibration']), p=probabilities)
    samples = [(*items[index], 0) for index in chosen] + reading_targets(labels)
    spelled_sample = spelled_module.batch_sampler(spelled)
    return {'words': [word for word, _, _ in samples], 'phones': [list(phones) for _, phones, _ in samples],
            'readings': [reading for _, _, reading in samples],
            'spelled': spelled_sample(numpy.random.default_rng(0), int(EXPORT['calibration']))[0]}


# The G2P's matrices' input covariances (for GPTQ - quant.py) - {name: E[x x^T]} over the calibration words, teacher-
# forced: each GRU's input and its previous state (the encoder's forward and reverse), the bridge's and the ambiguity
# head's encoder ends, the decoder's attention, combine, and output inputs (the embedding tables - lookups - none);
# and the spelled head's, over its own strings
def g2p_covariances(model, calibration):
    import torch  # pylint: disable=import-outside-toplevel
    sums = collections.defaultdict(lambda: [0.0, 0])

    def add(name, inputs):
        inputs = inputs.reshape(-1, inputs.shape[-1]).double()
        sums[name][0] = sums[name][0] + (inputs.T @ inputs).numpy()
        sums[name][1] += inputs.shape[0]

    words, phones, readings = calibration['words'], calibration['phones'], calibration['readings']
    hidden = model.config['hidden']
    with torch.no_grad():
        for start in range(0, len(words), 1024):
            letters, targets = make_batch(words[start:start + 1024], phones[start:start + 1024])
            encoded, mask, state, ends = model.encode(letters, torch.tensor(readings[start:start + 1024]))
            embedded = model.letter_embed(letters)
            forward, reverse = encoded[..., :hidden], encoded[..., hidden:]
            previous_forward = torch.cat([torch.zeros_like(forward[:, :1]), forward[:, :-1]], 1)
            next_valid = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], 1)[..., None]
            next_reverse = torch.cat([reverse[:, 1:], torch.zeros_like(reverse[:, :1])], 1) * next_valid
            add('encoder.weight_ih_l0', embedded[mask])
            add('encoder.weight_ih_l0_reverse', embedded[mask])
            add('encoder.weight_hh_l0', previous_forward[mask])
            add('encoder.weight_hh_l0_reverse', next_reverse[mask])
            add('bridge.weight', ends)
            add('ambiguity_layer.weight', ends)
            add('ambiguity.weight', torch.tanh(model.ambiguity_layer(ends)))
            previous = torch.cat([torch.full_like(targets[:, :1], EOS), targets[:, :-1]], 1)
            steps = targets != 0
            phone_inputs = model.phone_embed(previous)
            decoded, _ = model.decoder(phone_inputs, state)
            add('decoder.weight_ih_l0', phone_inputs[steps])
            add('decoder.weight_hh_l0', torch.cat([state.transpose(0, 1), decoded[:, :-1]], 1)[steps])
            add('attention.weight', decoded[steps])
            scores = torch.einsum('bsh,bth->bst', model.attention(decoded), encoded).masked_fill(~mask.unsqueeze(1), float('-inf'))
            combined = torch.cat([decoded, torch.einsum('bst,bth->bsh', torch.softmax(scores, -1), encoded)], -1)
            add('combine.weight', combined[steps])
            add('output.weight', torch.tanh(model.combine(combined))[steps])
        for start in range(0, len(calibration['spelled']), 1024):
            ends = encoder_ends(model, calibration['spelled'][start:start + 1024])
            add('spelled_layer.weight', ends)
            add('spelled.weight', torch.tanh(model.spelled_layer(ends)))
    return {name: total / count for name, (total, count) in sums.items()}


# The smallest encoding of a quantized state - (manifest fields {layout, deflate}, base64 text)
def encode_weights(quantized):
    candidates = []
    for layout in ('packed', 'bytes'):
        data = weight_bytes(quantized, layout)
        compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
        deflated = compressor.compress(data) + compressor.flush()
        for deflate, payload in ((False, data), (True, deflated)):
            if layout == 'bytes' and not deflate:
                continue
            candidates.append((len(payload), layout, deflate, base64.b64encode(payload).decode('ascii')))
    _, layout, deflate, text = min(candidates, key=lambda candidate: candidate[0])
    return {'layout': layout, 'deflate': deflate}, text


# A manifest's tensors list (lines at the given indent)
def tensor_lines(quantized, indent):
    return ',\n'.join(f'{indent}{{\'name\': \'{item["name"]}\', \'shape\': {json.dumps(item["shape"])}, \'bits\': {item["bits"]}}}'
                      for item in quantized)


# Base64 text as lines of MODULE_LINE_LENGTH
def base64_lines(text):
    return [text[index:index + MODULE_LINE_LENGTH] for index in range(0, len(text), MODULE_LINE_LENGTH)]


def module_text(g2p_config, g2p_quantized, context_config, context_quantized, comment):
    """The generated module's text - g2p.py's G2P module with 'readings' and the reading tensors, and a 'context'
    section (the context network's manifest and weights)"""
    config = model_config(g2p_config)
    g2p_encoding, g2p_text = encode_weights(g2p_quantized)
    context_encoding, context_text = encode_weights(context_quantized)
    phone_lines = []
    for phone in (f'\'{phone}\'' for phone in PHONES):
        if phone_lines and 8 + len(phone_lines[-1]) + len(phone) + 3 <= MODULE_LINE_LENGTH:
            phone_lines[-1] += f', {phone}'
        else:
            phone_lines.append(phone)
    lines = [
        '// Generated by picovox_train.context export - do not edit',
        '//',
        *[f'// {line}' for line in comment],
        '',
        'export const g2pModel = {',
        '    \'version\': 1,',
        f'    \'design\': \'{config["design"]}\',',
        '    \'letters\': "' + ''.join(LETTERS[1:]) + '",',
        '    \'phones\': [',
        ',\n'.join(f'        {line}' for line in phone_lines),
        '    ],',
        f'    \'embed\': {config["embed"]},',
        f'    \'hidden\': {config["hidden"]},',
        f'    \'decoder\': {config["decoder"]},',
        f'    \'upsample\': {UPSAMPLE},',
        f'    \'maxLetters\': {MAX_WORD},',
        f'    \'readings\': {g2p_config["readings"]},',
        f'    \'layout\': \'{g2p_encoding["layout"]}\',',
        f'    \'deflate\': {"true" if g2p_encoding["deflate"] else "false"},',
        '    \'tensors\': [',
        tensor_lines(g2p_quantized, '        '),
        '    ],',
        '    \'context\': {',
        '        \'design\': \'bigru\',',
        f'        \'chars\': "{CONTEXT_CHARS}",',
        f'        \'maxChars\': {MAX_CHARS},',
        f'        \'window\': {WINDOW},',
        f'        \'embed\': {context_config["embed"]},',
        f'        \'hidden\': {context_config["hidden"]},',
        f'        \'layer\': {context_config["layer"]},',
        f'        \'slots\': {context_config["slots"]},',
        f'        \'layout\': \'{context_encoding["layout"]}\',',
        f'        \'deflate\': {"true" if context_encoding["deflate"] else "false"},',
        '        \'tensors\': [',
        tensor_lines(context_quantized, '            '),
        '        ],',
        '        \'weights\': `',
        *base64_lines(context_text),
        '`',
        '    },',
        '    \'weights\': `',
        *base64_lines(g2p_text),
        '`',
        '};',
        ''
    ]
    return '\n'.join(lines)


def command_export(args):
    """Write the generated module from the G2P and context checkpoints - the G2P's matrices rounded by GPTQ (its
    checkpoint's calibration words), the context network's (trained at the export's rounding) to the nearest; the weights
    checked to round-trip exactly"""
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(1)
    g2p_checkpoint = torch.load(args.g2p, map_location='cpu', weights_only=False)
    context_checkpoint = torch.load(args.context, map_location='cpu', weights_only=False)
    covariances = g2p_covariances(load_g2p(args.g2p).eval(), g2p_checkpoint['calibration'])
    g2p_quantized = quantize_state(g2p_checkpoint['state'], EXPORT, covariances)
    context_quantized = quantize_state(context_checkpoint['state'], EXPORT)
    g2p_parameters = sum(item['integers'].size for item in g2p_quantized)
    context_parameters = sum(item['integers'].size for item in context_quantized)
    digests = [file_sha256(path)[:16] for path in (args.g2p, args.context)]
    config = g2p_checkpoint['config']
    comment = [
        f'The neural grapheme-to-phoneme model run by lib/g2p.js (and picovox_train/g2pmodel.py) - {config["design"]} design,',
        f'embedding {config["embed"]}, hidden {config["hidden"]}, {g2p_parameters} parameters with the reading input '
        f'({config["readings"]} readings) and the',
        f'ambiguity and spelled heads; and the context network, {context_parameters} parameters; {EXPORT["bits"]}-bit weights; trained by',
        f'picovox_train.g2p and picovox_train.context (checkpoints sha256 {digests[0]}, {digests[1]}).'
    ]
    text = module_text(config, g2p_quantized, context_checkpoint['config'], context_quantized, comment)
    model = g2pmodel.parse_module(text)
    tensors = g2pmodel.unpack_tensors(model)
    context_tensors = g2pmodel.unpack_tensors(model['context'])
    if not all((tensors[item['name']] == item['dequantized']).all() for item in g2p_quantized) or \
            not all((context_tensors[item['name']] == item['dequantized']).all() for item in context_quantized):
        log_line(None, 'export: the module\'s weights differ from the quantized weights')
        return 1
    with open(args.out, 'w', encoding='utf-8') as file:
        file.write(text)
    log_line(None, f'export: wrote {args.out} - G2P {g2p_parameters} parameters, context {context_parameters} parameters, '
                   f'{EXPORT["bits"]}-bit, {len(text.encode("utf-8"))} bytes ({len(gzip.compress(text.encode("utf-8"), 9))} gzipped)')
    return 0


#
# The gate - the held-out aligned accuracy against the majority baseline
#

# The gate's words (the recipe's: read, use, close, lives) less the words whose readings differ only by S against Z (the
# stem-final voicing pairs - use, close: their held-out accuracy is reported per word, not gated; the regression set
# covers them) - and its margin over the majority baseline
GATE_WORDS = RECIPE['gate']['words']
GATE_MARGIN = RECIPE['gate']['margin']


def describe_heldout(result):
    """The held-out examples' accuracy - over all, and the mean of the words' (all sources pooled per word)"""
    words = collections.defaultdict(lambda: [0, 0])
    for key, counts in result['words'].items():
        words[key.split('/', 1)[1]][0] += counts['right']
        words[key.split('/', 1)[1]][1] += counts['total']
    right, total = sum(right for right, _ in words.values()), sum(total for _, total in words.values())
    macro = sum(right / total for right, total in words.values()) / max(1, len(words))
    return f'held-out {right}/{total} ({100 * right / max(1, total):.1f}%), word mean {100 * macro:.1f}% ({len(words)} words)'


def gate_words(labels):
    """The gate's words - GATE_WORDS less the S/Z-only pairs (by the labels' readings)"""
    return [word for word in GATE_WORDS
            if word in labels['words'] and not sz_only(*(reading[0] for reading in labels['words'][word]['readings'][:2]))]


def gate_result(result, labels):
    """The gate on the held-out aligned tokens of the gate's words (pooled) - (passed, line)"""
    right = total = majority = 0
    parts = []
    for word in gate_words(labels):
        counts = result['words'].get(f'aligned/{word}', {'right': 0, 'total': 0, 'majority': 0})
        right, total, majority = right + counts['right'], total + counts['total'], majority + counts['majority']
        parts.append(f'{word} {counts["right"]}/{counts["total"]} (majority {counts["majority"]})')
    accuracy, baseline = right / max(1, total), majority / max(1, total)
    passed = total > 0 and accuracy >= baseline + GATE_MARGIN - 1e-9
    line = (f'held-out aligned {100 * accuracy:.1f}% vs majority {100 * baseline:.1f}% (+{100 * GATE_MARGIN:.0f} needed): ' +
            ', '.join(parts) + ('' if passed else ' - FAILED'))
    return passed, line


def word_margins(result, labels, least=20):
    """The held-out aligned accuracy against the majority of every word with least tokens, S/Z-only pairs apart - a line"""
    parts = []
    for key, counts in sorted(result['words'].items(), key=lambda item: -item[1]['total']):
        word = key.split('/', 1)[1]
        if key.startswith('aligned/') and counts['total'] >= least and word in labels['words']:
            readings = labels['words'][word]['readings']
            kind = ' S/Z' if sz_only(readings[0][0], readings[1][0]) else ''
            margin = 100 * (counts['right'] - counts['majority']) / counts['total']
            parts.append(f'{word}{kind} {counts["right"]}/{counts["total"]} ({margin:+.0f})')
    return 'per word (vs majority): ' + ', '.join(parts)


#
# make train's check - a retrained module against the committed one: identical, else within the recipe's tolerances
#

def js_pronouncer(module_path):
    """A pronouncer of word lists by a module as the app runs it (lib/g2p.js - tools/pronounceLines.js)"""

    def pronounce(words):
        process = subprocess.run(['node', os.path.join(ROOT, 'tools', 'pronounceLines.js'),
                                  os.path.abspath(module_path)], input=''.join(word + '\n' for word in words),
                                 capture_output=True, text=True, check=True)
        lines = process.stdout.split('\n')
        return [line.split() for line in lines[:len(words)]]
    return pronounce


def heteronym_scores(module_path):
    """The regression set's scores of a module (tools/heteronyms.js) - ({kind: (right, total)}, its exit status)"""
    process = subprocess.run(['node', os.path.join(ROOT, 'tools', 'heteronyms.js'), '-m', os.path.abspath(module_path)],
                             capture_output=True, text=True, check=False)
    kinds = {kind: (int(right), int(total)) for kind, right, total in re.findall(r'(\w+) (\d+)/(\d+)', process.stdout)}
    return kinds, process.returncode


def command_check(args):
    """make train's G2P gate - the retrained module against the committed one: identical (below the comment
    header), else the quality gate (the regression set - tools/heteronyms.js's minimums, and no kind more than its
    heteronymKindDrop below the committed module's - and g2p.py's book measures, and each evaluation text's error,
    within the recipe's tolerances). A
    kind without a drop isn't compared, its minimum its guard: the probes, which the context network's seed alone
    moves 5 of 38 (29 to 34 on the same labels) - a retrained module's score against the committed one's is two draws
    of that spread, and a drop of 2 failed half the trainings; and the context set, whose draws on one recipe in two
    days (2026-11-10 and 11 - the labels a few examples apart) were 52, 51, 54, 51, and 50 of 54. The monitoring WER's
    tolerance is its seeds' spread: four seeds of one recipe (2026-11-11) ended at 41.9%, 42.1%, 42.8%, and 43.0% of
    its 1,562 words (two trainings pronounce some 600 of them differently) - 1.5 points, two standard deviations of
    two draws' difference (half a point failed a training for its seed)"""
    if module_body(args.model) == module_body(args.committed):
        print(f'check: {args.model} is identical to {args.committed} (below the comment header)')
        return 0
    gate = RECIPE['gate']
    failures = []
    new_kinds, status = heteronym_scores(args.model)
    old_kinds, _ = heteronym_scores(args.committed)
    if status != 0:
        failures.append('the regression set is below its minimums')
    drops = gate['heteronymKindDrop']
    for kind, (right, total) in new_kinds.items():
        if kind in drops and right < old_kinds.get(kind, (0, total))[0] - drops[kind]:
            failures.append(f'{kind} {right}/{total} (committed {old_kinds[kind][0]})')
    _, data = g2p.final_data()
    new_measures = g2p.evaluate(js_pronouncer(args.model), data)
    old_measures = g2p.evaluate(js_pronouncer(args.committed), data)
    for name, tolerance in (('book', gate['bookError']), ('bookWrongDistinct', gate['bookDistinct']),
                            ('top10kWrong', gate['top10kWrong']), ('monitorWer', gate['monitorWer'])):
        if new_measures[name] > old_measures[name] + tolerance:
            failures.append(f'{name} {new_measures[name]} (committed {old_measures[name]}, tolerance {tolerance})')
    for name, measures in new_measures['sets'].items():
        if measures['error'] > old_measures['sets'][name]['error'] + gate['evalError']:
            failures.append(f'{name} {measures["error"]} (committed {old_measures["sets"][name]["error"]}, tolerance {gate["evalError"]})')
    kinds = ', '.join(f'{kind} {right}/{total} (committed {old_kinds.get(kind, (0, 0))[0]})' for kind, (right, total) in new_kinds.items())
    print(f'check: {args.model} differs from {args.committed}; regression set {kinds}; {g2p.describe(new_measures)} '
          f'(committed: {g2p.describe(old_measures)}) - ' +
          (f'FAILED: {"; ".join(failures)}' if failures else 'within the tolerances'))
    return 1 if failures else 0


#
# The commands
#

def command_train_g2p(args):
    """Train the reading heads and the spelled head from a G2P checkpoint on g2p.py's data and recipe settings - <out> (a
    checkpoint)"""
    import torch  # pylint: disable=import-outside-toplevel
    labels = read_json(args.labels)
    settings, data = g2p.final_data()
    items, probabilities = g2p.training_items(data, settings)
    spelled = spelled_module.spelled_labels(data, SPELLED)
    model = load_g2p(args.init)
    overrides = json.loads(args.settings) if args.settings else {}
    train_g2p(model, items, probabilities, labels, spelled, overrides, lambda text: log_line(args.log, text))
    torch.save({'config': model.config, 'state': model.state_dict(), 'settings': {**G2P_SETTINGS, **overrides},
                'calibration': g2p_calibration(items, probabilities, labels, spelled)}, args.out)
    result = evaluate_g2p(model, labels, list(data['first']), spelled)
    errors = result['spelled']
    log_line(args.log, f'train-g2p: readings {result["readingsRight"]}/{result["readings"]}, ambiguity missed '
                       f'{len(result["ambiguousMissed"])}, false positives {result["falsePositives"]}')
    log_line(args.log, f'train-g2p: spelled head - {len(errors["spelledMissed"])}/{errors["spelled"]} spelled strings found read, '
                       f'{len(errors["wordsSpelled"])}/{errors["words"]} words found spelled ({" ".join(errors["wordsSpelled"][:20])}), '
                       f'{len(errors["heldOutSpelled"])}/{errors["heldOut"]} held-out words; alone - '
                       f'{len(errors["aloneMissed"])}/{len(spelled["alone"])} of its words found read, '
                       f'{len(errors["aloneDiffers"])} other strings another way')
    return 0


def command_train_context(args):
    """Train the context network and its cross-validation - the network (fold 0: the labels' own split, at its threads,
    distilled from its teachers) to <out> (a checkpoint), and each fold's held-out results to <folds>/fold-<k>.json (the
    other folds' networks the plain training, one thread each - they estimate the held-out accuracy and never ship, so
    they need no teachers: 4 teachers and 10 networks, not 50 trainings). The teachers first, then every fold's network:
    -j trains that many at once, each in a worker process (the results are the same for any -j). Resumable: a fold with
    results from the same labels and settings (fold 0 with its checkpoint) isn't trained again"""
    os.makedirs(args.folds, exist_ok=True)
    network = {**CONTEXT_SETTINGS, **(json.loads(args.settings) if args.settings else {})}
    settings = [network] + [{**network, 'threads': 1, 'teachers': 0}] * (FOLDS - 1)
    paths = [os.path.join(args.folds, f'fold-{fold}.json') for fold in range(FOLDS)]
    digests = [fold_digest(args.labels, fold_settings) for fold_settings in settings]
    folds = [fold for fold in range(FOLDS) if not fold_done(paths[fold], digests[fold]) or (fold == 0 and not os.path.exists(args.out))]
    seeds = teacher_seeds(network) if 0 in folds else []
    states = run_tasks(teacher_task, [(args.labels, 0, seed, settings[0], args.log) for seed in seeds], args.jobs)
    run_tasks(fold_task, [(args.labels, fold, settings[fold], args.log, states if fold == 0 else [], paths[fold], digests[fold],
                           args.out if fold == 0 else None) for fold in folds], args.jobs)
    return 0


# The digest of a fold's inputs - the labels file and its network's settings
def fold_digest(labels_path, settings):
    digest = hashlib.sha256()
    with open(labels_path, 'rb') as file:
        digest.update(file.read())
    digest.update(json.dumps(settings, sort_keys=True).encode('utf-8'))
    return digest.hexdigest()[:16]


# Whether a fold's results exist from the same inputs (the digest)
def fold_done(path, digest):
    if not os.path.exists(path):
        return False
    return read_json(path).get('digest') == digest


# One fold's network (a worker task) - (labels path, fold, settings, log path, the fold's teachers' states, results path,
# the inputs' digest, checkpoint path - fold 0's, else None): the context network trained without the fold's books
# (fold 0: the network, its checkpoint written), and its held-out results (quantized as exported; with the digest)
# written to the results path
def fold_task(task):
    import torch  # pylint: disable=import-outside-toplevel
    labels_path, fold, settings, log, states, path, digest, checkpoint = task
    labels = labels_of(labels_path, fold)
    model = train_student(labels, fold, settings, log, states)
    if checkpoint:
        torch.save({'config': model.config, 'state': model.state_dict(), 'settings': settings}, checkpoint)
    WeightQuantizer(model, EXPORT).quantize()
    result = evaluate_context(lambda windows: context_predict(model, windows), labels)
    if checkpoint:
        _, line = gate_result(result, labels)
        log_line(log, f'train-context: {line}; {describe_heldout(result)}')
    with open(path + '.tmp', 'w', encoding='utf-8') as file:
        json.dump({**result, 'digest': digest}, file, indent=1)
    os.replace(path + '.tmp', path)
    return path


def command_folds(args):
    """The cross-validated gate - the held-out results of every fold's context network (train-context's
    <folds>/fold-<k>.json) pooled"""
    pooled = {'words': collections.defaultdict(lambda: {'right': 0, 'total': 0, 'majority': 0})}
    for fold in range(FOLDS):
        result = read_json(os.path.join(args.folds, f'fold-{fold}.json'))
        for key, counts in result['words'].items():
            if key.startswith('aligned/'):
                for name in ('right', 'total', 'majority'):
                    pooled['words'][key][name] += counts[name]
    labels = read_json(args.labels)
    passed, line = gate_result(pooled, labels)
    print(f'{FOLDS} folds: {line}')
    print(word_margins(pooled, labels))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.context', description='The heteronym models')
    commands = parser.add_subparsers(dest='command', required=True)
    g2p_parser = commands.add_parser('train-g2p', help='the G2P\'s reading input and ambiguity head')
    g2p_parser.add_argument('--labels', required=True)
    g2p_parser.add_argument('--init', required=True, help='the trained G2P checkpoint')
    g2p_parser.add_argument('--out', required=True)
    g2p_parser.add_argument('--settings', help='JSON settings over G2P_SETTINGS')
    g2p_parser.add_argument('--log')
    g2p_parser.set_defaults(func=command_train_g2p)
    context_parser = commands.add_parser('train-context', help='the context network and its cross-validation (resumable)')
    context_parser.add_argument('--labels', required=True)
    context_parser.add_argument('--out', required=True, help='the network\'s checkpoint')
    context_parser.add_argument('--folds', required=True, help='the folds\' results directory')
    context_parser.add_argument('--settings', help='JSON settings over CONTEXT_SETTINGS')
    context_parser.add_argument('-j', '--jobs', type=int, default=1, help='trainings at once (worker processes)')
    context_parser.add_argument('--log')
    context_parser.set_defaults(func=command_train_context)
    folds_parser = commands.add_parser('folds', help='the cross-validated gate over the folds\' held-out results')
    folds_parser.add_argument('folds', help='the folds\' results directory')
    folds_parser.add_argument('--labels', required=True)
    folds_parser.set_defaults(func=command_folds)
    export_parser = commands.add_parser('export', help='the generated module')
    export_parser.add_argument('--g2p', required=True)
    export_parser.add_argument('--context', required=True)
    export_parser.add_argument('--out', default=g2pmodel.MODULE_PATH)
    export_parser.set_defaults(func=command_export)
    check_parser = commands.add_parser('check', help='make train\'s check - a retrained module against the committed one')
    check_parser.add_argument('model')
    check_parser.add_argument('--committed', default=g2pmodel.MODULE_PATH)
    check_parser.set_defaults(func=command_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
