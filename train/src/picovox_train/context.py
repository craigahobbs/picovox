# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The heteronym models - the G2P network's reading input and ambiguity head, and the context network that chooses a
reading from the neighboring words (lib/g2p.js runs both from lib/models/g2pModel.js; g2pmodel.py is the port). No
dictionary, word list, or rule at runtime: a word whose ambiguity head fires takes the reading slot the context network
chooses, and the G2P pronounces the word in that reading. The commands (python is
PYTHONPATH=train/src build/venv-train/bin/python -B):

    python -m picovox_train.context train-g2p --labels DIR/labels.json --init G2P.pt --out DIR/g2p.pt
    python -m picovox_train.context train-context --labels DIR/labels.json --out DIR/context.pt [-j N]
    python -m picovox_train.context export --g2p DIR/g2p.pt --context DIR/context.pt [--out lib/models/g2pModel.js]
    python -m picovox_train.context gate --labels DIR/labels.json [MODULE]                           # held-out accuracy (one split)
    python -m picovox_train.context cross-validate --labels DIR/labels.json --out DIR/folds [-j N]   # the gate, 10 folds
    python -m picovox_train.context folds DIR/folds/fold-*.json --labels DIR/labels.json             # the gate of the folds' results
    python -m picovox_train.context check NEW.js     # make train's G2P gate: identical to lib/models/g2pModel.js,
                                                     # else the quality gate (make train-install installs it - trained.py)

The settings are train/data/train.json's "context" section (g2p, network, export, gate).

The G2P additions (g2p.py's model):

- the reading input - a learned embedding per alternate reading (1 to READINGS - 1), added to the decoder's initial state
  (reading 0, the default, adds nothing - the network's default pronunciations are unchanged by the addition)
- the ambiguity head - one logit from the encoder's final states: positive for a word with 2+ readings that the
  neighboring words decide (the labels' words, contextlabels.py)

Their training continues a trained G2P: its own data (each word's reading 0) plus each ambiguous word's alternate readings
("readingShare" of the samples), the other words with an alternate reading index and their default target
("invariantShare" - so a reading index on a word the head wrongly fires for changes nothing), and the ambiguity head's
binary cross entropy ("ambiguityWeight").

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
words' other windows in the books, contextlabels.py), "transferShare" of a batch's size added to each batch.
"""

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import sys
import time

import numpy

from . import RECIPE_PATH, ROOT, g2pmodel
from .trained import module_body
from .g2p import (EOS, LETTERS, MAX_WORD, MODULE_LINE_LENGTH, PHONES, UPSAMPLE, build_model, learning_rate, log_line, make_batch,
                  model_config, quantize_array, quantize_state, weight_bytes)


# The recipe's settings - train/data/train.json's "context" (labels, g2p, network, export, gate)
def load_settings(path=RECIPE_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)['context']


RECIPE = load_settings()

# The reading slots (the default and two alternates), and the ambiguity head's hidden layer
READINGS = RECIPE['g2p']['readings']
AMBIGUITY_HIDDEN = RECIPE['g2p']['ambiguityHidden']

# The context network's input - its characters (id 0 is padding), the characters of an item it reads, and the window
CONTEXT_CHARS = '\'abcdefghijklmnopqrstuvwxyz,.?!'
CONTEXT_CHAR_IDS = {char: index + 1 for index, char in enumerate(CONTEXT_CHARS)}
MAX_CHARS = RECIPE['network']['maxChars']
WINDOW = RECIPE['network']['window']

# The export - each model's matrices' bits (per-row bfloat16 scales) and its vectors' bits
EXPORT = RECIPE['export']

# The context network's settings, and the G2P reading stage's
CONTEXT_SETTINGS = {key: value for key, value in RECIPE['network'].items() if key not in ('window', 'maxChars')}
G2P_SETTINGS = {key: value for key, value in RECIPE['g2p'].items() if key not in ('readings', 'ambiguityHidden')}

# The reading stage's new tensors (their learning rate is the stage's times headRate)
HEADS = ('reading', 'ambiguity_layer', 'ambiguity')


# The labels (contextlabels.py labels)
def load_labels(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


#
# The G2P's reading input and ambiguity head
#

def add_reading_heads(model, readings=READINGS):
    """Add the reading embedding and the ambiguity head to a G2P model (g2p.py build_model)"""
    from torch import nn  # pylint: disable=import-outside-toplevel
    model.reading = nn.Embedding(readings - 1, model.config['decoder'])
    nn.init.normal_(model.reading.weight, std=1.0)
    model.ambiguity_layer = nn.Linear(2 * model.config['hidden'], AMBIGUITY_HIDDEN)
    model.ambiguity = nn.Linear(AMBIGUITY_HIDDEN, 1)
    nn.init.constant_(model.ambiguity.bias, -4.0)
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


# Encode letters with readings - (outputs, mask, initial decoder state, the encoder's final states)
def encode_reading(model, letters, readings):
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn  # pylint: disable=import-outside-toplevel
    mask = letters != 0
    packed = nn.utils.rnn.pack_padded_sequence(model.letter_embed(letters), mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
    outputs, final = model.encoder(packed)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=letters.shape[1])
    ends = torch.cat([final[0], final[1]], -1)
    state = torch.tanh(model.bridge(ends))
    alternate = readings > 0
    state = state + model.reading((readings - 1).clamp(min=0)) * alternate[:, None]
    return outputs, mask, state.unsqueeze(0), ends


# The ambiguity head's logits from the encoder's final states
def ambiguity_logit(model, ends):
    import torch  # pylint: disable=import-outside-toplevel
    return model.ambiguity(torch.tanh(model.ambiguity_layer(ends))).squeeze(-1)


# The loss of a batch - teacher-forced cross entropy of the readings' targets, plus the ambiguity head's
def reading_loss(model, letters, targets, readings, ambiguous, ambiguity_weight):
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn  # pylint: disable=import-outside-toplevel
    encoded, mask, state, ends = encode_reading(model, letters, readings)
    previous = torch.cat([torch.full_like(targets[:, :1], EOS), targets[:, :-1]], 1)
    decoded, _ = model.decoder(model.phone_embed(previous), state)
    logits = model.attend(decoded, encoded, mask)
    phone_loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    ambiguity_loss = nn.functional.binary_cross_entropy_with_logits(ambiguity_logit(model, ends), ambiguous)
    return phone_loss + ambiguity_weight * ambiguity_loss, phone_loss, ambiguity_loss


# Greedy decoding of words in readings - phone lists (lib/g2p.js's limit: 3 phones a letter plus 6)
def decode_readings(model, words, readings, batch_size=512):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            chunk = words[start:start + batch_size]
            letters = make_batch(chunk, [[] for _ in chunk])[0]
            reading = torch.tensor(readings[start:start + batch_size], dtype=torch.long)
            encoded, mask, state, _ = encode_reading(model, letters, reading)
            previous = torch.full((len(chunk), 1), EOS, dtype=torch.long)
            phones = [[] for _ in chunk]
            done = [False] * len(chunk)
            limits = [3 * length + 6 for length in mask.sum(1).tolist()]
            for step in range(3 * letters.shape[1] + 6):
                decoded, state = model.decoder(model.phone_embed(previous), state)
                best = model.attend(decoded, encoded, mask)[:, 0].argmax(-1)
                for index, phone in enumerate(best.tolist()):
                    if not done[index]:
                        if phone <= EOS or step >= limits[index]:
                            done[index] = True
                        else:
                            phones[index].append(PHONES[phone])
                if all(done):
                    break
                previous = best.unsqueeze(1)
            results.extend(phones)
    return results


# The ambiguity head's logits of words
def ambiguity_logits(model, words, batch_size=2048):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            chunk = words[start:start + batch_size]
            letters = make_batch(chunk, [[] for _ in chunk])[0]
            _, _, _, ends = encode_reading(model, letters, torch.zeros(len(chunk), dtype=torch.long))
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


def train_g2p(model, items, probabilities, labels, settings, log):
    """Train the reading input and the ambiguity head (and the rest) from a trained G2P - items [(word, phones)] and their
    sampling probabilities are the G2P's own data (reading 0); returns the model"""
    import torch  # pylint: disable=import-outside-toplevel
    settings = {**G2P_SETTINGS, **settings}
    torch.set_num_threads(int(settings['threads']))
    torch.manual_seed(int(settings['seed']))
    rng = numpy.random.default_rng(int(settings['seed']))
    if not hasattr(model, 'reading'):
        add_reading_heads(model)
    ambiguous = set(labels['words'])
    targets = reading_targets(labels, {word: phones for word, phones in items if word in ambiguous})
    invariant = numpy.array([0.0 if word in ambiguous else probability for (word, _), probability in zip(items, probabilities)])
    invariant /= invariant.sum()
    reading_share, invariant_share = float(settings['readingShare']), float(settings['invariantShare'])
    steps, batch_size, peak, warmup = int(settings['steps']), int(settings['batch']), float(settings['lr']), int(settings['warmup'])
    trained = [parameter for name, parameter in model.named_parameters()
               if settings['train'] == 'all' or name.split('.')[0] in settings['train'].split(',')]
    for parameter in model.parameters():
        parameter.requires_grad_(any(parameter is other for other in trained))
    head_rate = float(settings['headRate'])
    heads = [parameter for name, parameter in model.named_parameters() if name.split('.')[0] in HEADS]
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
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optimizer.step()
        window['phone'] += phone_loss.item()
        window['ambiguity'] += ambiguity_loss.item()
        if step % 500 == 0 or step == steps:
            count = step % 500 or 500
            log(f'train-g2p: step {step}/{steps} phone loss {window["phone"] / count:.4f}, ambiguity loss '
                f'{window["ambiguity"] / count:.5f} ({(time.time() - window_start) / count * 1000:.0f} ms/step)')
            window.clear()
            window_start = time.time()
    model.eval()
    return model


def evaluate_g2p(model, labels, words):
    """The reading measures of a G2P - the alternate readings right, and the ambiguity head's errors over words (the
    dictionary and labeled words) - {readingsRight, readings, ambiguousMissed, falsePositives, falsePositiveWords}"""
    targets = reading_targets(labels, {})
    predictions = decode_readings(model, [word for word, _, _ in targets], [reading for _, _, reading in targets])
    right = 0
    wrong = []
    for (word, _, reading), predicted in zip(targets, predictions):
        variants = labels['words'][word]['readings'][reading]
        if ' '.join(predicted) in variants:
            right += 1
        else:
            wrong.append(f'{word}/{reading} {" ".join(predicted)}')
    ambiguous = set(labels['words'])
    words = sorted(set(words) | ambiguous)
    logits = ambiguity_logits(model, words)
    missed = [word for word, logit in zip(words, logits) if word in ambiguous and logit <= 0]
    false_positives = [word for word, logit in zip(words, logits) if word not in ambiguous and logit > 0]
    return {'readingsRight': right, 'readings': len(targets), 'readingsWrong': wrong, 'ambiguousMissed': missed,
            'falsePositives': len(false_positives), 'falsePositiveWords': false_positives[:50]}


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


# The cross-validation folds - by book (fold 0 is the labels' held-out books - contextlabels.py; a book of the others is in
# fold 1 + its hash's remainder by FOLDS - 1)
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


# The labels of a fold (None - all the labels' splits as they are)
def labels_of(labels_path, fold):
    labels = load_labels(labels_path)
    return labels if fold is None else fold_labels(labels, fold)


# The teachers' seeds - "teachers" of them, 100 * seed + 1, 2, ...
def teacher_seeds(settings):
    settings = {**CONTEXT_SETTINGS, **settings}
    return [100 * int(settings['seed']) + index + 1 for index in range(int(settings['teachers']))]


# One teacher (a worker task) - (labels path, fold or None, seed, settings, log path): the context network trained with
# that seed (no teachers of its own) - its state, quantized as exported
def teacher_task(task):
    labels_path, fold, seed, settings, log = task
    prefix = '' if fold is None else f'fold {fold}: '
    teacher = train_context(labels_of(labels_path, fold), {**settings, 'seed': seed},
                            lambda text: log_line(log, f'{prefix}teacher {seed}: {text}'))
    WeightQuantizer(teacher, EXPORT).quantize()
    return teacher.state_dict()


# Worker tasks' results in order - in jobs worker processes when more than one (the same results either way)
def run_tasks(function, tasks, jobs):
    if jobs > 1 and len(tasks) > 1:
        with multiprocessing.Pool(min(jobs, len(tasks))) as pool:
            return pool.map(function, tasks)
    return [function(task) for task in tasks]


def train_student(labels_path, fold, settings, log, states):
    """The context network trained on a fold's labels toward the teachers' mean probabilities (teachers' states - none,
    the plain training) - the model"""
    settings = {**CONTEXT_SETTINGS, **settings}
    teachers = []
    for state in states:
        teacher = build_context(settings)
        teacher.load_state_dict(state)
        teachers.append(teacher)
    prefix = '' if fold is None else f'fold {fold}: '
    return train_context(labels_of(labels_path, fold), settings, lambda text: log_line(log, f'{prefix}{text}'), teachers)


def train_distilled(labels_path, settings, log, jobs=1):
    """The recipe's context network - its teachers ("teachers" of them, each its own seed; in jobs worker processes),
    then the network distilled from them (train_student) - the model"""
    tasks = [(labels_path, None, seed, {**CONTEXT_SETTINGS, **settings}, log) for seed in teacher_seeds(settings)]
    return train_student(labels_path, None, settings, log, run_tasks(teacher_task, tasks, jobs))


def evaluate_context(slot_of, labels, split='heldout'):
    """The context choices' accuracy per word and source on a split, against each word's majority slot (in the training
    examples) - slot_of(windows) gives the chosen slots - {words: {word: {right, total, majority}}, sources: {...}}"""
    examples = [example for example in labels['examples'] if example['split'] == split]
    majority = {}
    training = collections.defaultdict(collections.Counter)
    for example in labels['examples']:
        if example['split'] == 'train':
            training[example['word']][example['slot']] += 1
    for word, counter in training.items():
        majority[word] = counter.most_common(1)[0][0]
    chosen = slot_of([example['window'] for example in examples])
    words = collections.defaultdict(lambda: {'right': 0, 'total': 0, 'majority': 0})
    sources = collections.defaultdict(lambda: {'right': 0, 'total': 0, 'majority': 0})
    for example, slot in zip(examples, chosen):
        for counts in (words[(example['source'], example['word'])], sources[example['source']]):
            counts['total'] += 1
            counts['right'] += slot == example['slot']
            counts['majority'] += majority.get(example['word'], 0) == example['slot']
    return {'words': {f'{source}/{word}': counts for (source, word), counts in sorted(words.items())}, 'sources': dict(sources)}


#
# The export - the G2P module with the reading heads and the context network
#

# The smallest encoding of a quantized state - (manifest fields {layout, deflate}, base64 text)
def encode_weights(quantized):
    import base64  # pylint: disable=import-outside-toplevel
    import zlib  # pylint: disable=import-outside-toplevel
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
    """Write the generated module from the G2P and context checkpoints - the weights checked to round-trip exactly"""
    import torch  # pylint: disable=import-outside-toplevel
    g2p_checkpoint = torch.load(args.g2p, map_location='cpu', weights_only=False)
    context_checkpoint = torch.load(args.context, map_location='cpu', weights_only=False)
    g2p_quantized = quantize_state(g2p_checkpoint['state'], EXPORT)
    context_quantized = quantize_state(context_checkpoint['state'], EXPORT)
    g2p_parameters = sum(item['integers'].size for item in g2p_quantized)
    context_parameters = sum(item['integers'].size for item in context_quantized)
    digests = []
    for path in (args.g2p, args.context):
        with open(path, 'rb') as file:
            digests.append(hashlib.sha256(file.read()).hexdigest()[:16])
    config = g2p_checkpoint['config']
    comment = [
        f'The neural grapheme-to-phoneme model run by lib/g2p.js (and picovox_train/g2pmodel.py) - {config["design"]} design,',
        f'embedding {config["embed"]}, hidden {config["hidden"]}, {g2p_parameters} parameters with the reading input '
        f'({config["readings"]} readings) and the',
        f'ambiguity head; and the context network, {context_parameters} parameters; {EXPORT["bits"]}-bit weights; trained by',
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

# The gate's words (the recipe's: read, use, close, lives) less the words whose readings differ only by S against Z - the
# aligner can't judge those (word-final voicing), so the regression set covers them - and its margin over the majority
# baseline
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
    from .readings import sz_only  # pylint: disable=import-outside-toplevel
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
    from .readings import sz_only  # pylint: disable=import-outside-toplevel
    parts = []
    for key, counts in sorted(result['words'].items(), key=lambda item: -item[1]['total']):
        word = key.split('/', 1)[1]
        if key.startswith('aligned/') and counts['total'] >= least and word in labels['words']:
            readings = labels['words'][word]['readings']
            kind = ' S/Z' if sz_only(readings[0][0], readings[1][0]) else ''
            margin = 100 * (counts['right'] - counts['majority']) / counts['total']
            parts.append(f'{word}{kind} {counts["right"]}/{counts["total"]} ({margin:+.0f})')
    return 'per word (vs majority): ' + ', '.join(parts)


def command_gate(args):
    """The held-out accuracy of a module's context network (as the app runs it - g2pmodel.py), and the gate"""
    labels = load_labels(args.labels)
    g2p = g2pmodel.G2p(g2pmodel.read_module(args.model))
    result = evaluate_context(g2p.choose_slots, labels)
    passed, line = gate_result(result, labels)
    sources = ', '.join(f'{source} {counts["right"]}/{counts["total"]} (majority {counts["majority"]})'
                        for source, counts in sorted(result['sources'].items()))
    print(f'{line}; all held-out: {sources}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=1)
    return 0 if passed else 1


#
# make train's check - a retrained module against the committed one: identical, else within the recipe's tolerances
#

def js_pronouncer(module_path):
    """A pronouncer of word lists by a module as the app runs it (lib/g2p.js - tools/pronounceLines.js)"""
    import subprocess  # pylint: disable=import-outside-toplevel

    def pronounce(words):
        process = subprocess.run(['node', os.path.join(ROOT, 'tools', 'pronounceLines.js'),
                                  os.path.abspath(module_path)], input=''.join(word + '\n' for word in words),
                                 capture_output=True, text=True, check=True)
        lines = process.stdout.split('\n')
        return [line.split() for line in lines[:len(words)]]
    return pronounce


def heteronym_scores(module_path):
    """The regression set's scores of a module (tools/heteronyms.js) - ({kind: (right, total)}, its exit status)"""
    import re  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    process = subprocess.run(['node', os.path.join(ROOT, 'tools', 'heteronyms.js'), '-m', os.path.abspath(module_path)],
                             capture_output=True, text=True, check=False)
    kinds = {kind: (int(right), int(total)) for kind, right, total in re.findall(r'(\w+) (\d+)/(\d+)', process.stdout)}
    return kinds, process.returncode


def command_check(args):
    """make train's G2P gate - the retrained module against the committed one: identical (below the comment
    header), else the quality gate (the regression set - tools/heteronyms.js's minimums, and no kind more than
    heteronymKindDrop below the committed module's - and g2p.py's book measures within the recipe's tolerances)"""
    from . import g2p  # pylint: disable=import-outside-toplevel
    if module_body(args.model) == module_body(args.committed):
        print(f'check: {args.model} is identical to {args.committed} (below the comment header)')
        return 0
    gate = RECIPE['gate']
    failures = []
    new_kinds, status = heteronym_scores(args.model)
    old_kinds, _ = heteronym_scores(args.committed)
    if status != 0:
        failures.append('the regression set is below its minimums')
    for kind, (right, total) in new_kinds.items():
        if right < old_kinds.get(kind, (0, total))[0] - gate['heteronymKindDrop']:
            failures.append(f'{kind} {right}/{total} (committed {old_kinds[kind][0]})')
    config = g2p.load_config()
    relabel = config['stages'][-1].get('relabel')
    data = g2p.load_data(relabel and os.path.join(g2p.ROOT, relabel))
    new_measures = g2p.evaluate(js_pronouncer(args.model), data)
    old_measures = g2p.evaluate(js_pronouncer(args.committed), data)
    for name, tolerance in (('book', gate['bookError']), ('bookWrongDistinct', gate['bookDistinct']),
                            ('top10kWrong', gate['top10kWrong']), ('monitorWer', gate['monitorWer'])):
        if new_measures[name] > old_measures[name] + tolerance:
            failures.append(f'{name} {new_measures[name]} (committed {old_measures[name]}, tolerance {tolerance})')
    kinds = ', '.join(f'{kind} {right}/{total} (committed {old_kinds.get(kind, (0, 0))[0]})' for kind, (right, total) in new_kinds.items())
    print(f'check: {args.model} differs from {args.committed}; regression set {kinds}; {g2p.describe(new_measures)} '
          f'(committed: {g2p.describe(old_measures)}) - ' +
          (f'FAILED: {"; ".join(failures)}' if failures else 'within the tolerances'))
    return 1 if failures else 0


#
# The commands
#

def command_train_g2p(args):
    """Train the reading heads from a G2P checkpoint on g2p.py's data and recipe settings - <out> (a checkpoint)"""
    import torch  # pylint: disable=import-outside-toplevel
    from . import g2p  # pylint: disable=import-outside-toplevel
    labels = load_labels(args.labels)
    config = g2p.load_config(args.config)
    stage = config['stages'][-1]
    settings = {key: value for key, value in config.items() if key != 'stages'}
    settings = {**settings, **stage}
    data = g2p.load_data(settings.get('relabel') and os.path.join(g2p.ROOT, settings['relabel']))
    items, probabilities = g2p.training_items(data, settings)
    model = load_g2p(args.init)
    overrides = json.loads(args.settings) if args.settings else {}
    train_g2p(model, items, probabilities, labels, overrides, lambda text: log_line(args.log, text))
    torch.save({'config': model.config, 'state': model.state_dict(), 'settings': {**G2P_SETTINGS, **overrides}}, args.out)
    result = evaluate_g2p(model, labels, list(data['first']))
    log_line(args.log, f'train-g2p: readings {result["readingsRight"]}/{result["readings"]}, ambiguity missed '
                       f'{len(result["ambiguousMissed"])}, false positives {result["falsePositives"]}')
    return 0


def command_train_context(args):
    """Train the context network (its teachers first - in -j worker processes) - <out> (a checkpoint)"""
    import torch  # pylint: disable=import-outside-toplevel
    labels = load_labels(args.labels)
    overrides = json.loads(args.settings) if args.settings else {}
    model = train_distilled(args.labels, overrides, args.log, args.jobs)
    torch.save({'config': model.config, 'state': model.state_dict(), 'settings': {**CONTEXT_SETTINGS, **overrides}}, args.out)
    quantizer = WeightQuantizer(model, EXPORT)
    quantizer.quantize()
    result = evaluate_context(lambda windows: context_predict(model, windows), labels)
    _, line = gate_result(result, labels)
    log_line(args.log, f'train-context: {line}; {describe_heldout(result)}')
    with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    return 0


# The digest of a cross-validation's inputs - the labels file and the context network's settings (less the threads)
def fold_digest(labels_path):
    digest = hashlib.sha256()
    with open(labels_path, 'rb') as file:
        digest.update(file.read())
    digest.update(json.dumps({key: value for key, value in CONTEXT_SETTINGS.items() if key != 'threads'}, sort_keys=True).encode('utf-8'))
    return digest.hexdigest()[:16]


# Whether a fold's results exist from the same inputs (the digest)
def fold_done(path, digest):
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as file:
        return json.load(file).get('digest') == digest


# One fold of the cross-validation (a worker task) - (labels path, fold, results path, log path, settings, the fold's
# teachers' states): the context network trained without the fold's books, and its held-out results (with the
# inputs' digest) written to the results path
def fold_task(task):
    labels_path, fold, path, log, settings, states = task
    fold_set = labels_of(labels_path, fold)
    model = train_student(labels_path, fold, settings, log, states)
    WeightQuantizer(model, EXPORT).quantize()
    result = {**evaluate_context(lambda windows: context_predict(model, windows), fold_set), 'digest': fold_digest(labels_path)}
    with open(path + '.tmp', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    os.replace(path + '.tmp', path)
    return path


def command_cross_validate(args):
    """Train the context network once per fold of the books - each fold's held-out results to
    <out>/fold-<k>.json (resumable: a fold with results from the same labels and settings is skipped - one from other
    inputs is retrained) - then the cross-validated gate (command_folds). Every fold's teachers, then every fold's
    network: -j trains that many at once, each in a worker process, and every training is one thread (the results are the
    same for any -j)"""
    os.makedirs(args.out, exist_ok=True)
    paths = [os.path.join(args.out, f'fold-{fold}.json') for fold in range(FOLDS)]
    digest = fold_digest(args.labels)
    folds = [fold for fold, path in enumerate(paths) if not fold_done(path, digest)]
    settings = {**CONTEXT_SETTINGS, 'threads': 1}
    seeds = teacher_seeds(settings)
    states = run_tasks(teacher_task, [(args.labels, fold, seed, settings, args.log) for fold in folds for seed in seeds], args.jobs)
    run_tasks(fold_task, [(args.labels, fold, paths[fold], args.log, settings, states[index * len(seeds):(index + 1) * len(seeds)])
                          for index, fold in enumerate(folds)], args.jobs)
    return command_folds(argparse.Namespace(results=paths, labels=args.labels))


def command_folds(args):
    """The cross-validated gate - the held-out results of every fold's context network (cross-validate's
    <out>/fold-<k>.json) pooled"""
    pooled = {'words': collections.defaultdict(lambda: {'right': 0, 'total': 0, 'majority': 0}), 'sources': {}}
    for path in args.results:
        with open(path, encoding='utf-8') as file:
            result = json.load(file)
        for key, counts in result['words'].items():
            if key.startswith('aligned/'):
                for name in ('right', 'total', 'majority'):
                    pooled['words'][key][name] += counts[name]
    labels = load_labels(args.labels)
    passed, line = gate_result(pooled, labels)
    print(f'{len(args.results)} folds: {line}')
    print(word_margins(pooled, labels))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.context', description='The heteronym models')
    commands = parser.add_subparsers(dest='command', required=True)
    g2p_parser = commands.add_parser('train-g2p', help='the G2P\'s reading input and ambiguity head')
    g2p_parser.add_argument('--labels', required=True)
    g2p_parser.add_argument('--init', required=True, help='the trained G2P checkpoint')
    g2p_parser.add_argument('--out', required=True)
    g2p_parser.add_argument('--config', default=RECIPE_PATH, help='g2p.py\'s settings')
    g2p_parser.add_argument('--settings', help='JSON settings over G2P_SETTINGS')
    g2p_parser.add_argument('--log')
    g2p_parser.set_defaults(func=command_train_g2p)
    context_parser = commands.add_parser('train-context', help='the context network')
    context_parser.add_argument('--labels', required=True)
    context_parser.add_argument('--out', required=True)
    context_parser.add_argument('--settings', help='JSON settings over CONTEXT_SETTINGS')
    context_parser.add_argument('-j', '--jobs', type=int, default=1, help='teachers at once (worker processes)')
    context_parser.add_argument('--log')
    context_parser.set_defaults(func=command_train_context)
    cross_parser = commands.add_parser('cross-validate', help='train every fold, then the cross-validated gate (resumable)')
    cross_parser.add_argument('--labels', required=True)
    cross_parser.add_argument('--out', required=True, help='the folds\' results directory')
    cross_parser.add_argument('--log')
    cross_parser.add_argument('-j', '--jobs', type=int, default=1, help='trainings at once (worker processes, one thread each)')
    cross_parser.set_defaults(func=command_cross_validate)
    folds_parser = commands.add_parser('folds', help='the cross-validated gate over the folds\' held-out results')
    folds_parser.add_argument('results', nargs='+')
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
    gate_parser = commands.add_parser('gate', help='the held-out accuracy of a module\'s context network (exit 1 below the gate)')
    gate_parser.add_argument('model', nargs='?', default=g2pmodel.MODULE_PATH)
    gate_parser.add_argument('--labels', required=True)
    gate_parser.add_argument('--out')
    gate_parser.set_defaults(func=command_gate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
