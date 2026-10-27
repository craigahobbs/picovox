# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Training the grapheme-to-phoneme (G2P) network - the app's only pronouncer (lib/g2p.js runs lib/models/g2pModel.js for
every word; g2pmodel.py is its Python port). The recipe is train/data/train.json's "g2p" settings:

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2p train --out build/nn/g2p-train   # every stage (resumable)
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2p evaluate [lib/models/g2pModel.js | CHECKPOINT]
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.g2p book TEXT --name NAME --title TITLE --url URL

The data:

- every word of the CMU Pronouncing Dictionary (downloads/cmudict/cmudict.dict - the lowercase words textToPhrases can look up) with
  its first pronunciation (lib/lexicon.js's target), but a 2,000-word monitoring set (validation-split words that occur
  in none of the 128 books), sampled by frequency - "uniform" of the samples uniform over the words, the rest in
  proportion to count ** "alpha" (train/data/g2p-words.json evalCounts + weightCounts, the words' counts in 128 Project
  Gutenberg books)
- "unknown" of the samples uniform over 100,812 labeled words the dictionary lacks (train/data/g2p-words.json labels - a
  teacher ensemble's, or the spelling and -s rules' where they apply to a dictionary word)
- in a stage with "relabel" (a JSON file {"relabel": {word: "PH ON ES"}} - train/data/g2p-relabel.json, heteronyms.py)
  other targets for some dictionary words - the rule-labeled words derived from them follow - and ("focusBoost":
  {"share"}) a share of the samples uniform over them
- the words of the books listeners hear (train/data/g2p-books.json - "book" counts a book's words): labels for its words
  the dictionary lacks or has wrong (names - Rocinante, Cide Hamete - and archaic words), each a dictionary word in
  training (a corrected dictionary word's rule-labeled forms following), its possessives the dictionary lacks (the
  stem's target with its -s suffix), and its word counts, which join the frequency weights (the monitoring set is drawn
  without them, then loses the books' words)

Each stage trains from the previous one's weights (AdamW, linear warmup then cosine decay to 2%, gradient clipping at
1) with its own seed, learning rate, and sampling ("topBoost": {"share", "count"} - a share of the dictionary samples
uniform over the most frequent words). The network is exported with the heteronym models (picovox_train.context
export, by the quantization here - "bits" per weight, bfloat16 scales per row, 8-bit biases).

The measures: the book-token error (the share of the 22 evaluation books' dictionary-word tokens whose prediction isn't
the first pronunciation; "any" - none of the word's pronunciations), the top-10K's wrong words, and the monitoring
set's word error (any pronunciation).
"""

import argparse
import collections
import datetime
import json
import math
import os
import random
import re
import sys
import time

import numpy

from . import RECIPE_PATH, ROOT, corpora, frontend, g2pmodel, read_json
from .frontend import vocabulary


WORDS_PATH = os.path.join(ROOT, 'train', 'data', 'g2p-words.json')
FIXES_PATH = os.path.join(ROOT, 'train', 'data', 'g2p-fixes.json')
BOOKS_PATH = os.path.join(ROOT, 'train', 'data', 'g2p-books.json')
MODULE_PATH = g2pmodel.MODULE_PATH

# The dictionary's split (the monitoring set is drawn from its validation words)
SPLIT_SEED = 20261015
TEST_FRACTION = 0.05
VALIDATION_FRACTION = 0.03
MONITOR_WORDS = 2000

WORD_REGEX = re.compile(r"^[a-z']+$")
LETTERS = ['<pad>', '\''] + [chr(code) for code in range(ord('a'), ord('z') + 1)]
LETTER_IDS = {letter: index for index, letter in enumerate(LETTERS)}
PHONES = ['<blank>', '<eos>'] + [token for token in vocabulary() if token[0].isalpha() and token[0].isupper()]
PHONE_IDS = {phone: index for index, phone in enumerate(PHONES)}
BLANK, EOS = 0, 1
MAX_WORD = 32

# The generated module - its line length, and the (fixed) CTC upsampling factor its manifest carries
MODULE_LINE_LENGTH = 120
UPSAMPLE = 3

# The scale clipping ratios the quantization tries (the best by squared error, per row)
CLIP_RATIOS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]


# A progress line - stderr, and the log file if given
def log_line(path, text):
    line = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {text}'
    print(line, file=sys.stderr, flush=True)
    if path:
        with open(path, 'a', encoding='utf-8') as file:
            file.write(line + '\n')


# The settings - train/data/train.json's "g2p"
def load_config(path=RECIPE_PATH):
    return read_json(path)['g2p']


#
# The data
#

# The dictionary - word to its pronunciations (each a list of phones), for the words textToPhrases can look up
def read_pronunciations():
    pronunciations = {}
    for word, variants in corpora.read_dictionary().items():
        if WORD_REGEX.match(word) and word.strip('\'') == word:
            pronunciations[word] = []
            for phones in variants:
                if phones not in pronunciations[word]:
                    pronunciations[word].append(phones)
    return pronunciations


# The dictionary's split by word - {test, validation, train} (fixed seed)
def split_words(words):
    shuffled = sorted(words)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    test_count = round(len(shuffled) * TEST_FRACTION)
    validation_count = round(len(shuffled) * VALIDATION_FRACTION)
    return {'test': shuffled[:test_count], 'validation': shuffled[test_count:test_count + validation_count]}


def load_data(relabel=None, fixes=FIXES_PATH, books=BOOKS_PATH):
    """The data - {first (word: target phones), all (word: every pronunciation), labels (unknown words' phones), stems,
    evalCounts, weightCounts, monitor, relabeled (the relabeled words)}; fixes (train/data/g2p-fixes.json - None for the
    dictionary as it is) corrects the dictionary's label errors first, and books (train/data/g2p-books.json - None for
    none) adds the listeners' books' words"""
    pronunciations = read_pronunciations()
    words = read_json(WORDS_PATH)
    first = {word: phones[0] for word, phones in pronunciations.items()}
    labels = {word: phones.split() for word, phones in words['labels'].items()}
    stems = words['stems']

    # The dictionary's corrected labels (g2paudit.py) - the corrected pronunciation first, the wrong one dropped
    if fixes:
        corrections = {word: phones.split() for word, phones in read_json(fixes)['fixes'].items() if word in first}
        derive_targets(labels, stems, first, corrections)
        for word, phones in corrections.items():
            pronunciations[word] = [phones] + [variant for variant in pronunciations[word] if variant not in (phones, first[word])]
            first[word] = phones

    # The monitoring set - the dictionary's validation words in none of the books (the 128 books - the listeners' books'
    # words leave it, so they train)
    eval_counts, weight_counts = words['evalCounts'], words['weightCounts']
    validation = sorted(word for word in split_words(first)['validation'] if word not in eval_counts and word not in weight_counts)
    random.Random(SPLIT_SEED).shuffle(validation)
    monitor = validation[:MONITOR_WORDS]

    # The books' words - their labels, their possessives, and their counts (weights)
    if books:
        book_counts = add_books(books, pronunciations, first, labels, stems)
        weight_counts = dict(collections.Counter(weight_counts) + collections.Counter(book_counts))
        monitor = [word for word in monitor if word not in book_counts]

    # Other targets for some dictionary words - and the rule-labeled words derived from them
    relabeled = set()
    if relabel:
        targets = {word: phones.split() for word, phones in read_json(relabel)['relabel'].items()}
        derive_targets(labels, stems, first, targets)
        first.update((word, phones) for word, phones in targets.items() if word in first)
        relabeled = set(targets) & set(first)

    return {'first': first, 'all': pronunciations, 'labels': labels, 'stems': stems, 'evalCounts': eval_counts,
            'weightCounts': weight_counts, 'monitor': sorted(monitor), 'relabeled': relabeled}


# Add the books' words (train/data/g2p-books.json, in place) - each label (a word the dictionary lacks, or a correction of
# its label - the rule-labeled words derived from it follow) a dictionary word, and each possessive in the books the
# dictionary lacks the stem's target with its -s suffix (a teacher's label of the word replaced); returns the books' word
# counts, summed
def add_books(path, pronunciations, first, labels, stems):
    books = read_json(path)
    targets = {word: phones.split() for word, phones in books['labels'].items()}
    derive_targets(labels, stems, first, {word: phones for word, phones in targets.items() if word in first})
    for word, phones in targets.items():
        pronunciations[word] = [phones] + [variant for variant in pronunciations.get(word, []) if variant not in (phones, first.get(word))]
        first[word] = phones
        labels.pop(word, None)
    counts = collections.Counter()
    for book in books['books'].values():
        counts.update(book['counts'])
    for word in sorted(counts):
        stem = word[:-2]
        if word.endswith('\'s') and word not in first and stem in first:
            first[word] = first[stem] + s_suffix(first[stem])
            pronunciations[word] = [first[word]]
            labels.pop(word, None)
    return dict(counts)


# The rule-labeled words derived from dictionary words with new targets follow them (in place): a spelling variant
# takes the target, and a derived form (-s, -'s) the target with its suffix - the -s suffix chosen again by the target's
# last phone
def derive_targets(labels, stems, first, targets):
    for word, (stem, kind) in stems.items():
        old = first.get(stem)
        if stem in targets and old is not None:
            if kind == 'spelling' and labels[word] == old:
                labels[word] = targets[stem]
            elif labels[word][:len(old)] == old:
                suffix = labels[word][len(old):]
                if suffix in (['S'], ['Z'], ['IH0', 'Z']) and suffix == s_suffix(old):
                    suffix = s_suffix(targets[stem])
                labels[word] = targets[stem] + suffix


# The -s suffix after a pronunciation's last phone - IH0 Z after a sibilant, S after another voiceless consonant, else Z
def s_suffix(phones):
    if phones and phones[-1] in ('S', 'Z', 'SH', 'ZH', 'CH', 'JH'):
        return ['IH0', 'Z']
    if phones and phones[-1] in ('P', 'T', 'K', 'F', 'TH'):
        return ['S']
    return ['Z']


def training_items(data, settings):
    """The training items [(word, phones)] and their sampling probabilities"""
    held_out = set(data['monitor'])
    words = sorted(word for word in data['first'] if word not in held_out)
    items = [(word, data['first'][word]) for word in words]
    counts = collections.Counter(data['weightCounts'])
    counts.update(data['evalCounts'])
    alpha, uniform = float(settings['alpha']), float(settings['uniform'])
    weights = numpy.array([counts.get(word, 0) ** alpha if counts.get(word, 0) else 0.0 for word in words])
    probabilities = uniform / len(words) + (1 - uniform) * weights / weights.sum()
    if settings.get('topBoost'):
        share, top_count = float(settings['topBoost']['share']), int(settings['topBoost']['count'])
        ranked = sorted(range(len(words)), key=lambda index: -counts.get(words[index], 0))[:top_count]
        boost = numpy.zeros(len(words))
        boost[ranked] = 1.0 / len(ranked)
        probabilities = (1 - share) * probabilities + share * boost
    if settings.get('focusBoost') and data['relabeled']:
        share = float(settings['focusBoost']['share'])
        boost = numpy.array([1.0 if word in data['relabeled'] else 0.0 for word in words])
        probabilities = (1 - share) * probabilities + share * boost / boost.sum()
    unknown_share = float(settings['unknown'])
    if unknown_share > 0:
        # (rule labels built from a held-out word's pronunciation would leak it)
        unknown = [word for word in sorted(data['labels']) if data['stems'].get(word, [None])[0] not in held_out]
        items += [(word, data['labels'][word]) for word in unknown]
        probabilities = numpy.concatenate([(1 - unknown_share) * probabilities, numpy.full(len(unknown), unknown_share / len(unknown))])
    return items, probabilities / probabilities.sum()


#
# The model - an autoregressive encoder-decoder: a bidirectional GRU over the letters, a GRU decoder over the previous
# phones (started from the encoder's ends), dot-product attention after the decoder (no input feeding, so training runs
# the decoder in parallel), and an output layer over the phones and <eos>; greedy decoding (lib/g2p.js)
#

# The model's hyperparameters from the settings
def model_config(config):
    return {'design': 'ar', 'embed': int(config['embed']), 'hidden': int(config['hidden']),
            'decoder': int(config.get('decoder') or 2 * int(config['hidden']))}


def build_model(config):
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn  # pylint: disable=import-outside-toplevel

    config = model_config(config)

    # (no forward - the model is run by its loss and decode)
    class ArModel(nn.Module):  # pylint: disable=abstract-method
        """The autoregressive encoder-decoder"""

        def __init__(self):
            super().__init__()
            self.config = config
            embed, hidden, decoder = config['embed'], config['hidden'], config['decoder']
            self.letter_embed = nn.Embedding(len(LETTERS), embed, padding_idx=0)
            self.encoder = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)
            self.bridge = nn.Linear(2 * hidden, decoder)
            self.phone_embed = nn.Embedding(len(PHONES), embed)
            self.decoder = nn.GRU(embed, decoder, batch_first=True)
            self.attention = nn.Linear(decoder, 2 * hidden, bias=False)
            self.combine = nn.Linear(decoder + 2 * hidden, decoder)
            self.output = nn.Linear(decoder, len(PHONES))

        # Encode letters [batch, time] - (outputs [batch, time, 2H], mask [batch, time], initial decoder state)
        def encode(self, letters):
            mask = letters != 0
            packed = nn.utils.rnn.pack_padded_sequence(self.letter_embed(letters), mask.sum(1).cpu(), batch_first=True,
                                                       enforce_sorted=False)
            outputs, final = self.encoder(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=letters.shape[1])
            state = torch.tanh(self.bridge(torch.cat([final[0], final[1]], -1))).unsqueeze(0)
            return outputs, mask, state

        # The logits of each step from the decoder outputs [batch, steps, D]
        def attend(self, decoded, encoded, mask):
            scores = torch.einsum('bsh,bth->bst', self.attention(decoded), encoded)
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            context = torch.einsum('bst,bth->bsh', torch.softmax(scores, -1), encoded)
            return self.output(torch.tanh(self.combine(torch.cat([decoded, context], -1))))

        # The loss of a batch - teacher-forced cross entropy (targets end with <eos>, padded with 0)
        def loss(self, letters, targets):
            encoded, mask, state = self.encode(letters)
            previous = torch.cat([torch.full_like(targets[:, :1], EOS), targets[:, :-1]], 1)
            decoded, _ = self.decoder(self.phone_embed(previous), state)
            logits = self.attend(decoded, encoded, mask)
            return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)

        # Greedy decoding - a list of phone id lists (lib/g2p.js's limit: 3 phones a letter plus 6)
        def decode(self, letters):
            encoded, mask, state = self.encode(letters)
            batch = letters.shape[0]
            previous = torch.full((batch, 1), EOS, dtype=torch.long)
            results = [[] for _ in range(batch)]
            done = [False] * batch
            limits = [3 * length + 6 for length in mask.sum(1).tolist()]
            for step in range(3 * letters.shape[1] + 6):
                decoded, state = self.decoder(self.phone_embed(previous), state)
                best = self.attend(decoded, encoded, mask)[:, 0].argmax(-1)
                for index, phone in enumerate(best.tolist()):
                    if not done[index]:
                        if phone in (BLANK, EOS) or step >= limits[index]:
                            done[index] = True
                        else:
                            results[index].append(phone)
                if all(done):
                    break
                previous = best.unsqueeze(1)
            return results

    return ArModel()


# The letter ids of a word (unknown characters dropped, at most MAX_WORD letters)
def word_ids(word):
    return [LETTER_IDS[letter] for letter in word if letter in LETTER_IDS][:MAX_WORD]


# A batch of words as padded tensors - (letters [batch, time], targets [batch, steps] ending with <eos>)
def make_batch(words, pronunciations):
    import torch  # pylint: disable=import-outside-toplevel
    letters = [word_ids(word) or [LETTER_IDS['\'']] for word in words]
    targets = [[PHONE_IDS[phone] for phone in phones] + [EOS] for phones in pronunciations]
    letter_tensor = torch.zeros(len(words), max(len(ids) for ids in letters), dtype=torch.long)
    target_tensor = torch.zeros(len(words), max(len(ids) for ids in targets), dtype=torch.long)
    for index, (letter_ids, target_ids) in enumerate(zip(letters, targets)):
        letter_tensor[index, :len(letter_ids)] = torch.tensor(letter_ids, dtype=torch.long)
        target_tensor[index, :len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
    return letter_tensor, target_tensor


# Predict words' phones with a model - a list of phone lists
def predict(model, words, batch_size=512):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            chunk = words[start:start + batch_size]
            letters = make_batch(chunk, [[] for _ in chunk])[0]
            results.extend([PHONES[phone] for phone in phones] for phones in model.decode(letters))
    return results


# The learning rate at a step - linear warmup, then cosine decay to 2% of the peak
def learning_rate(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


#
# The measures
#

def evaluate(pronounce, data):
    """A pronouncer's measures (pronounce: a list of words to their phone lists) - {book, bookAny, bookWrongDistinct,
    top10kWrong, monitorWer, wrong (the most frequent wrong book words)}"""
    counts = data['evalCounts']
    words = list(counts)
    predictions = dict(zip(words, pronounce(words)))
    first, allprons = data['first'], data['all']
    tokens = sum(counts.values())
    wrong_first = [word for word in words if predictions[word] != first[word]]
    wrong_any = {word for word in wrong_first if predictions[word] not in allprons[word]}
    top = set(words[:10000])
    monitor = data['monitor']
    monitor_wrong = sum(1 for word, phones in zip(monitor, pronounce(monitor)) if phones not in allprons[word])
    return {
        'book': sum(counts[word] for word in wrong_first) / tokens,
        'bookAny': sum(counts[word] for word in wrong_any) / tokens,
        'bookWrongDistinct': len(wrong_first),
        'top10kWrong': sum(1 for word in wrong_first if word in top),
        'monitorWer': monitor_wrong / len(monitor),
        'wrong': [[word, counts[word], ' '.join(predictions[word]), ' '.join(first[word])] for word in wrong_first[:30]]
    }


# A one-line summary of the measures
def describe(result):
    return (f'book err {100 * result["book"]:.3f}% (any {100 * result["bookAny"]:.3f}%, {result["bookWrongDistinct"]} distinct), '
            f'top-10K wrong {result["top10kWrong"]}, monitor WER {100 * result["monitorWer"]:.2f}%')


#
# Training
#

def train_stage(settings, stage, init, out_path, data, log):
    """Train one stage - its final weights to out_path; the measures"""
    import torch  # pylint: disable=import-outside-toplevel
    settings = {**settings, **stage}
    torch.set_num_threads(int(settings.get('threads', 1)))
    torch.manual_seed(int(settings['seed']))
    rng = numpy.random.default_rng(int(settings['seed']))
    model = build_model(settings['model'])
    if init:
        model.load_state_dict(torch.load(init, weights_only=False)['state'])
    steps, batch_size, peak = int(settings['steps']), int(settings['batch']), float(settings['lr'])
    warmup = int(settings['warmup'])
    items, probabilities = training_items(data, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.0)
    log(f'{stage["name"]}: {sum(parameter.numel() for parameter in model.parameters())} parameters, {len(items)} items, '
        f'{steps} steps from {init or "scratch"}')
    started = window_start = time.time()
    window_loss = 0.0
    batches = []
    model.train()
    for step in range(1, steps + 1):
        if not batches:
            # The next pool of batches - sampled, sorted by word length (less padding), and shuffled as batches
            pool = rng.choice(len(items), size=batch_size * 32, p=probabilities)
            pool = sorted(pool.tolist(), key=lambda index: (len(items[index][0]), len(items[index][1])))
            batches = [pool[start:start + batch_size] for start in range(0, len(pool), batch_size)]
            rng.shuffle(batches)
        batch = [items[index] for index in batches.pop()]
        letters, targets = make_batch([word for word, _ in batch], [phones for _, phones in batch])
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, steps, peak, warmup)
        loss = model.loss(letters, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        window_loss += loss.item()
        if step % 1000 == 0 or step == steps:
            count = step % 1000 or 1000
            log(f'{stage["name"]}: step {step}/{steps} loss {window_loss / count:.4f} '
                f'({(time.time() - window_start) / count * 1000:.1f} ms/step)')
            window_loss, window_start = 0.0, time.time()
    torch.save({'config': model.config, 'state': model.state_dict(), 'stage': stage['name'], 'settings': settings},
               out_path + '.tmp')
    os.replace(out_path + '.tmp', out_path)
    result = evaluate(lambda words: predict(model, words), data)
    log(f'{stage["name"]}: done in {time.time() - started:.0f} s - {describe(result)}')
    return result


def command_train(args):
    """Every stage of the recipe, each resumed if its checkpoint exists - <out>/<stage>.pt, train.log, done.json"""
    config = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'train.log')

    def log(text):
        log_line(log_path, text)
    settings = {key: value for key, value in config.items() if key != 'stages'}
    init = None
    results = {}
    for stage in config['stages']:
        stage = {**stage, **({'steps': args.steps} if args.steps else {})}
        path = os.path.join(args.out, f'{stage["name"]}.pt')
        if os.path.exists(path):
            log(f'{stage["name"]}: done before ({path})')
        else:
            relabel = stage.get('relabel', settings.get('relabel'))
            data = load_data(relabel and os.path.join(ROOT, relabel))
            results[stage['name']] = train_stage(settings, stage, init, path, data, log)
        init = path
    with open(os.path.join(args.out, 'done.json'), 'w', encoding='utf-8') as file:
        json.dump({'checkpoint': init, 'config': config, 'results': results}, file, indent=1)
    return 0


#
# The export's quantization (context.py export) - each tensor's values are symmetric integers of its bits (matrices
# "bits", vectors - biases - "biasBits") with a bfloat16 scale per row; each scale is the best of CLIP_RATIOS by squared
# error. The bytes are every scale, then every tensor's values - bit-packed, or a byte each - deflated when that makes
# the module smaller.
#

# Round float32 values to bfloat16 (nearest, ties to even) - (uint16 codes, values as float64)
def bfloat16(values):
    bits = numpy.asarray(values, dtype=numpy.float32).view(numpy.uint32).astype(numpy.uint64)
    codes = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(numpy.uint16)
    return codes, (codes.astype(numpy.uint32) << 16).view(numpy.float32).astype(numpy.float64)


# Quantize a 2-D array - (integers, scale codes [rows], dequantized values)
def quantize_array(values, bits):
    limit = (1 << (bits - 1)) - 1
    peak = numpy.abs(values).max(1)
    best_error, best_codes, best_scales = None, None, None
    for ratio in CLIP_RATIOS:
        codes, scales = bfloat16(peak * ratio / limit)
        safe = numpy.where(scales > 0, scales, 1.0)
        integers = numpy.clip(numpy.round(values / safe[:, None]), -limit, limit) * (scales > 0)[:, None]
        error = ((integers * scales[:, None] - values) ** 2).sum(1)
        if best_error is None:
            best_error, best_codes, best_scales = error, codes, scales
        else:
            better = error < best_error
            best_error = numpy.where(better, error, best_error)
            best_codes = numpy.where(better, codes, best_codes)
            best_scales = numpy.where(better, scales, best_scales)
    safe = numpy.where(best_scales > 0, best_scales, 1.0)
    integers = (numpy.clip(numpy.round(values / safe[:, None]), -limit, limit) * (best_scales > 0)[:, None]).astype(numpy.int64)
    return integers, best_codes, integers * best_scales[:, None]


# Quantize a model's state - [{name, shape, bits, integers, codes, dequantized}] in state order
def quantize_state(state, spec):
    quantized = []
    for name, tensor in state.items():
        values = tensor.detach().double().cpu().numpy()
        bits = spec['bits'] if values.ndim == 2 else spec.get('biasBits', 8)
        integers, codes, dequantized = quantize_array(values.reshape(1, -1) if values.ndim == 1 else values, bits)
        quantized.append({'name': name, 'shape': list(values.shape), 'bits': bits, 'integers': integers, 'codes': codes,
                          'dequantized': dequantized.reshape(values.shape)})
    return quantized


# The weight bytes of a quantized state - the scales, then the values in a layout ("packed" or "bytes")
def weight_bytes(quantized, layout):
    parts = [item['codes'].astype('<u2').tobytes() for item in quantized]
    for item in quantized:
        unsigned = (item['integers'].reshape(-1) + ((1 << (item['bits'] - 1)) - 1)).astype(numpy.uint8)
        if layout == 'bytes':
            parts.append(unsigned.tobytes())
        else:
            stream = ((unsigned[:, None] >> numpy.arange(item['bits'])) & 1).astype(numpy.uint8).reshape(-1)
            parts.append(numpy.packbits(stream, bitorder='little').tobytes())
    return b''.join(parts)


def command_evaluate(args):
    """The measures of a generated module (as the app pronounces - g2pmodel.py) or a checkpoint"""
    data = load_data(args.relabel)
    if args.model.endswith('.js'):
        g2p = g2pmodel.G2p(g2pmodel.read_module(args.model))
        result = evaluate(g2p.pronounce_many, data)
    else:
        import torch  # pylint: disable=import-outside-toplevel
        torch.set_num_threads(args.threads)
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        model = build_model(checkpoint['config'])
        model.load_state_dict(checkpoint['state'])
        result = evaluate(lambda words: predict(model, words), data)
    log_line(None, f'evaluate {args.model}: {describe(result)}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=1)
    return 0


def command_book(args):
    """A book's word counts (the front end's words - frontend.text_items) into the books file - the words the network
    trains on (the dictionary's, the labels', and their possessives)"""
    with open(args.text, encoding='utf-8') as file:
        items = frontend.text_items(file.read())
    books = read_json(args.books)
    known = set(read_pronunciations()) | set(books['labels'])
    counts = collections.Counter(item for item, _ in items if isinstance(item, str))
    kept = {word: count for word, count in sorted(counts.items()) if word in known or (word.endswith('\'s') and word[:-2] in known)}
    books['books'][args.name] = {'title': args.title, 'url': args.url, 'tokens': sum(counts.values()), 'counts': kept}
    write_books(args.books, books)
    log_line(None, f'book {args.name}: {sum(counts.values())} tokens, {len(kept)} of its {len(counts)} words kept')
    return 0


# Write the books file - its labels one word a line, and each book's counts on one line
def write_books(path, books):
    lines = ['{', f' "description": {json.dumps(books["description"])},', ' "labels": {']
    lines += [f'  {json.dumps(word)}: {json.dumps(phones)},' for word, phones in sorted(books['labels'].items())]
    lines[-1] = lines[-1].rstrip(',')
    lines += [' },', ' "books": {']
    entries = [f'  {json.dumps(name)}: {json.dumps(book)}' for name, book in sorted(books['books'].items())]
    lines += [entry + ',' for entry in entries[:-1]] + entries[-1:]
    lines += [' }', '}']
    with open(path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.g2p', description='Train and evaluate the G2P network')
    commands = parser.add_subparsers(dest='command', required=True)
    train_parser = commands.add_parser('train', help='every stage of the recipe (resumable)')
    train_parser.add_argument('--out', required=True, help='the output directory')
    train_parser.add_argument('--config', default=RECIPE_PATH)
    train_parser.add_argument('--steps', type=int, help='each stage this many steps (a smoke test)')
    train_parser.set_defaults(func=command_train)
    evaluate_parser = commands.add_parser('evaluate', help='the measures of a module or a checkpoint')
    evaluate_parser.add_argument('model', nargs='?', default=MODULE_PATH)
    evaluate_parser.add_argument('--relabel', help='the relabel file the targets follow')
    evaluate_parser.add_argument('--threads', type=int, default=4)
    evaluate_parser.add_argument('--out', help='the measures as JSON')
    evaluate_parser.set_defaults(func=command_evaluate)
    book_parser = commands.add_parser('book', help='count a listened-to book\'s words into train/data/g2p-books.json')
    book_parser.add_argument('text', help='the book\'s text (UTF-8)')
    book_parser.add_argument('--name', required=True, help='the book\'s name (its key)')
    book_parser.add_argument('--title', required=True)
    book_parser.add_argument('--url', required=True, help='the text\'s source')
    book_parser.add_argument('--books', default=BOOKS_PATH)
    book_parser.set_defaults(func=command_book)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
