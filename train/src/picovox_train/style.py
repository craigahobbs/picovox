# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The style network - a quotation's speaking style, neutral, angry, shouting, or fearful, for the package's style "auto"
(lib/style.js runs lib/models/styleModel.js). Listeners preferred the high-energy styles on the quoted words alone, and
the quieter styles made a book worse - so only these, and a quotation is neutral unless the network is confident. The
commands (python is PYTHONPATH=train/src build/venv-nn/bin/python -B):

    python -m picovox_train.style quotations --out DIR      # the pinned books' quotations (nn/js/quoteLines.js)
    python -m picovox_train.style label --out DIR           # a teacher's labels (Ollama) - data/nn/style-labels.json
    python -m picovox_train.style train --out DIR           # the network - DIR/style.pt
    python -m picovox_train.style export DIR/style.pt [--out lib/models/styleModel.js]
    python -m picovox_train.style evaluate [MODULE]         # against the hand-labeled quotations (data/nn/style-test.json)

The settings are data/nn/train.json's "style" section.

The data: every quotation of the pinned books (books.py - lib/style.js textQuotations, by nn/js/quoteLines.js, so the
quotations are the package's own), a sample labeled by a teacher - a local chat model (Ollama's /api/chat, called
directly, its answer constrained to the four styles, its probabilities from the answer's logprobs) - each part of the
books sampled at its own rate: "enrichedCount" of the exclamations with a high-energy word around them (the rare
styles' examples), "exclaimedCount" of the other exclamations, and "plainCount" of the rest. The labels are committed as
data/nn/style-labels.json (each with its quotation's book, index, and text digest - so make train needs no teacher, and
a changed text or splitting fails loudly). The hand-labeled quotations (data/nn/style-test.json) are never trained on.

The network (lib/style.js's): each quotation's words - its own, its attribution's (the narration beside it in its
sentences), and the neighboring sentences' - hashed (FNV-1a) into one embedding table, each kind's mean and max; cue
features (exclamation and question marks, capitals, emphasis, length, dashes); a tanh layer; and the styles' logits.
The word table is pretrained without labels (each quotation's attribution's words from its own and its context's), then
fixed; the rest is trained with cross entropy on the teacher's probabilities. Each style's threshold is the least
probability whose precision - over every label held out once (folds), weighted to the books' own mix - has a one-sided
95% lower bound of at least "precision" (so a style is chosen only with the evidence for it). The export quantizes to
int8 (a scale per tensor) and writes the module.
"""

import argparse
import collections
import concurrent.futures
import datetime
import functools
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.request

from .frontend import ROOT
from . import books as books_module


CONFIG_PATH = os.path.join(ROOT, 'data', 'nn', 'train.json')
LABELS_PATH = os.path.join(ROOT, 'data', 'nn', 'style-labels.json')
TEST_PATH = os.path.join(ROOT, 'data', 'nn', 'style-test.json')
MODULE_PATH = os.path.join(ROOT, 'lib', 'models', 'styleModel.js')
QUOTE_LINES = os.path.join(ROOT, 'nn', 'js', 'quoteLines.js')

CLASSES = ['neutral', 'angry', 'shouting', 'fearful']
CUE_COUNT = 8

# The module's line length
MODULE_LINE_LENGTH = 120

# A quotation with these around it (and an exclamation) is a high-energy candidate - the enriched sample
CUE_REGEX = re.compile(r'\b(shout|roar|bellow|yell|scream|shriek|cried|cry|thunder|angr|furious|rage|fierce|savage|terror|terrif|'
                       r'fright|afraid|alarm|trembl|gasp|panic)', re.IGNORECASE)

# The teacher's instructions
TEACHER_SYSTEM = (
    'You label lines of dialogue in novels for an audiobook narrator. For the quotation, answer how the narrator should voice '
    'the quoted words: "shouting" - the words are shouted, screamed, yelled, or called out loudly (a shout across a distance, '
    'a battle command, a crowd\'s roar, a scream); "angry" - real rage, fury, or a threatening, furious rebuke; "fearful" - '
    'terror, alarm, panic, a desperate plea for help; "neutral" - everything else: ordinary speech, mild irritation, surprise, '
    'excitement, joy, sadness, pleading, whispers, and exclamations that are only emphatic ("cried" often just means '
    '"exclaimed"). Most quotations are neutral; choose another label only when the text clearly calls for it.'
)


# A progress line - stderr, and the log file if given
def log_line(path, text):
    line = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {text}'
    print(line, file=sys.stderr, flush=True)
    if path:
        with open(path, 'a', encoding='utf-8') as file:
            file.write(line + '\n')


# The settings - data/nn/train.json's "style"
def load_config(path=CONFIG_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)['style']


#
# The features - lib/style.js's, exactly
#

WORD_REGEX = re.compile(r"[a-z]+(?:'[a-z]+)*")
MARKS_REGEX = re.compile('[̀-ͯ]')
LETTER_REGEX = re.compile(r'[^\W\d_]')
DASH_REGEX = re.compile('[—-]{2,}|—')


# A text's words for the features (styleWords)
def style_words(text):
    plain = MARKS_REGEX.sub('', unicodedata.normalize('NFD', text)).lower().replace('‘', "'").replace('’', "'")
    return WORD_REGEX.findall(plain)


# The FNV-1a hash of a string's UTF-16 code units (fnv1a)
def fnv1a(text):
    value = 0x811c9dc5
    units = text.encode('utf-16-le')
    for index in range(0, len(units), 2):
        value ^= units[index] | (units[index + 1] << 8)
        value = (value * 0x01000193) & 0xffffffff
    return value


# A word's hash (cached - the books have a few hundred thousand words)
@functools.lru_cache(maxsize=None)
def word_hash(word):
    return fnv1a(word)


# A text's word buckets (wordBuckets) - one table for every kind of text
def word_buckets(text, buckets):
    return [word_hash(word) % buckets for word in style_words(text)]


# A quotation's cue features (styleCues)
def style_cues(quotation):
    text = quotation['text']
    letters = LETTER_REGEX.findall(text)
    capitals = sum(1 for letter in letters if letter != letter.lower())
    words = len(style_words(text))
    attribution = f'{quotation.get("before", "")} {quotation.get("after", "")}'
    return [
        min(3, text.count('!')) / 3,
        1.0 if '?' in text else 0.0,
        capitals / len(letters) if letters else 0.0,
        1.0 if '_' in text else 0.0,
        math.log1p(words) / 4,
        1.0 if words <= 3 else 0.0,
        1.0 if '!' in attribution else 0.0,
        1.0 if DASH_REGEX.search(text) else 0.0
    ]


# A quotation's three kinds of text - its own, its attribution's, and its neighbors'
def kind_texts(quotation):
    return [quotation['text'], f'{quotation.get("before", "")} {quotation.get("after", "")}',
            f'{quotation.get("previous", "")} {quotation.get("next", "")}']


#
# The quotations and their labels
#

# A quotation's text digest - its label's check against a changed text or splitting
def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]


# The pinned books' quotations - {book: [quotation]} (nn/js/quoteLines.js, the books read as the package reads them)
def book_quotations(names, books_dir=books_module.BOOKS_DIR):
    texts = []
    for name in names:
        with open(os.path.join(books_dir, f'{name}.txt'), encoding='utf-8', newline='') as file:
            texts.append(file.read())
    process = subprocess.run(['node', QUOTE_LINES], input=''.join(json.dumps(text) + '\n' for text in texts), capture_output=True,
                             text=True, check=True, cwd=ROOT)
    return {name: json.loads(line) for name, line in zip(names, process.stdout.splitlines())}


# Is a quotation a high-energy candidate - an exclamation, and a cue word around it?
def is_enriched(quotation):
    context = ' '.join([quotation['before'], quotation['after'], quotation['previous'], quotation['next']])
    return '!' in quotation['text'] and CUE_REGEX.search(context) is not None


# A quotation's part of the books - "enriched" (an exclamation, and a high-energy word around it), "exclaimed" (another
# exclamation), or "plain" - each sampled uniformly, at its own rate (nearly every styled quotation exclaims)
def partition(quotation):
    if '!' not in quotation['text']:
        return 'plain'
    return 'enriched' if is_enriched(quotation) else 'exclaimed'


# The labeling pool - [(book, quotation)] within the length limits, the hand-labeled ones left out
def labeling_pool(quotations, settings, excluded=frozenset()):
    return [(book, quotation) for book in sorted(quotations) for quotation in quotations[book]
            if settings['minChars'] <= len(quotation['text']) <= settings['maxChars']
            and f'{book}:{quotation["quotation"]}' not in excluded]


# The quotations to label - [(book, quotation, stratum)]: each part's "<part>Count" of its quotations (seeded), in order
# enriched, exclaimed, plain (the most informative first); and the parts' sizes
def sample_quotations(quotations, settings, excluded):
    pool = labeling_pool(quotations, settings, excluded)
    rng = random.Random(settings['seed'])
    sample, sizes = [], {}
    for part in ('enriched', 'exclaimed', 'plain'):
        items = [item for item in pool if partition(item[1]) == part]
        sizes[part] = len(items)
        chosen = rng.sample(items, min(settings[f'{part}Count'], len(items)))
        sample += [(book, quotation, part) for book, quotation in chosen]
    return sample, sizes


# The teacher's label of a quotation and its probabilities of the four styles (Ollama's /api/chat, the answer constrained
# to the styles - the probabilities from the first token of its answer's label, by each style's first letters; a bare
# string answer is 15% faster but a worse teacher - accuracy 0.880 against 0.907 on the hand labels)
def teacher_label(quotation, model, url='http://127.0.0.1:11434'):
    user = (f'Before: {quotation["previous"][-400:]}\nQuotation (with the narration around it): {quotation["before"]} '
            f'{quotation["text"][:600]} {quotation["after"]}\nAfter: {quotation["next"][:300]}')
    body = {'model': model, 'stream': False, 'think': False, 'options': {'temperature': 0}, 'logprobs': True, 'top_logprobs': 10,
            'format': {'type': 'object', 'properties': {'label': {'type': 'string', 'enum': CLASSES}}, 'required': ['label']},
            'messages': [{'role': 'system', 'content': TEACHER_SYSTEM}, {'role': 'user', 'content': user}]}
    request = urllib.request.Request(f'{url}/api/chat', json.dumps(body).encode('utf-8'), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        answer = json.loads(response.read())
    content = answer['message']['content']
    label = json.loads(content)['label']
    return label, label_probabilities(content, label, answer.get('logprobs') or [])


# The styles' probabilities at an answer's label - the token where the label starts, its alternatives each a style's
# first letters (normalized; None if the answer has no logprobs there)
def label_probabilities(content, label, logprobs):
    start = content.index(f'"{label}"') + 1
    position = 0
    for token in logprobs:
        end = position + len(bytes(token['bytes']).decode('utf-8', errors='replace'))
        if end > start:
            totals = [0.0] * len(CLASSES)
            for alternative in token['top_logprobs']:
                text = bytes(alternative['bytes']).decode('utf-8', errors='replace').lstrip(' "')
                for index, name in enumerate(CLASSES):
                    if text and name.startswith(text) and alternative['logprob'] > -100:
                        totals[index] += math.exp(alternative['logprob'])
            total = sum(totals)
            return [round(value / total, 4) for value in totals] if total > 0 else None
        position = end
    return None


# Read the labels file - {description, teacher, counts, labels: {book: [[quotation, label, stratum, digest]]}}
def read_labels(path=LABELS_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# Write the labels file - a book's labels on one line
def write_labels(path, labels):
    head = {key: value for key, value in labels.items() if key != 'labels'}
    lines = ['{'] + [f' {json.dumps(key)}: {json.dumps(value)},' for key, value in head.items()] + [' "labels": {']
    entries = [f'  {json.dumps(book)}: {json.dumps(rows)}' for book, rows in sorted(labels['labels'].items())]
    lines += [entry + ',' for entry in entries[:-1]] + entries[-1:] + [' }', '}']
    with open(f'{path}.tmp', 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines) + '\n')
    os.replace(f'{path}.tmp', path)


def command_quotations(args):
    """The pinned books' quotations - <out>/quotations.json"""
    with open(args.config, encoding='utf-8') as file:
        names = json.load(file)['context']['labels']['texts']
    quotations = book_quotations(names)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'quotations.json'), 'w', encoding='utf-8') as file:
        json.dump(quotations, file)
    log_line(args.log, f'quotations: {sum(len(items) for items in quotations.values())} in {len(quotations)} books')
    return 0


def command_label(args):
    """The teacher's labels of the sample - data/nn/style-labels.json (resumable: labels written before are kept)"""
    settings = load_config(args.config)
    with open(os.path.join(args.out, 'quotations.json'), encoding='utf-8') as file:
        quotations = json.load(file)
    with open(args.test, encoding='utf-8') as file:
        excluded = {item['id'] for item in json.load(file)['quotations']}
    sample, sizes = sample_quotations(quotations, settings, excluded)
    labels = read_labels(args.labels) or {
        'description': ('The style network\'s training labels (picovox_train/style.py label): a teacher\'s style of a sample of '
                        'the pinned books\' quotations - each [quotation index, label, stratum, text digest] by book'),
        'teacher': settings['teacher'], 'labels': {}}
    done = {(book, row[0]) for book, rows in labels['labels'].items() for row in rows}
    # The exclamations first, the two parts interleaved (seeded), then the plain quotations
    todo = [item for item in sample if (item[0], item[1]['quotation']) not in done]
    exclamations = [item for item in todo if item[2] != 'plain']
    random.Random(settings['seed']).shuffle(exclamations)
    todo = exclamations + [item for item in todo if item[2] == 'plain']
    log_line(args.log, f'label: the parts {sizes}')
    log_line(args.log, f'label: {len(todo)} of {len(sample)} quotations to label ({settings["teacher"]}, {args.jobs} at a time)')
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        answers = executor.map(lambda item: teacher_label(item[1], settings['teacher'], args.url), todo)
        for index, ((book, quotation, stratum), (label, probabilities)) in enumerate(zip(todo, answers)):
            row = [quotation['quotation'], label, stratum, digest(quotation['text'])]
            rows = [item for item in labels['labels'].get(book, []) if item[0] != row[0]]
            labels['labels'][book] = rows + [row + ([probabilities] if probabilities is not None else [])]
            if (index + 1) % 200 == 0 or index + 1 == len(todo):
                for rows in labels['labels'].values():
                    rows.sort()
                write_labels(args.labels, labels)
                log_line(args.log, f'label: {index + 1}/{len(todo)}')
    return 0


# The labeled examples - [(quotation, label index, target (the teacher's probabilities, else its label's))], each label's
# quotation checked by its digest
def labeled_examples(quotations, labels):
    examples = []
    for book, rows in sorted(labels['labels'].items()):
        by_index = {quotation['quotation']: quotation for quotation in quotations[book]}
        for index, label, _stratum, text_digest, *soft in rows:
            quotation = by_index.get(index)
            if quotation is None or digest(quotation['text']) != text_digest:
                raise ValueError(f'{book} quotation {index}: its text changed - relabel (python -m picovox_train.style label)')
            target = soft[0] if soft else [1.0 if name == label else 0.0 for name in CLASSES]
            examples.append((quotation, CLASSES.index(label), target))
    return examples


# Each example's weight - the inverse of its chance of being sampled (a random draw, or an enriched one too), normalized
def sample_weights(examples, quotations, settings, power=1):
    # A part's quotations were each labeled with the chance (labeled)/(size), whatever their draw
    sizes = collections.Counter(partition(quotation) for _, quotation in labeling_pool(quotations, settings))
    labeled = collections.Counter(partition(quotation) for quotation, _, _ in examples)
    weights = [(sizes[partition(quotation)] / labeled[partition(quotation)]) ** power for quotation, _, _ in examples]
    mean = sum(weights) / len(weights)
    return [weight / mean for weight in weights]


#
# The network
#

def build_model(config, dropout=0.0, heads=0):
    import torch  # pylint: disable=import-outside-toplevel

    class StyleModel(torch.nn.Module):
        """The style network (lib/style.js's)"""

        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.EmbeddingBag(config['buckets'], config['dim'], mode='mean')
            self.hidden = torch.nn.Linear(6 * config['dim'] + CUE_COUNT, config['hidden'])
            self.output = torch.nn.Linear(config['hidden'], len(CLASSES))
            self.dropout = torch.nn.Dropout(dropout)
            # The pretraining's head - the attribution's words (not exported)
            self.attribution = torch.nn.Linear(config['hidden'], heads) if heads else None

        # Each kind's mean and max embedding (an empty kind's zero), and the cues, through the tanh layer
        def encode(self, indexes, offsets, cues):
            count = cues.shape[0]
            mean = self.embedding(indexes, offsets).reshape(count, 3 * config['dim'])
            peak = torch.nn.functional.embedding_bag(indexes, self.embedding.weight, offsets, mode='max').reshape(count, 3 * config['dim'])
            return self.dropout(torch.tanh(self.hidden(self.dropout(torch.cat([mean, peak, cues], dim=1)))))

        def forward(self, indexes, offsets, cues, pretraining=False):
            return (self.attribution if pretraining else self.output)(self.encode(indexes, offsets, cues))

    return StyleModel()


# A quotation's features - its kinds' word buckets and its cues (lib/style.js's)
def quotation_features(quotation, buckets):
    return [word_buckets(text, buckets) for text in kind_texts(quotation)], style_cues(quotation)


# A batch's tensors from features - the kinds' bags (the attribution's left out, and its cue, for the pretraining) and cues
def batch_tensors(features, hide_attribution=False):
    import torch  # pylint: disable=import-outside-toplevel
    indexes, offsets, cues = [], [], []
    for bags, quotation_cues in features:
        for kind, bag in enumerate(bags):
            offsets.append(len(indexes))
            if not (hide_attribution and kind == 1):
                indexes.extend(bag)
        cues.append([0.0 if hide_attribution and index == 6 else value for index, value in enumerate(quotation_cues)])
    return torch.tensor(indexes, dtype=torch.long), torch.tensor(offsets, dtype=torch.long), torch.tensor(cues, dtype=torch.float32)


# The pretraining's examples - [(features, the attribution's words' indexes)] of the attributed quotations, the
# attribution's "top" most common words its targets
def attribution_examples(quotations, features, top):
    words = [set(style_words(f'{quotation["before"]} {quotation["after"]}')) for quotation in quotations]
    counts = collections.Counter(word for item in words for word in item)
    vocabulary = {word: index for index, (word, _) in enumerate(counts.most_common(top))}
    return [(feature, [vocabulary[word] for word in item if word in vocabulary]) for feature, item in zip(features, words)
            if any(word in vocabulary for word in item)]


# Pretrain the network on the attribution's words - each quotation's attribution's words from its own and its context's
# (the attribution hidden): the word table learns what a shouted, angry, or frightened quotation reads like
def pretrain(model, examples, settings, log=None):
    import torch  # pylint: disable=import-outside-toplevel
    rng = random.Random(settings['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weightDecay'])
    steps = int(settings['pretrainEpochs'] * len(examples) / settings['pretrainBatch'])
    heads = model.attribution.out_features
    for step in range(1, steps + 1):
        batch = [examples[rng.randrange(len(examples))] for _ in range(settings['pretrainBatch'])]
        targets = torch.zeros(len(batch), heads)
        for row, (_, words) in enumerate(batch):
            targets[row, words] = 1
        logits = model(*batch_tensors([feature for feature, _ in batch], True), pretraining=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0:
            log_line(log, f'train: pretraining step {step}/{steps} loss {loss.item():.4f}')


# A quotation's fold - its book and index hashed
def fold_of(book, index, folds):
    return int(hashlib.sha256(f'{book}:{index}'.encode('utf-8')).hexdigest()[:8], 16) % folds


# Fine-tune the network on weighted examples ([((quotation, label, target), weight)]) - the word table fixed
def fine_tune(model, train, features, settings, rng, steps=None):
    import torch  # pylint: disable=import-outside-toplevel
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=settings['lr'],
                                  weight_decay=settings['weightDecay'])
    for _ in range(steps or int(settings['epochs'] * len(train) / settings['batch'])):
        batch = [train[rng.randrange(len(train))] for _ in range(settings['batch'])]
        targets = torch.tensor([example[2] for example, _ in batch], dtype=torch.float32)
        batch_weights = torch.tensor([weight for _, weight in batch], dtype=torch.float32)
        logits = model(*batch_tensors([features[id(example[0])] for example, _ in batch]))
        loss = (torch.nn.functional.cross_entropy(logits, targets, reduction='none') * batch_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# The probabilities of features by a model (a trained torch model)
def predict(model, features, batch_size=512):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            rows.extend(torch.softmax(model(*batch_tensors(features[start:start + batch_size])), dim=1).tolist())
    model.train()
    return rows


# Each style's threshold - the least probability whose precision on the examples (weighted) is at least the target -
# with the precision and recall there; a style that never reaches the target has threshold 1.01 (never chosen)
def choose_thresholds(probabilities, labels, weights, target):
    thresholds = {}
    for index, name in enumerate(CLASSES[1:], 1):
        # Down the ranking, the picks at or above each distinct probability - their right, total, and squared weights
        ranked = sorted(zip((row[index] for row in probabilities), labels, weights), key=lambda item: -item[0])
        best = 1.01
        right = total = squares = 0.0
        for position, (probability, label, weight) in enumerate(ranked):
            right += weight if label == index else 0.0
            total += weight
            squares += weight * weight
            if position + 1 < len(ranked) and ranked[position + 1][0] == probability:
                continue
            if precision_bound(right, total, squares) >= target:
                best = probability
        thresholds[name] = best
    return thresholds


# The one-sided 95% lower bound of a weighted precision - the Wilson bound at the weights' effective count, so a few
# right picks can't pass the gate (two of two bound at 0.42; a clean 95% needs about fifty)
CONFIDENCE_Z = 1.645


def precision_bound(right, total, squares):
    count = total * total / squares
    precision = right / total
    z2 = CONFIDENCE_Z * CONFIDENCE_Z
    center = precision + z2 / (2 * count)
    spread = CONFIDENCE_Z * math.sqrt(precision * (1 - precision) / count + z2 / (4 * count * count))
    return (center - spread) / (1 + z2 / count)


# The styles chosen by thresholds - each quotation's best style that reaches its threshold, else neutral (quotationStyle)
def choose_styles(probabilities, thresholds):
    styles = []
    for row in probabilities:
        best, best_probability = 0, 0.0
        for index, name in enumerate(CLASSES[1:], 1):
            if row[index] >= thresholds[name] and row[index] > best_probability:
                best, best_probability = index, row[index]
        styles.append(best)
    return styles


# The measures of chosen styles against labels - {accuracy, styled (count), precision (a styled quotation's style right),
# styledPrecision (styled at all, rightly), recall (a styled label's style found), neutralStyled (neutral labels styled),
# confusion}
def measures(chosen, labels):
    styled = [(choice, label) for choice, label in zip(chosen, labels) if choice != 0]
    labeled = sum(1 for label in labels if label != 0)
    confusion = collections.Counter((CLASSES[label], CLASSES[choice]) for choice, label in zip(chosen, labels))
    return {
        'accuracy': sum(1 for choice, label in zip(chosen, labels) if choice == label) / max(1, len(labels)),
        'styled': len(styled),
        'precision': sum(1 for choice, label in styled if choice == label) / max(1, len(styled)),
        'styledPrecision': sum(1 for _, label in styled if label != 0) / max(1, len(styled)),
        'recall': sum(1 for choice, label in styled if choice == label) / max(1, labeled),
        'neutralStyled': sum(1 for choice, label in styled if label == 0),
        'confusion': {f'{label}>{choice}': count for (label, choice), count in sorted(confusion.items())}
    }


def command_train(args):
    """The network - <out>/style.pt (its config, weights, thresholds, and validation measures)"""
    import torch  # pylint: disable=import-outside-toplevel
    settings = load_config(args.config)
    with open(os.path.join(args.out, 'quotations.json'), encoding='utf-8') as file:
        quotations = json.load(file)
    with open(args.test, encoding='utf-8') as file:
        excluded = {item['id'] for item in json.load(file)['quotations']}
    config = settings['network']
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(settings['seed'])
    rng = random.Random(settings['seed'])

    # Every quotation's features (the hand-labeled ones left out)
    pool = labeling_pool(quotations, settings, excluded)
    features = {id(quotation): quotation_features(quotation, config['buckets']) for _, quotation in pool}
    log_line(args.log, f'train: {len(pool)} quotations\' features')

    # Pretrain on the attribution's words, then fix the word table
    model = build_model(config, settings['dropout'], settings['pretrainTop'])
    pretraining = attribution_examples([quotation for _, quotation in pool], [features[id(quotation)] for _, quotation in pool],
                                       settings['pretrainTop'])
    log_line(args.log, f'train: pretraining on {len(pretraining)} attributed quotations')
    pretrain(model, pretraining, settings, args.log)
    model.embedding.weight.requires_grad = False

    # The teacher's labels (weightPower tempers the examples' weights; the thresholds' are the books' own mix), in folds
    examples = [example for example in labeled_examples(quotations, read_labels(args.labels)) if id(example[0]) in features]
    weights = sample_weights(examples, quotations, settings, settings['weightPower'])
    population = sample_weights(examples, quotations, settings)
    books_of = {id(quotation): book for book, items in quotations.items() for quotation in items}
    folds = [fold_of(books_of[id(quotation)], quotation['quotation'], settings['folds']) for quotation, _, _ in examples]
    pretrained = {key: value.clone() for key, value in model.state_dict().items()}

    # Each fold's probabilities by a network trained on the others - the thresholds' validation, every label held out once
    probabilities = [None] * len(examples)
    for fold in range(settings['folds']):
        model.load_state_dict(pretrained)
        train = [(example, weight) for example, weight, other in zip(examples, weights, folds) if other != fold]
        fine_tune(model, train, features, settings, rng, args.steps)
        held = [index for index, other in enumerate(folds) if other == fold]
        for index, row in zip(held, predict(model, [features[id(examples[index][0])] for index in held])):
            probabilities[index] = row
    labels_of = [example[1] for example in examples]
    with open(os.path.join(args.out, 'heldout.json'), 'w', encoding='utf-8') as file:
        json.dump([{'id': f'{books_of[id(example[0])]}:{example[0]["quotation"]}', 'label': example[1], 'part': partition(example[0]),
                    'weight': weight, 'probabilities': row} for example, weight, row in zip(examples, population, probabilities)], file)
    thresholds = choose_thresholds(probabilities, labels_of, population, settings['precision'])
    result = measures(choose_styles(probabilities, thresholds), labels_of)
    log_line(args.log, f'train: {len(examples)} examples in {settings["folds"]} folds - thresholds {thresholds}; held out (the teacher\'s '
                       f'labels) {describe(result)}')

    # The network - trained on every label
    model.load_state_dict(pretrained)
    fine_tune(model, list(zip(examples, weights)), features, settings, rng, args.steps)
    state = {key: value for key, value in model.state_dict().items() if not key.startswith('attribution.')}
    torch.save({'config': config, 'state': state, 'thresholds': thresholds, 'validation': result}, os.path.join(args.out, 'style.pt'))
    return 0


# A measures line
def describe(result):
    return (f'accuracy {result["accuracy"]:.3f}, {result["styled"]} styled - precision {result["precision"]:.3f} '
            f'(styled rightly {result["styledPrecision"]:.3f}), recall {result["recall"]:.3f}, {result["neutralStyled"]} neutral styled')


#
# The export
#

# Quantize a tensor to int8 - (the bytes, the scale)
def quantize(values):
    import numpy  # pylint: disable=import-outside-toplevel
    array = numpy.asarray(values, dtype=numpy.float32).reshape(-1)
    scale = float(numpy.abs(array).max()) / 127 or 1.0
    return numpy.clip(numpy.round(array / scale), -127, 127).astype(numpy.int8).tobytes(), scale


# The generated module's text
def module_text(checkpoint, comment):
    import base64  # pylint: disable=import-outside-toplevel
    state = checkpoint['state']
    tensors = {'embedding': state['embedding.weight'], 'hidden': state['hidden.weight'], 'hiddenBias': state['hidden.bias'],
               'output': state['output.weight'], 'outputBias': state['output.bias']}
    data, scales = b'', {}
    for name, tensor in tensors.items():
        quantized, scale = quantize(tensor.detach().cpu().numpy())
        data += quantized
        scales[name] = scale
    text = base64.b64encode(data).decode('ascii')
    lines = [f'// {line}' for line in comment] + ['', 'export const styleModel = {', '    \'version\': 2,',
             f'    \'config\': {json.dumps(checkpoint["config"]).replace(chr(34), chr(39))},',
             f'    \'thresholds\': {json.dumps(checkpoint["thresholds"]).replace(chr(34), chr(39))},',
             '    \'scales\': {', *[f'        \'{name}\': {scale}{"," if index + 1 < len(scales) else ""}'
                                   for index, (name, scale) in enumerate(scales.items())], '    },', '    \'weights\': `']
    lines += [text[start:start + MODULE_LINE_LENGTH] for start in range(0, len(text), MODULE_LINE_LENGTH)]
    lines += ['`', '};', '']
    return '\n'.join(lines)


def command_export(args):
    """The generated module - lib/models/styleModel.js"""
    import torch  # pylint: disable=import-outside-toplevel
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    digest_text = hashlib.sha256(b''.join(tensor.numpy().tobytes() for tensor in checkpoint['state'].values())).hexdigest()[:16]
    comment = ['Generated by picovox_train/style.py export - do not edit', '',
               f'The style network run by lib/style.js - {sum(t.numel() for t in checkpoint["state"].values())} parameters, int8 weights;',
               f'trained by picovox_train/style.py (weights sha256 {digest_text}).']
    with open(args.out, 'w', encoding='utf-8') as file:
        file.write(module_text(checkpoint, comment))
    log_line(None, f'export: wrote {args.out}')
    return 0


# A module's probabilities of quotations (lib/style.js, under Node - the package's own inference)
def module_probabilities(module_path, quotations):
    script = ("import {loadStyleModel, styleProbabilities} from './lib/style.js';"
              "import {readFileSync} from 'node:fs';"
              f"const module = await import({json.dumps(os.path.abspath(module_path))});"
              "const model = loadStyleModel(module);"
              "const quotations = JSON.parse(readFileSync(0, 'utf8'));"
              "process.stdout.write(JSON.stringify(quotations.map((quotation) => styleProbabilities(model, quotation))));")
    process = subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(quotations), capture_output=True,
                             text=True, check=True, cwd=ROOT)
    return json.loads(process.stdout)


def command_evaluate(args):
    """The module's measures against the hand-labeled quotations (data/nn/style-test.json) - all, and its random stratum"""
    with open(args.test, encoding='utf-8') as file:
        items = json.load(file)['quotations']
    probabilities = module_probabilities(args.model, items)
    script = (f'const m = await import({json.dumps(os.path.abspath(args.model))}); '
              'process.stdout.write(JSON.stringify(m.styleModel.thresholds));')
    process = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    thresholds = json.loads(process.stdout)
    chosen = choose_styles(probabilities, thresholds)
    labels = [CLASSES.index(item['label']) for item in items]
    for name, keep in (('all', lambda item: True), ('random', lambda item: item['stratum'] == 'random')):
        rows = [(choice, label) for choice, label, item in zip(chosen, labels, items) if keep(item)]
        result = measures([row[0] for row in rows], [row[1] for row in rows])
        log_line(None, f'evaluate {name} ({len(rows)}): {describe(result)}; {result["confusion"]}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.style', description='The style network')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, function, help_text in (('quotations', command_quotations, 'the pinned books\' quotations'),
                                      ('label', command_label, 'a teacher\'s labels (Ollama)'),
                                      ('train', command_train, 'the network')):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument('--out', required=True)
        sub.add_argument('--config', default=CONFIG_PATH)
        sub.add_argument('--labels', default=LABELS_PATH)
        sub.add_argument('--test', default=TEST_PATH)
        sub.add_argument('--url', default='http://127.0.0.1:11434')
        sub.add_argument('--steps', type=int)
        sub.add_argument('--jobs', type=int, default=1)
        sub.add_argument('--log')
        sub.set_defaults(func=function)
    export_parser = commands.add_parser('export', help='the generated module')
    export_parser.add_argument('checkpoint')
    export_parser.add_argument('--out', default=MODULE_PATH)
    export_parser.set_defaults(func=command_export)
    evaluate_parser = commands.add_parser('evaluate', help='against the hand-labeled quotations')
    evaluate_parser.add_argument('model', nargs='?', default=MODULE_PATH)
    evaluate_parser.add_argument('--test', default=TEST_PATH)
    evaluate_parser.set_defaults(func=command_evaluate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
