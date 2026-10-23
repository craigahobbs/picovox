# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The dictionary audit - the CMU Pronouncing Dictionary's label errors the G2P network would learn (HOBBES HH AA1 B IY0
Z - "hobbies" - beside the dictionary's own HOBBESIAN HH AA1 B Z IY0 AH0 N), found by cross-validated disagreement and
adjudicated with the LibriTTS-R readers' audio; the confirmed corrections are data/nn/g2p-fixes.json, which g2p.py's
data applies in every stage (python is PYTHONPATH=train/src build/venv-nn/bin/python -B, listeners the same in
build/venv-listeners):

    python -m picovox_train.g2paudit folds --out DIR [-j 5]          # the screening models (resumable)
    python -m picovox_train.g2paudit suspects --out DIR --books BOOKS [--catalog CATALOG]
    listeners -m picovox_train.g2paudit audio --out DIR [-j 4]       # the readers' audio votes (resumable)
    python -m picovox_train.g2paudit adjudicate --out DIR            # verdicts.json, fixes.json (g2p-fixes.json's format)

1. folds - REPEATS independent partitions of the dictionary's words into FOLDS folds (a word's possessive forms in
   its fold), and a screening model per fold trained without the fold's words (SCREEN - the recipe's network and
   sampling, one shorter stage, no rule- or teacher-labeled words): every word is held out by REPEATS models
2. suspects - each held-out word's prediction by its fold models and the label's and the prediction's sequence log
   probabilities: a suspect when every fold model's prediction differs from the label in its phones (not stress
   alone) with a log-probability margin of at least MARGIN, ranked by its count in the books (g2p-words.json
   evalCounts + weightCounts), with its proper-noun share (capitalized mid-sentence in the books) and whether a name
   PicoShelf speaks (its catalog's authors and titles) contains it
3. audio - for each suspect the LibriTTS-R readers spoke (the MFA word alignments, up to PER_WORD tokens, speakers
   first), the CTC phoneme recognizer's (cremad.py) likelihood of the label and of the alternative on the word with
   its neighbors (their aligned phones) - a vote when one is VOTE_MARGIN more likely; and the same for control words
   (the models agree with the label) against their models' second guess, the method's error rate
4. adjudicate - a confirmed error when the alternative's votes are a significant majority (the Wilson 95% lower bound
   above one half, from 3+ speakers), confirmed correct when the label's are, else no evidence
"""

import argparse
import collections
import datetime
import json
import math
import multiprocessing
import os
import random
import re
import sys

from . import g2p
from .g2p import log_line
from .frontend import ROOT


# The audit's partitions - REPEATS partitions of FOLDS folds (every word held out REPEATS times)
FOLDS = 5
REPEATS = 2
PARTITION_SEED = 20261017

# The screening models' training - the recipe's network and sampling (g2p.py), one stage, no unknown words
SCREEN = {'steps': 6000, 'lr': 0.003, 'warmup': 300, 'unknown': 0.0, 'threads': 1}

# A suspect - every fold model's phones differ from the label's (ignoring stress) by at least this log-probability
MARGIN = 3.0


# The controls - words every fold model agrees with, in the books at least CONTROL_COUNT times, with one pronunciation
CONTROL_COUNT = 20
CONTROL_CANDIDATES = 400


# A phone string without stress
def unstressed(phones):
    return re.sub(r'\d', '', phones)


# The word a fold groups by - possessive forms with their word
def group_key(word):
    for suffix in ("'s", "s'", "'"):
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[:-len(suffix)] + ('s' if suffix == "s'" else '')
    return word


# The folds of a partition - {word: fold}
def partition(words, repeat, folds=FOLDS):
    groups = sorted({group_key(word) for word in words})
    random.Random(PARTITION_SEED + repeat).shuffle(groups)
    fold_of = {group: index % folds for index, group in enumerate(groups)}
    return {word: fold_of[group_key(word)] for word in words}


#
# 1. The screening models
#

# One screening model (a worker task) - (out directory, repeat, fold, steps): its checkpoint, and its measures (the
# held-out words' error - g2p.py's monitorWer over the fold) in fold-<repeat>-<fold>.json
def train_fold(task):
    out, repeat, fold, steps = task
    name = f'fold-{repeat}-{fold}'
    path = os.path.join(out, f'{name}.pt')
    if os.path.exists(path + '.json'):
        return path
    config = g2p.load_config()
    data = g2p.load_data(fixes=None, books=None)
    folds = partition(sorted(data['first']), repeat)
    data['monitor'] = sorted(word for word, index in folds.items() if index == fold)
    settings = {key: value for key, value in config.items() if key not in ('stages', 'export')}
    settings.update({key: value for key, value in SCREEN.items() if key in ('unknown', 'threads')})
    stage = {'name': name, 'steps': steps or SCREEN['steps'], 'lr': SCREEN['lr'], 'warmup': SCREEN['warmup'],
             'seed': 100 + 10 * repeat + fold}
    result = g2p.train_stage(settings, stage, None, path, data, lambda text: log_line(os.path.join(out, 'folds.log'), text))
    with open(path + '.json', 'w', encoding='utf-8') as file:
        json.dump({'heldOut': len(data['monitor']), 'heldOutWer': result['monitorWer'], 'book': result['book'],
                   'top10kWrong': result['top10kWrong']}, file, indent=1)
    return path


def command_folds(args):
    """Train the screening models - <out>/fold-<repeat>-<fold>.pt (resumable), jobs at once"""
    os.makedirs(args.out, exist_ok=True)
    repeats = range(args.repeats) if args.repeat is None else [args.repeat]
    tasks = [(args.out, repeat, fold, args.steps) for repeat in repeats for fold in range(FOLDS)]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            for path in pool.imap_unordered(train_fold, tasks):
                log_line(os.path.join(args.out, 'folds.log'), f'done: {path}')
    else:
        for task in tasks:
            train_fold(task)
    return 0


#
# 2. The suspects
#

# A model's held-out words - {word: [prediction, label log probability, prediction log probability, [the dictionary's
# other variants' log probabilities]]} (every: word to its pronunciations, the label first)
def fold_scores(model, words, every):
    predictions = g2p.predict(model, words)
    label_scores = sequence_log_probs(model, words, [every[word][0] for word in words])
    prediction_scores = sequence_log_probs(model, words, predictions)
    others = [(word, phones) for word in words for phones in every[word][1:]]
    other_scores = sequence_log_probs(model, [word for word, _ in others], [phones for _, phones in others]) if others else []
    variant_scores = collections.defaultdict(list)
    for (word, _), score in zip(others, other_scores):
        variant_scores[word].append(round(score, 3))
    return {word: [' '.join(prediction), round(label_score, 3), round(prediction_score, 3), variant_scores.get(word, [])]
            for word, prediction, label_score, prediction_score in zip(words, predictions, label_scores, prediction_scores)}


# The log probabilities of words' phone sequences (each with its <eos>) under a model, teacher forced - a list of floats
def sequence_log_probs(model, words, phone_lists, batch_size=512):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            letters, targets = g2p.make_batch(words[start:start + batch_size], phone_lists[start:start + batch_size])
            log_probs = step_log_probs(model, letters, targets).gather(2, targets.unsqueeze(2)).squeeze(2)
            results.extend((log_probs * (targets != 0)).sum(1).tolist())
    return results


# The log softmax of each decoder step, teacher forced on targets [batch, steps] - [batch, steps, phones]
def step_log_probs(model, letters, targets):
    import torch  # pylint: disable=import-outside-toplevel
    encoded, mask, state = model.encode(letters)
    previous = torch.cat([torch.full_like(targets[:, :1], g2p.EOS), targets[:, :-1]], 1)
    decoded, _ = model.decoder(model.phone_embed(previous), state)
    return torch.log_softmax(model.attend(decoded, encoded, mask), -1)


# A model's second guess at a word it pronounces as phones - the least certain step's runner-up phone, then greedy
# decoding from there (None if no step has a runner-up phone)
def second_guess(model, word, phones):
    import torch  # pylint: disable=import-outside-toplevel
    model.eval()
    with torch.no_grad():
        letters, targets = g2p.make_batch([word], [phones])
        log_probs = step_log_probs(model, letters, targets)[0]
        best = None
        for step, target in enumerate(targets[0].tolist()):
            ranked = [index for index in log_probs[step].argsort(descending=True).tolist() if index not in (target, g2p.BLANK, g2p.EOS)]
            margin = float(log_probs[step, target] - log_probs[step, ranked[0]])
            if best is None or margin < best[0]:
                best = (margin, step, ranked[0])
        _, step, phone = best
        return continue_greedy(model, letters, [g2p.PHONE_IDS[item] for item in phones[:step]] + [phone])


# Greedy decoding of one word's letters [1, time] after a forced prefix of phone ids - the phones
def continue_greedy(model, letters, prefix):
    import torch  # pylint: disable=import-outside-toplevel
    encoded, mask, state = model.encode(letters)
    previous = torch.tensor([[g2p.EOS] + prefix[:-1]], dtype=torch.long)
    _, state = model.decoder(model.phone_embed(previous), state)
    result = list(prefix)
    current = prefix[-1]
    limit = 3 * int(mask.sum()) + 6
    while len(result) < limit:
        decoded, state = model.decoder(model.phone_embed(torch.tensor([[current]], dtype=torch.long)), state)
        current = int(model.attend(decoded, encoded, mask)[0, 0].argmax())
        if current in (g2p.BLANK, g2p.EOS):
            break
        result.append(current)
    return [g2p.PHONES[index] for index in result]


# The book counts of words' capitalizations mid-sentence - {word: [capitalized, lowercase]} (a word after the start of a
# paragraph, or after anything but a space or a comma since the previous word, isn't mid-sentence)
def book_capitals(books_dir, words):
    counts = {word: [0, 0] for word in words}
    pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
    for name in sorted(os.listdir(books_dir)):
        if not name.endswith('.txt'):
            continue
        with open(os.path.join(books_dir, name), encoding='utf-8', errors='replace') as file:
            paragraphs = re.split(r'\n\s*\n', file.read())
        for paragraph in paragraphs:
            paragraph = ' '.join(paragraph.split())
            end = None
            for match in pattern.finditer(paragraph):
                token = match.group(0)
                count = counts.get(token.lower())
                if count is not None and end is not None and re.fullmatch(r'[ ,]*', paragraph[end:match.start()]):
                    if token[0].isupper() and not token.isupper():
                        count[0] += 1
                    elif token.islower():
                        count[1] += 1
                end = match.end()
    return counts


# The words of the names PicoShelf speaks - its catalog's authors and titles, lowercase, without accents
def catalog_words(path):
    import unicodedata  # pylint: disable=import-outside-toplevel
    with open(path, encoding='utf-8') as file:
        books = json.load(file)['books']
    words = set()
    for book in books:
        for text in (book.get('author') or '', book.get('title') or ''):
            plain = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').lower()
            words.update(re.findall(r"[a-z]+(?:'[a-z]+)*", plain))
    return words


# Load a screening model checkpoint
def load_model(path):
    import torch  # pylint: disable=import-outside-toplevel
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = g2p.build_model(checkpoint['config'])
    model.load_state_dict(checkpoint['state'])
    return model


# A word's suspect entry from its fold models' scores [[prediction, label score, prediction score, variant scores]]
# and its pronunciations (the label first) - (flagged, entry): flagged when every model's prediction differs from the
# label (not by stress alone) by MARGIN, or every model prefers one of the other variants (that differs from the label
# not by stress alone) by MARGIN
def judge(variants, scores):
    label = variants[0]
    disagree = all(unstressed(prediction) != unstressed(label) and prediction_score - label_score >= MARGIN
                   for prediction, label_score, prediction_score, _ in scores)
    preferred = [index for index in range(1, len(variants)) if unstressed(variants[index]) != unstressed(label) and
                 all(variant_scores[index - 1] - label_score >= MARGIN for _, label_score, _, variant_scores in scores)]
    agree = all(unstressed(prediction) == unstressed(label) for prediction, _, _, _ in scores)
    flags = (['prediction'] if disagree else []) + (['variant'] if preferred else [])
    entry = {'label': label, 'models': [score[:3] for score in scores], 'agree': agree, 'flags': flags,
             'preferred': [variants[index] for index in preferred]}
    return bool(flags), entry


def command_suspects(args):
    """suspects.json - every word all its fold models confidently disagree with, ranked by its count in the books, with
    its proper-noun share and catalog names; and controls.json - words every fold model agrees with, and a model's second
    guess"""
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(args.threads)
    data = g2p.load_data(fixes=None, books=None)
    first, every = data['first'], data['all']
    labels = {word: ' '.join(phones) for word, phones in first.items()}
    scores = collections.defaultdict(list)
    os.makedirs(os.path.join(args.out, 'scores'), exist_ok=True)
    for repeat in range(args.repeats):
        folds = partition(sorted(first), repeat)
        for fold in range(FOLDS):
            path = os.path.join(args.out, 'scores', f'fold-{repeat}-{fold}.json')
            if not os.path.exists(path):
                words = sorted(word for word, index in folds.items() if index == fold)
                result = fold_scores(load_model(os.path.join(args.out, f'fold-{repeat}-{fold}.pt')), words, every)
                with open(path, 'w', encoding='utf-8') as file:
                    json.dump(result, file)
                log_line(args.log, f'suspects: scored fold-{repeat}-{fold} ({len(words)} words)')
            with open(path, encoding='utf-8') as file:
                for word, score in json.load(file).items():
                    scores[word].append(score)
    counts = collections.Counter(data['weightCounts'])
    counts.update(data['evalCounts'])
    relabel, context_words = set(), set()
    if args.relabel:
        with open(args.relabel, encoding='utf-8') as file:
            relabel = set(json.load(file)['relabel'])
    if args.heteronyms:
        with open(args.heteronyms, encoding='utf-8') as file:
            context_words = set(json.load(file)['words'])
    suspects, agreed = [], []
    for word in sorted(scores):
        flagged, entry = judge([' '.join(phones) for phones in every[word]], scores[word])
        if flagged:
            entry.update({'word': word, 'count': counts.get(word, 0), 'variants': [' '.join(phones) for phones in every[word]],
                          'relabeled': word in relabel, 'heteronym': word in context_words})
            suspects.append(entry)
        elif entry['agree'] and counts.get(word, 0) >= CONTROL_COUNT and len(word) >= 4 and len(every[word]) == 1:
            agreed.append(word)
    capitals = book_capitals(args.books, [entry['word'] for entry in suspects]) if args.books else {}
    names = catalog_words(args.catalog) if args.catalog else set()
    for entry in suspects:
        capitalized, lowercase = capitals.get(entry['word'], [0, 0])
        entry.update({'capitalized': capitalized, 'lowercase': lowercase,
                      'proper': round(capitalized / (capitalized + lowercase), 3) if capitalized + lowercase else None,
                      'catalog': entry['word'] in names})
    suspects.sort(key=lambda entry: (-entry['count'], entry['word']))
    with open(os.path.join(args.out, 'suspects.json'), 'w', encoding='utf-8') as file:
        json.dump(suspects, file, indent=1)

    # The controls - a model's second guess at words every model agrees with
    rng = random.Random(PARTITION_SEED)
    rng.shuffle(agreed)
    model = load_model(os.path.join(args.out, 'fold-0-0.pt'))
    controls = []
    for word in agreed[:CONTROL_CANDIDATES]:
        guess = second_guess(model, word, first[word])
        if unstressed(' '.join(guess)) != unstressed(labels[word]):
            controls.append({'word': word, 'label': labels[word], 'alternative': ' '.join(guess), 'count': counts.get(word, 0)})
    with open(os.path.join(args.out, 'controls.json'), 'w', encoding='utf-8') as file:
        json.dump(controls, file, indent=1)
    held_out = held_out_accuracy(scores, labels, every)
    log_line(args.log, f'suspects: {len(suspects)} of {len(scores)} words ({sum(entry["count"] > 0 for entry in suspects)} in '
                       f'the books, {sum(entry["catalog"] for entry in suspects)} in the catalog\'s names); held-out accuracy '
                       f'{100 * held_out:.2f}% (any pronunciation, stress included); {len(controls)} controls')
    return 0


# The screening models' held-out accuracy - the share of their predictions that are one of the word's pronunciations
def held_out_accuracy(scores, labels, every):
    right = total = 0
    for word, items in scores.items():
        variants = {' '.join(phones) for phones in every.get(word, [labels[word].split()])}
        for prediction, *_ in items:
            right += prediction in variants
            total += 1
    return right / max(1, total)


#
# 3. The readers' audio
#

# The recognizer's IPA tokens each phone may be heard as (cremad.py's IPA_TOKENS, without the tokens of another vowel's
# quality - the candidates differ there); an unstressed vowel may be reduced (REDUCTIONS)
STRICT_TOKENS = {
    'AA': ['ɑː', 'ɑ', 'ɑːɹ', 'a', 'aː', 'ɒ'],
    'AE': ['æ', 'a', 'æː', 'eə'],
    'AH': ['ʌ', 'ə', 'ɐ'],
    'AO': ['ɔː', 'ɔ', 'ɒ', 'ɔːɹ', 'oːɹ', 'ɑː', 'ɑ'],
    'AW': ['aʊ', 'au'],
    'AY': ['aɪ', 'aɪɚ', 'aɪə', 'ai'],
    'EH': ['ɛ', 'e', 'ɛɹ', 'eː', 'eə'],
    'ER': ['ɜː', 'ɚ', 'ɜ'],
    'EY': ['eɪ', 'e', 'eː', 'ɛɪ'],
    'IH': ['ɪ', 'ᵻ', 'ɪɹ'],
    'IY': ['iː', 'i', 'iə', 'ɪɹ'],
    'OW': ['oʊ', 'o', 'oː', 'əʊ', 'oːɹ'],
    'OY': ['ɔɪ', 'oɪ'],
    'UH': ['ʊ', 'ʊɹ'],
    'UW': ['uː', 'u', 'ʉ'],
    'B': ['b', 'β'],
    'CH': ['tʃ'],
    'D': ['d', 'ɾ'],
    'DH': ['ð'],
    'F': ['f'],
    'G': ['ɡ'],
    'HH': ['h'],
    'JH': ['dʒ'],
    'K': ['k', 'kh', 'kʰ'],
    'L': ['l', 'ɫ', 'əl', 'l̩'],
    'M': ['m'],
    'N': ['n', 'n̩'],
    'NG': ['ŋ'],
    'P': ['p', 'ph', 'pʰ'],
    'R': ['ɹ', 'r', 'ɚ', 'ɑːɹ', 'ɔːɹ', 'oːɹ', 'ɛɹ', 'ɪɹ', 'ʊɹ'],
    'S': ['s', 'z'],
    'SH': ['ʃ'],
    'T': ['t', 'ɾ', 'ʔ', 'th', 'tʰ'],
    'TH': ['θ'],
    'V': ['v'],
    'W': ['w'],
    'Y': ['j'],
    'Z': ['z', 's'],
    'ZH': ['ʒ']
}
REDUCTIONS = ['ə', 'ɐ', 'ɪ', 'ᵻ']

# The tokens - of the words MIN_SPEAKERS spoke (a verdict needs them), at most PER_WORD a word (speakers first) and
# CONTROL_TOKENS a control word, CONTROL_WORDS controls; a
# neighboring word is the context when it's adjacent (within ADJACENT seconds) and at most NEIGHBOR_SECONDS long, else
# PAD seconds of the audio (silence)
PER_WORD = 12
CONTROL_TOKENS = 6
CONTROL_WORDS = 150
ADJACENT = 0.03
NEIGHBOR_SECONDS = 1.0
PAD = 0.12

# A token's vote - the candidate at least VOTE_MARGIN more likely (log) than every other; a verdict - its votes from
# MIN_SPEAKERS speakers or more (as heteronyms.py's)
VOTE_MARGIN = 2.0
MIN_SPEAKERS = 3


# The recognizer token ids of an aligned phone (stress digit and all)
def phone_token_ids(phone, vocab):
    tokens = STRICT_TOKENS[unstressed(phone)] + (REDUCTIONS if phone[-1:] == '0' else [])
    return [vocab[token] for token in dict.fromkeys(tokens) if token in vocab]


# The CTC log likelihood of a phone sequence (each phone a class of recognizer tokens, never merged with its neighbor)
# given a segment's log posteriors [frames, vocab]
def ctc_log_likelihood(log_posteriors, phones, vocab):
    import numpy  # pylint: disable=import-outside-toplevel
    frames = len(log_posteriors)
    count = len(phones)
    if frames < count:
        return -math.inf
    blank = log_posteriors[:, vocab['<pad>']]
    emit = numpy.empty((frames, 2 * count + 1))
    emit[:, 0::2] = blank[:, None]
    for index, phone in enumerate(phones):
        emit[:, 2 * index + 1] = numpy.logaddexp.reduce(log_posteriors[:, phone_token_ids(phone, vocab)], axis=1)
    score = numpy.full(2 * count + 1, -numpy.inf)
    score[0], score[1] = emit[0, 0], emit[0, 1]
    for frame in range(1, frames):
        step = numpy.concatenate(([-numpy.inf], score[:-1]))
        skip = numpy.concatenate(([-numpy.inf, -numpy.inf], score[:-2]))
        skip[0::2] = -numpy.inf
        score = numpy.logaddexp(numpy.logaddexp(score, step), skip) + emit[frame]
    return float(numpy.logaddexp(score[-1], score[-2]))


# A candidate's log likelihood in its context - the better of its phones and (when it has a vowel and R the recognizer
# may hear as one r-colored token, which a run of one frame can't split) the phones with those Rs dropped
def candidate_log_likelihood(log_posteriors, before, phones, after, vocab):
    sequence = before + phones + after
    merged_r = [phone for index, phone in enumerate(sequence)
                if not (phone == 'R' and index and unstressed(sequence[index - 1]) in R_COLORED)]
    score = ctc_log_likelihood(log_posteriors, sequence, vocab)
    return max(score, ctc_log_likelihood(log_posteriors, merged_r, vocab)) if merged_r != sequence else score


# The vowels with an r-colored token (STRICT_TOKENS) - ɑːɹ, ɔːɹ, oːɹ, ɛɹ, ɪɹ, ʊɹ
R_COLORED = {'AA', 'AO', 'OW', 'EH', 'IH', 'IY', 'UH'}


# A token's vote among candidate scores - the best's index when it's VOTE_MARGIN above the rest, else None
def vote(scores):
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    return order[0] if scores[order[0]] - scores[order[1]] >= VOTE_MARGIN else None


# The aligned occurrences of words in the LibriTTS-R labels - {word: [{rel, start, end, before, after}]}, before and after
# the neighboring words' (start, end, word) or None
def find_tokens(words, labels_dir):
    from . import heteronyms  # pylint: disable=import-outside-toplevel
    found = collections.defaultdict(list)
    for subset in heteronyms.SUBSETS:
        root = os.path.join(labels_dir, 'word', subset)
        for dirpath, _, names in sorted(os.walk(root)):
            for name in sorted(names):
                if not name.endswith('.lab'):
                    continue
                rows = heteronyms.read_lab(os.path.join(dirpath, name))
                for index, (start, end, word) in enumerate(rows):
                    if word in words:
                        found[word].append({'rel': os.path.relpath(os.path.join(dirpath, name), os.path.join(labels_dir, 'word')),
                                            'start': start, 'end': end, 'before': rows[index - 1] if index else None,
                                            'after': rows[index + 1] if index + 1 < len(rows) else None})
    return found


# Up to count of a word's tokens - one a speaker at a time (seeded)
def choose_tokens(tokens, count, rng):
    by_speaker = collections.defaultdict(list)
    for token in tokens:
        by_speaker[token['rel'].split(os.sep)[1]].append(token)
    queues = [by_speaker[speaker] for speaker in sorted(by_speaker)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    chosen = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop())
    return chosen


# A token's segment and context - (start, end, phones before, phones after) from the utterance's aligned phones
def token_context(token, phones):
    def inside(span):
        return [phone for start, end, phone in phones if start >= span[0] - 1e-4 and end <= span[1] + 1e-4]
    start, end, before, after = token['start'], token['end'], [], []
    neighbor = token['before']
    if neighbor and token['start'] - neighbor[1] <= ADJACENT and neighbor[1] - neighbor[0] <= NEIGHBOR_SECONDS and \
            'spn' not in inside(neighbor):
        start, before = neighbor[0], inside(neighbor)
    else:
        start = max(neighbor[1] if neighbor else 0.0, token['start'] - PAD)
    neighbor = token['after']
    if neighbor and neighbor[0] - token['end'] <= ADJACENT and neighbor[1] - neighbor[0] <= NEIGHBOR_SECONDS and \
            'spn' not in inside(neighbor):
        end, after = neighbor[1], inside(neighbor)
    else:
        end = min(neighbor[0] if neighbor else token['end'] + PAD, token['end'] + PAD)
    return start, end, [phone for phone in before if phone not in ('sil', 'sp')], [phone for phone in after if phone not in ('sil', 'sp')]


# A segment of an utterance's audio at 16 kHz
def read_segment(path, start, end):
    import soundfile  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    from .cremad import CTC_SAMPLE_RATE  # pylint: disable=import-outside-toplevel
    info = soundfile.info(path)
    samples, rate = soundfile.read(path, start=int(start * info.samplerate), stop=int(end * info.samplerate), dtype='float64',
                                   always_2d=True)
    return soxr.resample(samples.mean(axis=1), rate, CTC_SAMPLE_RATE, quality='VHQ')


# Why an alternative can't correct a label (None if it can) - as heteronyms.py selects its relabeled words: not a
# function word (its reduced forms are contextual) or a word of one or two letters (letters, abbreviations, Roman
# numerals), and a lexical difference (not stress or vowel reduction alone, and not a consonant dropped or added - the
# recognizer hardly hears an unreleased or assimilated consonant: EY1 T TH); and
# not a merger's difference alone (AA and AO; AE, EH, and EY before R - the readers' dialects differ, and the recognizer
# learned espeak's forms)
def unfixable(word, label, alternative):
    from . import heteronyms  # pylint: disable=import-outside-toplevel
    if word in heteronyms.FUNCTION_WORDS:
        return 'function word'
    if len(word) <= 2:
        return 'a letter or an abbreviation'
    difference = heteronyms.kind(label.split(), alternative.split())
    if difference in ('stress', 'reduced', 'consonant-drop') or \
            (difference == 'vowel' and not heteronyms.lexical(label, alternative)):
        return f'{difference if difference != "vowel" else "reduction"} only'
    if merged(label) == merged(alternative):
        return 'merger only'
    return None


# A pronunciation with the mergers' vowels merged, the reduced vowels (AH0, IH0) one schwa, and without stress
def merged(phones):
    items = phones.split()
    bases = unstressed(phones).split()
    return ' '.join('A*' if phone in ('AA', 'AO') else
                    ('E*' if phone in ('AE', 'EH', 'EY') and index + 1 < len(bases) and bases[index + 1] == 'R' else
                     ('@' if items[index] in ('AH0', 'IH0') else phone))
                    for index, phone in enumerate(bases))


# The audio plan - [{word, control, candidates, token}] for the suspects (the label and the alternatives that could
# correct it - the fold models' predictions and the dictionary's variants) and controls the readers spoke
def audio_plan(suspects, controls, found):
    rng = random.Random(PARTITION_SEED)
    plan = []
    def speakers(tokens):
        return len({token['rel'].split(os.sep)[1] for token in tokens})
    for entry in suspects:
        candidates = [entry['label']]
        for phones in [prediction for prediction, _, _ in entry['models']] + entry['variants']:
            if all(unstressed(phones) != unstressed(candidate) for candidate in candidates) and \
                    unfixable(entry['word'], entry['label'], phones) is None:
                candidates.append(phones)
        tokens = found.get(entry['word'], [])
        if len(candidates) > 1 and speakers(tokens) >= MIN_SPEAKERS:
            for token in choose_tokens(tokens, PER_WORD, rng):
                plan.append({'word': entry['word'], 'control': False, 'candidates': candidates, 'token': token})
    used = 0
    for entry in controls:
        tokens = found.get(entry['word'], [])
        if speakers(tokens) >= MIN_SPEAKERS and used < CONTROL_WORDS and \
                unfixable(entry['word'], entry['label'], entry['alternative']) is None:
            used += 1
            for token in choose_tokens(tokens, CONTROL_TOKENS, rng):
                plan.append({'word': entry['word'], 'control': True, 'candidates': [entry['label'], entry['alternative']],
                             'token': token})
    return plan


def command_audio(args):
    """audio.jsonl - each planned token's candidate log likelihoods and vote (resumable: tokens scored before are kept)"""
    import torch  # pylint: disable=import-outside-toplevel
    from . import cremad, heteronyms  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(args.threads)
    with open(os.path.join(args.out, 'suspects.json'), encoding='utf-8') as file:
        suspects = [entry for entry in json.load(file) if not entry['relabeled'] and not entry['heteronym']]
    with open(os.path.join(args.out, 'controls.json'), encoding='utf-8') as file:
        controls = json.load(file)
    found = find_tokens({entry['word'] for entry in suspects} | {entry['word'] for entry in controls}, heteronyms.LABELS)
    plan = audio_plan(suspects, controls, found)
    path = os.path.join(args.out, 'audio.jsonl')
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            done = {(item['word'], item['rel'], item['start']) for item in map(json.loads, file)}
    plan = [item for item in plan if (item['word'], item['token']['rel'], item['token']['start']) not in done]
    log_line(args.log, f'audio: {len(plan)} tokens to score ({len(done)} before), '
                       f'{len({item["word"] for item in plan if not item["control"]})} suspects')
    model, vocab, device = cremad.load_ctc('cpu')
    with open(path, 'a', encoding='utf-8') as file:
        for offset in range(0, len(plan), args.batch):
            group = plan[offset:offset + args.batch]
            segments = []
            for item in group:
                token = item['token']
                subset, speaker, chapter, name = token['rel'].split(os.sep)
                phones = heteronyms.read_lab(os.path.join(heteronyms.LABELS, 'phone', token['rel']))
                start, end, before, after = token_context(token, phones)
                wav = os.path.join(heteronyms.AUDIO, subset, speaker, chapter, name[:-4] + '.wav')
                segments.append((before, after, read_segment(wav, start, end)))
            posteriors = cremad.ctc_log_posteriors(model, device, [samples for _, _, samples in segments])
            for item, (before, after, _), log_posteriors in zip(group, segments, posteriors):
                scores = [candidate_log_likelihood(log_posteriors, before, candidate.split(), after, vocab)
                          for candidate in item['candidates']]
                file.write(json.dumps({'word': item['word'], 'control': item['control'], 'rel': item['token']['rel'],
                                       'start': item['token']['start'], 'candidates': item['candidates'],
                                       'scores': [round(score, 3) for score in scores], 'vote': vote(scores)}) + '\n')
            file.flush()
            log_line(args.log, f'audio: {offset + len(group)}/{len(plan)} tokens')
    return 0


#
# 4. The verdicts
#

# A word's verdict from its tokens' votes - {verdict, votes (per candidate), speakers (per candidate), best (the
# alternative with the most votes), lower (the Wilson lower bound of the winner's share against the label or the best)}
def verdict(candidates, votes):
    from .heteronyms import wilson_lower  # pylint: disable=import-outside-toplevel
    counts = [0] * len(candidates)
    speakers = [set() for _ in candidates]
    for index, speaker in votes:
        if index is not None:
            counts[index] += 1
            speakers[index].add(speaker)
    best = max(range(1, len(candidates)), key=lambda index: (counts[index], -index))
    label_votes, best_votes = counts[0], counts[best]
    result = {'votes': counts, 'speakers': [len(names) for names in speakers], 'best': best, 'tokens': len(votes)}
    if wilson_lower(best_votes, best_votes + label_votes) > 0.5 and len(speakers[best]) >= MIN_SPEAKERS:
        result.update(verdict='error', lower=round(wilson_lower(best_votes, best_votes + label_votes), 3))
    elif wilson_lower(label_votes, best_votes + label_votes) > 0.5 and len(speakers[0]) >= MIN_SPEAKERS:
        result.update(verdict='correct', lower=round(wilson_lower(label_votes, best_votes + label_votes), 3))
    else:
        result.update(verdict='inconclusive' if votes else 'none', lower=None)
    return result


# A word's scored tokens' candidates (every token's, in order - the label first) and votes [(candidate index or None,
# speaker)] - the tokens may have been scored with different candidates
def token_votes(items):
    candidates = list(dict.fromkeys(candidate for item in items for candidate in item['candidates']))
    votes = [(None if item['vote'] is None else candidates.index(item['candidates'][item['vote']]), item['rel'].split(os.sep)[1])
             for item in items]
    return candidates, votes


# The corrected pronunciation - the dictionary's variant of the winner (its stress), else the fold models' highest scored
# prediction of it
def corrected(entry, winner):
    for variant in entry['variants']:
        if unstressed(variant) == unstressed(winner):
            return variant
    predictions = [(prediction_score, prediction) for prediction, _, prediction_score in entry['models']
                   if unstressed(prediction) == unstressed(winner)]
    return max(predictions)[1]


def command_adjudicate(args):
    """verdicts.json (every suspect's verdict, and the controls' error rate) and fixes.json (data/nn/g2p-fixes.json's
    format - the confirmed errors' corrections and their evidence)"""
    with open(os.path.join(args.out, 'suspects.json'), encoding='utf-8') as file:
        suspects = json.load(file)
    by_word = collections.defaultdict(list)
    with open(os.path.join(args.out, 'audio.jsonl'), encoding='utf-8') as file:
        for item in map(json.loads, file):
            by_word[(item['word'], item['control'])].append(item)
    verdicts = []
    for entry in suspects:
        items = by_word.get((entry['word'], False), [])
        candidates, votes = token_votes(items) if items else ([entry['label'], entry['models'][0][0]], [])
        result = verdict(candidates, votes)
        result.update(word=entry['word'], label=entry['label'], alternative=candidates[result['best']], count=entry['count'],
                      proper=entry['proper'], catalog=entry['catalog'], relabeled=entry['relabeled'], heteronym=entry['heteronym'],
                      candidates=candidates)
        reason = unfixable(entry['word'], entry['label'], candidates[result['best']])
        if result['verdict'] == 'error' and reason:
            result.update(verdict='excluded', reason=reason)
        elif result['verdict'] == 'error':
            result['fix'] = corrected(entry, candidates[result['best']])
        verdicts.append(result)

    # The controls - the method's false confirmations (and its tokens' share of votes for the alternative)
    control_words = sorted({word for word, control in by_word if control})
    control_results = [verdict(*token_votes(by_word[(word, True)])) for word in control_words]
    control_votes = [item['vote'] for word in control_words for item in by_word[(word, True)] if item['vote'] is not None]
    controls = {'words': len(control_words), 'falseErrors': sum(result['verdict'] == 'error' for result in control_results),
                'confirmedCorrect': sum(result['verdict'] == 'correct' for result in control_results),
                'tokenVotes': len(control_votes), 'tokenVotesForAlternative': sum(1 for index in control_votes if index != 0),
                'errors': [word for word, result in zip(control_words, control_results) if result['verdict'] == 'error']}
    summary = collections.Counter(result['verdict'] for result in verdicts)
    with open(os.path.join(args.out, 'verdicts.json'), 'w', encoding='utf-8') as file:
        json.dump({'summary': dict(summary), 'controls': controls, 'verdicts': verdicts}, file, indent=1)

    # The fixes - the confirmed errors, and the extra fixes (--extra: {word: {phones, evidence}} - corrections on the
    # dictionary's own evidence, or a confirmed error's correction from the recognizer's own phones when neither
    # candidate is what the readers say - added to its evidence)
    fixes, evidence = {}, {}
    today = datetime.date.today().isoformat()
    for result in verdicts:
        if result['verdict'] == 'error' and not result['relabeled'] and not result['heteronym']:
            entry = next(item for item in suspects if item['word'] == result['word'])
            fixes[result['word']] = result['fix']
            evidence[result['word']] = {
                'was': result['label'], 'date': today,
                'folds': [[prediction, round(prediction_score - label_score, 2)]
                          for prediction, label_score, prediction_score in entry['models']],
                'audio': {'candidates': result['candidates'], 'votes': result['votes'], 'speakers': result['speakers'],
                          'tokens': result['tokens'], 'lower': result['lower']}
            }
    if args.extra:
        with open(args.extra, encoding='utf-8') as file:
            for word, item in json.load(file).items():
                fixes[word] = item['phones']
                evidence[word] = {**evidence.get(word, {}), **item['evidence'], 'date': item['evidence'].get('date', today)}
    with open(os.path.join(args.out, 'fixes.json'), 'w', encoding='utf-8') as file:
        json.dump({'description': FIXES_DESCRIPTION, 'fixes': dict(sorted(fixes.items())), 'evidence': dict(sorted(evidence.items()))},
                  file, indent=1, ensure_ascii=False)
    log_line(args.log, f'adjudicate: {dict(summary)}; controls {controls["words"]} words, {controls["falseErrors"]} false errors, '
                       f'{controls["tokenVotesForAlternative"]}/{controls["tokenVotes"]} token votes for the alternative; '
                       f'{len(fixes)} fixes')
    return 0


FIXES_DESCRIPTION = (
    'The CMU Pronouncing Dictionary\'s corrected labels (picovox_train/g2paudit.py; g2p.py applies them in every stage, before '
    'g2p-relabel.json, and the rule-labeled words derived from them follow) - words whose first pronunciation the '
    'cross-validated screening models confidently disagree with (fold models trained without the word: every model\'s '
    'phones differ, by a log-probability margin) and whose LibriTTS-R readers\' audio confirms the alternative (the CTC '
    'phoneme recognizer\'s likelihood of each candidate on the aligned word with its neighbors: the votes\' Wilson 95% '
    'lower bound above one half, from 3+ speakers) - or, where noted, the dictionary\'s own evidence (its derived forms). '
    'evidence: was (the dictionary\'s first), folds ([prediction, margin] per fold model), audio (candidates - the label '
    'first - votes and speakers per candidate, tokens, lower), date'
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.g2paudit', description='The dictionary audit')
    commands = parser.add_subparsers(dest='command', required=True)
    folds_parser = commands.add_parser('folds', help='the screening models (resumable)')
    folds_parser.add_argument('--out', required=True)
    folds_parser.add_argument('--repeats', type=int, default=REPEATS)
    folds_parser.add_argument('--repeat', type=int, help='only this partition')
    folds_parser.add_argument('--steps', type=int, help='each model this many steps (a smoke test)')
    folds_parser.add_argument('-j', '--jobs', type=int, default=1)
    folds_parser.set_defaults(func=command_folds)
    suspects_parser = commands.add_parser('suspects', help='the suspect labels and the controls')
    suspects_parser.add_argument('--out', required=True)
    suspects_parser.add_argument('--repeats', type=int, default=REPEATS)
    suspects_parser.add_argument('--books', help='a directory of the books\' texts (the proper-noun shares)')
    suspects_parser.add_argument('--catalog', help='PicoShelf\'s catalog.json (the names it speaks)')
    suspects_parser.add_argument('--relabel', default=os.path.join(ROOT, 'data', 'nn', 'g2p-relabel.json'))
    suspects_parser.add_argument('--heteronyms', help='the context labels (labels.json) - their words are the heteronym models\'')
    suspects_parser.add_argument('--threads', type=int, default=1)
    suspects_parser.add_argument('--log')
    suspects_parser.set_defaults(func=command_suspects)
    audio_parser = commands.add_parser('audio', help='the readers\' audio votes (resumable)')
    audio_parser.add_argument('--out', required=True)
    audio_parser.add_argument('--batch', type=int, default=32)
    audio_parser.add_argument('--threads', type=int, default=2)
    audio_parser.add_argument('--log')
    audio_parser.set_defaults(func=command_audio)
    adjudicate_parser = commands.add_parser('adjudicate', help='the verdicts and the fixes')
    adjudicate_parser.add_argument('--out', required=True)
    adjudicate_parser.add_argument('--extra', help='other fixes on the dictionary\'s own evidence - {word: {phones, evidence}}')
    adjudicate_parser.add_argument('--log')
    adjudicate_parser.set_defaults(func=command_adjudicate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
