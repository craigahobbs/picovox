# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The dictionary audit - the CMU Pronouncing Dictionary's label errors the G2P network would learn (HOBBES HH AA1 B IY0
Z - "hobbies" - beside the dictionary's own HOBBESIAN HH AA1 B Z IY0 AH0 N), found by cross-validated disagreement and
adjudicated with the training speech's audio (tts.py's corpus); the confirmed corrections are train/data/g2p-fixes.json, which g2p.py's
data applies in every stage (python is PYTHONPATH=train/src build/venv-train/bin/python -B):

    python -m picovox_train.g2paudit folds --out DIR [-j 5]          # the screening models (resumable)
    python -m picovox_train.g2paudit suspects --out DIR --books BOOKS [--catalog CATALOG]
    python -m picovox_train.g2paudit audio --out DIR [--speak]       # the training speech's votes (resumable)
    python -m picovox_train.g2paudit adjudicate --out DIR            # verdicts.json, fixes.json (g2p-fixes.json's format)

1. folds - REPEATS independent partitions of the dictionary's words into FOLDS folds (a word's possessive forms in
   its fold), and a screening model per fold trained without the fold's words (SCREEN - the recipe's network and
   sampling, one shorter stage, no rule- or teacher-labeled words): every word is held out by REPEATS models
2. suspects - each held-out word's prediction by its fold models and the label's and the prediction's sequence log
   probabilities: a suspect when every fold model's prediction differs from the label in its phones (not stress
   alone) with a log-probability margin of at least MARGIN, ranked by its count in the books (g2p-words.json
   evalCounts + weightCounts), with its proper-noun share (capitalized mid-sentence in the books) and whether a name
   PicoShelf speaks (its catalog's authors and titles) contains it
3. audio - for each suspect the corpus's voices spoke (its clips' CTC alignments, up to PER_WORD tokens, books first;
   with --speak, a word of fewer has the context labels' books' sentences with it spoken by the engine, in turn by
   the voices, and aligned - <corpus>/audit/), the CTC phoneme recognizer's (ctc.py) likelihood of the label and of
   the alternative on the word with its neighbors (their aligned phones) - a vote when one is VOTE_MARGIN more likely;
   and the same for control words (the models agree with the label) against their models' second guess, the method's
   error rate. The engine's pronunciation of a rare word or a name is its own G2P's guess: a verdict is the engine and
   the fold models together.
4. adjudicate - a confirmed error when the alternative's votes are a significant majority (the Wilson 95% lower bound
   above one half, from MIN_BOOKS books or more), confirmed correct when the label's are, else no evidence
"""

import argparse
import collections
import datetime
import json
import multiprocessing
import os
import random
import re
import sys

from . import RECIPE_PATH, ROOT, g2p, read_json
from .books import book_capitals
from .ctc import candidate_log_likelihood, merged, vote
from .g2p import log_line


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
    settings = {key: value for key, value in config.items() if key != 'stages'}
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
# 3. The training speech's audio
#

# The tokens - of the words MIN_BOOKS books spoke (a verdict needs them), at most PER_WORD a word (books first) and
# CONTROL_TOKENS a control word, CONTROL_WORDS controls; a verdict - its votes from MIN_BOOKS books or more (as
# heteronyms.py's)
PER_WORD = 12
CONTROL_TOKENS = 6
CONTROL_WORDS = 150
MIN_BOOKS = 3


# The corpus's tokens of words - {word: [{clip, wav, book, segment}]} (each clip's words by its alignment - readings.py's
# word times and context); clips, [(clip, alignment record)]
def find_tokens(words, clips, wav_of):
    from . import measure  # pylint: disable=import-outside-toplevel
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    found = collections.defaultdict(list)
    for clip, record in clips:
        if not record or not record['labels']:
            continue
        names = [word['text'] for word in tokenize(clip['text'])['words']]
        timed = measure.word_times(record)
        for position, name in enumerate(names):
            if name in words and position < len(timed) and timed[position][1] is not None:
                found[name].append({'clip': clip['id'], 'wav': wav_of(clip), 'book': clip['book'],
                                    'segment': measure.word_context(timed, position, record['seconds'])})
    return found


# Up to count of a word's tokens - one a book at a time (seeded)
def choose_tokens(tokens, count, rng):
    by_book = collections.defaultdict(list)
    for token in tokens:
        by_book[token['book']].append(token)
    queues = [by_book[book] for book in sorted(by_book)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    chosen = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop())
    return chosen


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


# The audio plan - [{word, control, candidates, token}] for the suspects (the label and the alternatives that could
# correct it - the fold models' predictions and the dictionary's variants) and controls the readers spoke
def audio_plan(suspects, controls, found):
    rng = random.Random(PARTITION_SEED)
    plan = []
    def books(tokens):
        return len({token['book'] for token in tokens})
    for entry in suspects:
        candidates = [entry['label']]
        for phones in [prediction for prediction, _, _ in entry['models']] + entry['variants']:
            if all(unstressed(phones) != unstressed(candidate) for candidate in candidates) and \
                    unfixable(entry['word'], entry['label'], phones) is None:
                candidates.append(phones)
        tokens = found.get(entry['word'], [])
        if len(candidates) > 1 and books(tokens) >= MIN_BOOKS:
            for token in choose_tokens(tokens, PER_WORD, rng):
                plan.append({'word': entry['word'], 'control': False, 'candidates': candidates, 'token': token})
    used = 0
    for entry in controls:
        tokens = found.get(entry['word'], [])
        if books(tokens) >= MIN_BOOKS and used < CONTROL_WORDS and \
                unfixable(entry['word'], entry['label'], entry['alternative']) is None:
            used += 1
            for token in choose_tokens(tokens, CONTROL_TOKENS, rng):
                plan.append({'word': entry['word'], 'control': True, 'candidates': [entry['label'], entry['alternative']],
                             'token': token})
    return plan


# The audit's extra clips - book sentences with the words of too few tokens (at most PER_WORD a word), spoken by the
# engine in turn by the voices and aligned - [(clip, alignment record)], in <corpus>/audit/ (kept)
def speak_extra(words, found, settings, directory, jobs):
    from . import ctc, tts  # pylint: disable=import-outside-toplevel
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    wanted = {word: PER_WORD - len(found.get(word, [])) for word in words if len(found.get(word, [])) < PER_WORD}
    if not wanted:
        return []
    recipe = read_json(RECIPE_PATH)
    section = {'books': settings['texts']['books'], 'pins': settings['texts']['pins'], 'texts': recipe['context']['labels']['texts']}
    candidates = collections.defaultdict(list)
    for path in tts.book_paths(section):
        for _, sentence in tts.book_candidates(path):
            for word in set(tts.WORD.findall(sentence.lower())) & wanted.keys():
                if len(candidates[word]) < 4 * wanted[word]:
                    candidates[word].append((os.path.basename(path), sentence))
    texts = tts.checked([sentence for items in candidates.values() for _, sentence in items], settings, jobs)
    order = tts.dealing_order(settings['voices'])
    clips, spoken = [], set()
    for word in sorted(candidates):
        chosen = [(book, texts[sentence]) for book, sentence in candidates[word] if texts.get(sentence)][:wanted[word]]
        for book, text in chosen:
            # (a sentence of two words chosen again in the same voice is the same clip - its tokens once)
            voice = settings['voices'][order[len(clips) % len(order)]]
            clip = {'id': tts.clip_id(voice['voice'], text), 'voice': voice['voice'], 'book': book, 'text': text}
            if clip['id'] not in spoken:
                spoken.add(clip['id'])
                clips.append(clip)
    audit = os.path.join(directory, 'audit')
    tts.check_engine(settings['engine'])
    for clip in clips:
        path = tts.wav_path(audit, clip)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + '.part', 'wb') as file:
                file.write(tts.speak(settings['engine'], clip['voice'], clip['text']))
            os.replace(path + '.part', path)
    records = tts.read_alignments(audit)
    todo = [clip for clip in clips if clip['id'] not in records]
    if todo:
        model, vocab, device = ctc.load_ctc()
        samples_list = [ctc.read_16k(tts.wav_path(audit, clip)) for clip in todo]
        with open(os.path.join(audit, 'alignments.jsonl'), 'a', encoding='utf-8') as file:
            for clip, samples, log_posteriors in zip(todo, samples_list, ctc.ctc_log_posteriors(model, device, samples_list)):
                tokens = tokenize(clip['text'])['tokens']
                result = ctc.ctc_labels(tokens, log_posteriors, vocab, samples)
                labels = None if result is None else [[label, round(begin, 4), round(end, 4)] for label, begin, end in result[0]]
                record = {'id': clip['id'], 'tokens': tokens, 'seconds': round(len(samples) / ctc.CTC_SAMPLE_RATE, 4), 'labels': labels}
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
        records = tts.read_alignments(audit)
    return [(clip, records.get(clip['id'])) for clip in clips]


def command_audio(args):
    """audio.jsonl - each planned token's candidate log likelihoods and vote (resumable: tokens scored before are kept)"""
    import torch  # pylint: disable=import-outside-toplevel
    from . import ctc, tts  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(args.threads)
    with open(os.path.join(args.out, 'suspects.json'), encoding='utf-8') as file:
        suspects = [entry for entry in json.load(file) if not entry['relabeled'] and not entry['heteronym']]
    with open(os.path.join(args.out, 'controls.json'), encoding='utf-8') as file:
        controls = json.load(file)
    settings = tts.load_settings()
    directory = tts.corpus_dir(settings)
    records = tts.read_alignments(directory)
    clips = [(clip, records.get(clip['id'])) for clip in tts.selection(settings)]
    words = {entry['word'] for entry in suspects} | {entry['word'] for entry in controls}
    found = find_tokens(words, clips, lambda clip: tts.wav_path(directory, clip))
    if args.speak:
        extra = speak_extra({entry['word'] for entry in suspects}, found, settings, directory, args.jobs)
        for word, tokens in find_tokens(words, extra, lambda clip: tts.wav_path(os.path.join(directory, 'audit'), clip)).items():
            found[word].extend(tokens)
    plan = audio_plan(suspects, controls, found)
    path = os.path.join(args.out, 'audio.jsonl')
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            done = {(item['word'], item['clip']) for item in map(json.loads, file)}
    plan = [item for item in plan if (item['word'], item['token']['clip']) not in done]
    log_line(args.log, f'audio: {len(plan)} tokens to score ({len(done)} before), '
                       f'{len({item["word"] for item in plan if not item["control"]})} suspects')
    model, vocab, device = ctc.load_ctc('cpu')
    with open(path, 'a', encoding='utf-8') as file:
        for offset in range(0, len(plan), args.batch):
            group = plan[offset:offset + args.batch]
            segments = [ctc.read_segment(item['token']['wav'], *item['token']['segment'][:2]) for item in group]
            posteriors = ctc.ctc_log_posteriors(model, device, segments)
            for item, log_posteriors in zip(group, posteriors):
                _, _, before, after = item['token']['segment']
                scores = [candidate_log_likelihood(log_posteriors, before, candidate.split(), after, vocab)
                          for candidate in item['candidates']]
                file.write(json.dumps({'word': item['word'], 'control': item['control'], 'clip': item['token']['clip'],
                                       'book': item['token']['book'], 'candidates': item['candidates'],
                                       'scores': [round(score, 3) for score in scores], 'vote': vote(scores)}) + '\n')
            file.flush()
            log_line(args.log, f'audio: {offset + len(group)}/{len(plan)} tokens')
    return 0


#
# 4. The verdicts
#

# A word's verdict from its tokens' votes - {verdict, votes (per candidate), books (per candidate), best (the
# alternative with the most votes), lower (the Wilson lower bound of the winner's share against the label or the best)}
def verdict(candidates, votes):
    from .heteronyms import wilson_lower  # pylint: disable=import-outside-toplevel
    counts = [0] * len(candidates)
    books = [set() for _ in candidates]
    for index, book in votes:
        if index is not None:
            counts[index] += 1
            books[index].add(book)
    best = max(range(1, len(candidates)), key=lambda index: (counts[index], -index))
    label_votes, best_votes = counts[0], counts[best]
    result = {'votes': counts, 'books': [len(names) for names in books], 'best': best, 'tokens': len(votes)}
    if wilson_lower(best_votes, best_votes + label_votes) > 0.5 and len(books[best]) >= MIN_BOOKS:
        result.update(verdict='error', lower=round(wilson_lower(best_votes, best_votes + label_votes), 3))
    elif wilson_lower(label_votes, best_votes + label_votes) > 0.5 and len(books[0]) >= MIN_BOOKS:
        result.update(verdict='correct', lower=round(wilson_lower(label_votes, best_votes + label_votes), 3))
    else:
        result.update(verdict='inconclusive' if votes else 'none', lower=None)
    return result


# A word's scored tokens' candidates (every token's, in order - the label first) and votes [(candidate index or None,
# book)] - the tokens may have been scored with different candidates
def token_votes(items):
    candidates = list(dict.fromkeys(candidate for item in items for candidate in item['candidates']))
    votes = [(None if item['vote'] is None else candidates.index(item['candidates'][item['vote']]), item['book']) for item in items]
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
    """verdicts.json (every suspect's verdict, and the controls' error rate) and fixes.json (train/data/g2p-fixes.json's
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
                'audio': {'candidates': result['candidates'], 'votes': result['votes'], 'books': result['books'],
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
    'phones differ, by a log-probability margin) and whose audio confirms the alternative (the CTC phoneme recognizer\'s '
    'likelihood of each candidate on the aligned word with its neighbors: the votes\' Wilson 95% lower bound above one '
    'half, from 3+ speakers - the LibriTTS-R readers, before 1.7 - or 3+ books of the training speech) - or, where noted, '
    'the dictionary\'s own evidence (its derived forms). evidence: was (the dictionary\'s first), folds ([prediction, '
    'margin] per fold model), audio (candidates - the label first - votes and speakers or books per candidate, tokens, '
    'lower), date'
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
    suspects_parser.add_argument('--relabel', default=os.path.join(ROOT, 'train', 'data', 'g2p-relabel.json'))
    suspects_parser.add_argument('--heteronyms', help='the context labels (labels.json) - their words are the heteronym models\'')
    suspects_parser.add_argument('--threads', type=int, default=1)
    suspects_parser.add_argument('--log')
    suspects_parser.set_defaults(func=command_suspects)
    audio_parser = commands.add_parser('audio', help='the training speech\'s votes (resumable)')
    audio_parser.add_argument('--out', required=True)
    audio_parser.add_argument('--speak', action='store_true', help='speak sentences for the suspects of too few tokens (Ollama)')
    audio_parser.add_argument('-j', '--jobs', type=int, default=None, help='the sentences\' check\'s worker processes')
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
