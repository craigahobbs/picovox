# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The heteronym labels - for the G2P network's reading input and ambiguity head and the context network (nn/context.py):
which words have more than one reading the neighboring words decide, their readings in the G2P's order, and labeled
examples of each reading in context. Training-time only: the app has no word list - the G2P's ambiguity head says when
to ask the context network.

    build/venv-nn/bin/python -B -m nn.contextlabels books                  # 0. the recipe's books (Project Gutenberg)
    build/venv-nn/bin/python -B -m nn.contextlabels aligned --out DIR      # 1. the aligned readings (LibriTTS-R MFA labels)
    build/venv-pos/bin/python -B -m nn.contextlabels tag --out DIR         # 2. part-of-speech tags (spaCy, requirements/pos.txt)
    build/venv-nn/bin/python -B -m nn.contextlabels labels --out DIR       # 3. the words, readings, and examples -> labels.json

The label sources (data/nn/context-labels.json is the curated part):

1. Aligned readings - every LibriTTS-R utterance with MFA labels: each token of a candidate word the aligner matched to a
   dictionary variant (nn/readings.py), in its sentence. The speaker's reading, heard in the audio.
2. Silver readings - Project Gutenberg books tagged for part of speech (training time only), each candidate token's tag
   mapped to a reading (the word's own mapping, else its family's - the noun/verb stress pairs, the -ate words, the S/Z
   words), only for the words whose mapping agrees with their aligned readings (the audio) on at least MIN_AGREEMENT
   of at least MIN_VALIDATE tokens.
3. Seeds - hand-labeled sentences (data/nn/context-labels.json "seeds") for the sense heteronyms (lead the metal, tear, bow,
   bass), the readings the dictionary lacks (does the deer, row the quarrel, house the verb), and the pairs whose tag
   mapping the audio doesn't validate (use, close, separate, increase, ...), and collocations
   ("collocations": a neighboring word that fixes the reading, reviewed by hand) over the books.

Held out: the regression set's sentences (data/nn/heteronyms-test.json), 10% of the aligned speakers (by a hash), and
two books. The settings are data/nn/train.json's context.labels.
"""

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import re
import sys

from .frontend import ROOT, context_sequence, text_items
from . import readings as readings_module


SPEC_PATH = os.path.join(ROOT, 'data', 'nn', 'context-labels.json')
TEST_PATH = os.path.join(ROOT, 'data', 'nn', 'heteronyms-test.json')
LIBRITTS_DIR = os.path.join(ROOT, 'data', 'corpora', 'libritts_r', 'LibriTTS_R')
LABEL_DIR = os.path.join(ROOT, 'data', 'corpora', 'libritts_label', 'lab', 'phone')
CONFIG_PATH = os.path.join(ROOT, 'data', 'nn', 'train.json')


# The recipe's label settings - data/nn/train.json's context.labels (and the context network's window)
def load_settings(path=CONFIG_PATH):
    with open(path, encoding='utf-8') as file:
        context = json.load(file)['context']
    return {**context['labels'], 'window': context['network']['window']}


SETTINGS = load_settings()

# The context window - WINDOW items each side of the word (words, and phrase ends as marks)
WINDOW = SETTINGS['window']

# The mapping validation - the least agreement with the aligned readings, and the least aligned tokens for a word's own check
MIN_AGREEMENT = SETTINGS['minAgreement']
MIN_VALIDATE = SETTINGS['minValidate']

# A candidate word is ambiguous (trained with readings) when its alternate readings have at least MIN_ALTERNATE examples
# and MIN_ALTERNATE_SHARE of its examples
MIN_ALTERNATE = SETTINGS['minAlternate']
MIN_ALTERNATE_SHARE = SETTINGS['minAlternateShare']

# The books (Project Gutenberg ids, fetched to the books directory as nn/distill.py fetches its books), the held-out
# books, and the held-out share of the aligned speakers (percent)
BOOKS_DIR = os.path.join(ROOT, SETTINGS['books'])
HELD_OUT_BOOKS = {f'pg{book_id}' for book_id in SETTINGS['heldOutBooks']}
HELD_OUT_PERCENT = SETTINGS['heldOutPercent']

# The most silver examples of a word kept (the most frequent words have thousands)
MAX_SILVER = SETTINGS['maxSilver']

# The part-of-speech tag groups of the families' mappings
NOUN_TAGS = {'NN', 'NNS'}
VERB_TAGS = {'VB', 'VBP', 'VBZ', 'VBD', 'VBN', 'VBG'}


# The curated spec - data/nn/context-labels.json
def load_spec(path=SPEC_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# A variant's vowels
def variant_vowels(variant):
    return [phone for phone in variant.split() if phone[-1].isdigit()]


# The index of a variant's primary-stressed vowel (None if none)
def primary_vowel(variant):
    return next((index for index, phone in enumerate(variant_vowels(variant)) if phone.endswith('1')), None)


# A word's family by its first two readings - 'stress' (the primary stress moves: noun/verb pairs), 'ate' (a final -ate
# full vowel against a reduced one), 'consonant' (S/Z), or 'vowel'
def word_family(word, readings):
    first, second = readings[0][0], readings[1][0]
    first_vowels, second_vowels = variant_vowels(first), variant_vowels(second)
    if len(first_vowels) == len(second_vowels) >= 2 and primary_vowel(first) != primary_vowel(second):
        return 'stress'
    if re.search(r'ates?$', word) and (first_vowels[-1:] + second_vowels[-1:]).count('EY2') == 1:
        return 'ate'
    if readings_module.consonant_change(first.split(), second.split()):
        return 'consonant'
    return 'vowel'


# The voiced and voiceless consonants of the S/Z words' pairs
VOICED = {'Z', 'ZH', 'V', 'DH', 'D', 'G', 'B', 'JH'}
VOICELESS = {'S', 'SH', 'F', 'TH', 'T', 'K', 'P', 'CH'}


# The reading (dictionary order) a family's mapping gives a tag - None if the tag isn't mapped
def family_reading(family, tag, readings):
    if family == 'stress':
        stresses = [primary_vowel(reading[0]) for reading in readings]
        if tag in NOUN_TAGS:
            return stresses.index(min(stresses))
        if tag in VERB_TAGS:
            return stresses.index(max(stresses))
    elif family == 'ate':
        full = [index for index, reading in enumerate(readings) if variant_vowels(reading[0])[-1:] == ['EY2']]
        reduced = [index for index in range(len(readings)) if index not in full]
        if tag in {'VB', 'VBP', 'VBZ'} and full:
            return full[0]
        if tag in NOUN_TAGS | {'JJ'} and reduced:
            return reduced[0]
    elif family == 'consonant':
        ops = [(left, right) for left, right in readings_module.edit_ops(readings[0][0].split(), readings[1][0].split())
               if left and right]
        voiced = 1 if ops and ops[0][0] in VOICELESS and ops[0][1] in VOICED else 0
        if tag in {'VB', 'VBP'}:
            return voiced
        if tag in {'NN', 'NNS'}:
            return 1 - voiced
    return None


# Whether a family word's readings are a stem-final voicing pair (use, close, excuse, uses - not an accent's S/Z or CH/SH)
def voicing_pair(readings):
    first, second = readings[0][0].split(), readings[1][0].split()
    ops = readings_module.edit_ops(first, second)
    if len(ops) != 1 or None in ops[0] or {ops[0][0], ops[0][1]} != {'S', 'Z'}:
        return False
    position = next(index for index, (left, right) in enumerate(zip(first, second)) if left != right)
    return position >= len(first) - 3


# The candidate words and their readings (dictionary order) - the inventory's family words (the noun/verb stress pairs,
# the -ate words, the S/Z words) and the spec's words (with the readings the dictionary lacks) - {word: {readings, family,
# tags, curated}}; tags maps a tag ("Cap" + tag for a capitalized token) to a reading for a word's own mapping,
# before its family's
def candidate_words(spec, inventory=None):
    inventory = readings_module.inventory() if inventory is None else inventory
    variants = readings_module.read_variants()
    words = {}
    for word, readings in inventory.items():
        family = word_family(word, readings)
        if family == 'consonant' and not voicing_pair(readings):
            family = 'vowel'
        if family != 'vowel' and word not in spec['exclude']:
            words[word] = {'readings': readings, 'family': family, 'tags': {}, 'curated': False}
    for word, entry in spec['words'].items():
        readings = [list(reading) for reading in inventory.get(word) or readings_module.readings_of(variants[word])]
        readings.extend([variant] for variant in entry.get('missing', []))
        tags = {}
        for variant, tag_list in entry.get('tags', {}).items():
            reading = readings_module.reading_of(variant, readings)
            for tag in tag_list:
                tags[tag] = reading
        words[word] = {'readings': readings, 'family': entry.get('family', 'own'), 'tags': tags,
                       'curated': True}
    return words


# The marks of a phrase's start (and the text's) in a context sequence
STARTS = {'', ',', '.', '?', '!'}


# The tagged reading of a candidate token (dictionary order) - its word's own mapping, else its family's; None if unmapped.
# An own mapping's key is a tag ("Cap" + tag for a capitalized token), optionally with the words that must come before
# ("VB<to|will|^" - "^" a phrase's start), or must not ("VBD<!and|or").
def tag_reading(entry, tag, surface=None, previous=''):
    capital = surface is not None and surface[:1].isupper()
    for key, reading in entry['tags'].items():
        key_tag, _, before = key.partition('<')
        if key_tag.startswith('Cap'):
            if not capital:
                continue
            key_tag = key_tag[3:]
        if key_tag != tag:
            continue
        negated = before.startswith('!')
        allowed = set(before.lstrip('!').split('|')) if before else None
        matched = allowed is not None and (previous in allowed or ('^' in allowed and previous in STARTS))
        if allowed is None or matched != negated:
            return reading
    if entry['family'] == 'own' or any(key.partition('<')[0].removeprefix('Cap') == tag for key in entry['tags']):
        return None
    return family_reading(entry['family'], tag, entry['readings'])


# The previous context item of each word item of a text - {offset: item} (the first word item at an offset)
def previous_items(text):
    items = text_items(text)
    sequence = context_sequence(items)
    previous = {}
    for index, (item, offset) in enumerate(items):
        if isinstance(item, str) and offset not in previous:
            previous[offset] = sequence[index - 1] if index > 0 else ''
    return previous


# A context window - the WINDOW items each side of the index ('' beyond the text), the word in the middle
def context_window(sequence, index):
    return [sequence[position] if 0 <= position < len(sequence) else '' for position in range(index - WINDOW, index + WINDOW + 1)]


#
# 1. The aligned readings
#

# The LibriTTS-R utterances with MFA labels - [(id, subset, speaker, text path, phone label path)]
def libritts_utterances():
    utterances = []
    for subset in sorted(os.listdir(LABEL_DIR)):
        for speaker in sorted(os.listdir(os.path.join(LABEL_DIR, subset))):
            for chapter in sorted(os.listdir(os.path.join(LABEL_DIR, subset, speaker))):
                for name in sorted(os.listdir(os.path.join(LABEL_DIR, subset, speaker, chapter))):
                    utterance = name[:-len('.lab')]
                    text_path = os.path.join(LIBRITTS_DIR, subset, speaker, chapter, utterance + '.normalized.txt')
                    if name.endswith('.lab') and os.path.exists(text_path):
                        utterances.append((utterance, subset, speaker, text_path, os.path.join(LABEL_DIR, subset, speaker, chapter, name)))
    return utterances


_WORDS = None


# The aligned candidate tokens of an utterance (a worker task) - a record or None
def aligned_task(utterance):
    utterance_id, subset, speaker, text_path, phone_lab = utterance
    with open(text_path, encoding='utf-8') as file:
        text = file.read().strip()
    inventory = {word: entry['readings'] for word, entry in _WORDS.items()}
    _, tokens = readings_module.utterance_readings(text, phone_lab, inventory, lambda word: [])
    if not tokens:
        return None
    offsets = [offset for item, offset in text_items(text) if isinstance(item, str)]
    return {'id': utterance_id, 'subset': subset, 'speaker': speaker, 'text': text,
            'tokens': [{'position': token['position'], 'offset': offsets[token['position']], 'word': token['word'],
                        'reading': token['reading']} for token in tokens]}


# Set a worker's candidate words
def set_words(words):
    global _WORDS  # pylint: disable=global-statement
    _WORDS = words


def aligned_command(args):
    """The aligned readings of the candidate words - <out>/aligned.jsonl"""
    words = candidate_words(load_spec())
    utterances = libritts_utterances()
    os.makedirs(args.out, exist_ok=True)
    count = 0
    with multiprocessing.Pool(args.jobs, initializer=set_words, initargs=(words,)) as pool, \
            open(os.path.join(args.out, 'aligned.jsonl'), 'w', encoding='utf-8') as file:
        for record in pool.imap(aligned_task, utterances, chunksize=64):
            if record is not None:
                file.write(json.dumps(record) + '\n')
                count += len(record['tokens'])
    print(f'aligned: {len(utterances)} utterances, {count} candidate tokens with an aligned reading ({len(words)} candidate words)')
    return 0


#
# 2. Part-of-speech tags (spaCy - training time only)
#

# The book texts - [(book id, text)] for the recipe's books (in the books directory), the Project Gutenberg header and
# footer removed
def book_texts(books_dir=BOOKS_DIR):
    books = []
    for book_id in sorted(f'pg{book_id}' for book_id in SETTINGS['bookIds']):
        with open(os.path.join(books_dir, f'{book_id}.txt'), encoding='utf-8-sig', errors='replace') as file:
            text = file.read()
        start = re.search(r'\*\*\* ?START OF.*?\*\*\*', text)
        end = re.search(r'\*\*\* ?END OF', text)
        books.append((book_id, text[start.end() if start else 0:end.start() if end else len(text)]))
    return books


def books_command(args):
    """Fetch the recipe's books (Project Gutenberg, by id - nn/distill.py's book_paths) into the books directory"""
    from .distill import book_paths  # pylint: disable=import-outside-toplevel
    paths = book_paths({'books': args.books or BOOKS_DIR, 'bookIds': SETTINGS['bookIds']})
    print(f'books: {len(paths)} in {args.books or BOOKS_DIR}')
    return 0


# A book's paragraphs - blank-line separated, each joined to one line
def paragraphs(text):
    return [' '.join(part.split()) for part in re.split(r'\n\s*\n', text) if part.strip()]


# The spaCy tags of texts at the candidate words' offsets - [{offset: tag}] (n_process worker processes)
def tag_texts(texts, words, jobs):
    import spacy  # pylint: disable=import-outside-toplevel
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])
    results = []
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=64, n_process=jobs)):
        offsets = {offset for item, offset in text_items(text) if isinstance(item, str) and item in words}
        results.append({str(token.idx): token.tag_ for token in doc if token.idx in offsets})
    return results


def tag_command(args):
    """Tag the aligned utterances and the books' paragraphs with the candidate words - <out>/aligned-tags.jsonl,
    <out>/books.jsonl"""
    words = candidate_words(load_spec())
    with open(os.path.join(args.out, 'aligned.jsonl'), encoding='utf-8') as file:
        records = [json.loads(line) for line in file]
    tags = tag_texts([record['text'] for record in records], words, args.jobs)
    with open(os.path.join(args.out, 'aligned-tags.jsonl'), 'w', encoding='utf-8') as file:
        for record, record_tags in zip(records, tags):
            file.write(json.dumps({'id': record['id'], 'tags': record_tags}) + '\n')
    print(f'tag: {len(records)} aligned utterances', flush=True)
    count = 0
    with open(os.path.join(args.out, 'books.jsonl'), 'w', encoding='utf-8') as file:
        for book, text in book_texts(args.books or BOOKS_DIR):
            chosen = [paragraph for paragraph in paragraphs(text) if len(paragraph) < 5000 and
                      any(isinstance(item, str) and item in words for item, _ in text_items(paragraph))]
            for paragraph, paragraph_tags in zip(chosen, tag_texts(chosen, words, args.jobs)):
                file.write(json.dumps({'book': book, 'text': paragraph, 'tags': paragraph_tags}) + '\n')
            count += len(chosen)
            print(f'tag: {book} {len(chosen)} paragraphs ({count} in all)', flush=True)
    return 0


#
# 3. The labels
#

# The held-out examples' windows - the regression set's
def regression_windows():
    with open(TEST_PATH, encoding='utf-8') as file:
        items = json.load(file)
    windows = set()
    for item in items:
        sequence = context_sequence(text_items(item['text']))
        positions = [index for index, entry in enumerate(sequence) if entry == item['word']]
        windows.add(tuple(context_window(sequence, positions[item.get('occurrence', 1) - 1])))
    return windows


# Whether an aligned speaker is held out
def held_out_speaker(speaker):
    return int(hashlib.sha1(speaker.encode('utf-8')).hexdigest(), 16) % 100 < HELD_OUT_PERCENT


# The examples of texts' words - (text, [(offset, reading in dictionary order)]) - [{window, word, reading}]
def text_examples(text, labeled):
    items = text_items(text)
    sequence = context_sequence(items)
    by_offset = {offset: index for index, (_, offset) in reversed(list(enumerate(items)))}
    examples = []
    for offset, word, reading in labeled:
        index = by_offset.get(offset)
        if index is not None and sequence[index] == word:
            examples.append({'window': context_window(sequence, index), 'word': word, 'reading': reading})
    return examples


# The validation of the tag mappings against the aligned readings - {word: {agree, total}} and {family: {agree, total}}
def validate_mappings(words, aligned, aligned_tags):
    by_word = collections.defaultdict(lambda: {'agree': 0, 'total': 0})
    by_family = collections.defaultdict(lambda: {'agree': 0, 'total': 0})
    for record in aligned:
        tags = aligned_tags.get(record['id'], {})
        previous = previous_items(record['text'])
        for token in record['tokens']:
            entry = words[token['word']]
            tag = tags.get(str(token['offset']))
            surface = record['text'][token['offset']:token['offset'] + len(token['word'])]
            mapped = tag_reading(entry, tag, surface, previous.get(token['offset'], '')) if tag else None
            if mapped is None:
                continue
            agree = mapped == token['reading']
            groups = [by_word[token['word']]]
            if entry['family'] != 'own':
                groups.append(by_family[entry['family']])
            for counts in groups:
                counts['total'] += 1
                counts['agree'] += agree
    return dict(by_word), dict(by_family)


# Whether a word's tag mapping is validated - it agrees with the word's aligned readings (the audio) on at least
# MIN_AGREEMENT of at least MIN_VALIDATE tokens
def mapping_valid(word, by_word):
    own = by_word.get(word, {'agree': 0, 'total': 0})
    return own['total'] >= MIN_VALIDATE and own['agree'] / own['total'] >= MIN_AGREEMENT


# The G2P's default reading of each word (dictionary order) - the reading of its G2P training target (nn/g2p.py's data
# with the relabeled targets: the readers' pronunciation, else the dictionary's first)
def default_readings(words):
    from . import g2p  # pylint: disable=import-outside-toplevel
    targets = g2p.load_data(g2p.load_config()['stages'][-1].get('relabel') and
                            os.path.join(ROOT, g2p.load_config()['stages'][-1]['relabel']))['first']
    return {word: readings_module.reading_of(' '.join(targets[word]), entry['readings']) if word in targets else 0
            for word, entry in words.items()}


def labels_command(args):
    """The ambiguous words, their readings in the G2P's order, and the labeled examples - <out>/labels.json"""
    spec = load_spec()
    words = candidate_words(spec)
    with open(os.path.join(args.out, 'aligned.jsonl'), encoding='utf-8') as file:
        aligned = [json.loads(line) for line in file]
    with open(os.path.join(args.out, 'aligned-tags.jsonl'), encoding='utf-8') as file:
        aligned_tags = {record['id']: record['tags'] for record in map(json.loads, file)}
    by_word, by_family = validate_mappings(words, aligned, aligned_tags)
    valid = {word for word in words if mapping_valid(word, by_word)}
    eligible = valid | {word for word, entry in words.items() if entry['curated']} | {seed['word'] for seed in spec['seeds']}

    # The examples (readings in dictionary order) - aligned, silver, seeds, and collocations
    examples = []
    for record in aligned:
        split = 'heldout' if held_out_speaker(record['speaker']) else 'train'
        labeled = [(token['offset'], token['word'], token['reading']) for token in record['tokens'] if token['word'] in eligible]
        for example in text_examples(record['text'], labeled):
            examples.append({**example, 'source': 'aligned', 'split': split, 'group': record['speaker']})
    silver = collections.Counter()
    collocations = {word: [(re.compile(pattern), variant) for pattern, variant in entries] for word, entries in spec['collocations'].items()}
    with open(os.path.join(args.out, 'books.jsonl'), encoding='utf-8') as file:
        for record in map(json.loads, file):
            split = 'heldout' if record['book'] in HELD_OUT_BOOKS else 'train'
            labeled, colloc = [], []
            previous = previous_items(record['text'])
            for item, offset in text_items(record['text']):
                if not isinstance(item, str) or item not in words:
                    continue
                entry = words[item]
                tag = record['tags'].get(str(offset))
                surface = record['text'][offset:offset + len(item)]
                reading = tag_reading(entry, tag, surface, previous.get(offset, '')) if tag and item in valid else None
                if reading is not None:
                    labeled.append((offset, item, reading))
                if item in collocations:
                    context = record['text'][max(0, offset - 40):offset + len(item) + 40].lower()
                    for pattern, variant in collocations[item]:
                        if pattern.search(context):
                            colloc.append((offset, item, readings_module.reading_of(variant, entry['readings'])))
                            break
            for source, labeled_items in (('silver', labeled), ('collocation', colloc)):
                for example in text_examples(record['text'], labeled_items):
                    if source == 'silver':
                        silver[example['word']] += 1
                        if silver[example['word']] > MAX_SILVER:
                            continue
                    examples.append({**example, 'source': source, 'split': split, 'group': record['book']})
    for seed in spec['seeds']:
        entry = words[seed['word']]
        reading = readings_module.reading_of(seed['phones'], entry['readings'])
        for text in seed['texts']:
            offset = next(offset for item, offset in text_items(text) if item == seed['word'])
            for example in text_examples(text, [(offset, seed['word'], reading)]):
                examples.append({**example, 'source': 'seed', 'split': 'train', 'group': 'seed'})
    held_out = regression_windows()
    examples = [example for example in examples if tuple(example['window']) not in held_out]

    # The ambiguous words - an alternate reading common enough in the examples; their readings in the G2P's order (the
    # default first, then by the aligned readers' counts, else dictionary order)
    counts = collections.defaultdict(collections.Counter)
    aligned_counts = collections.defaultdict(collections.Counter)
    for example in examples:
        counts[example['word']][example['reading']] += 1
        if example['source'] == 'aligned':
            aligned_counts[example['word']][example['reading']] += 1
    defaults = default_readings(words)
    ambiguous = {}
    for word, entry in sorted(words.items()):
        default = defaults[word]
        if default is None or word in spec['exclude']:
            continue
        total = sum(counts[word].values())
        common = [reading for reading, count in counts[word].items() if count >= max(MIN_ALTERNATE, MIN_ALTERNATE_SHARE * total)]
        if word not in eligible or len(common) < 2:
            continue
        others = sorted((reading for reading in range(len(entry['readings'])) if reading != default),
                        key=lambda reading: (-aligned_counts[word][reading], reading))
        order = [default] + others
        ambiguous[word] = {'readings': [entry['readings'][reading] for reading in order], 'order': order,
                           'family': entry['family'], 'counts': [counts[word][reading] for reading in order]}
    labeled = []
    for example in examples:
        entry = ambiguous.get(example['word'])
        if entry is not None:
            labeled.append({'window': example['window'], 'word': example['word'], 'slot': entry['order'].index(example['reading']),
                            'source': example['source'], 'split': example['split'], 'group': example['group']})
    validation = {'words': {word: by_word[word] for word in sorted(by_word)}, 'families': by_family}
    with open(os.path.join(args.out, 'labels.json'), 'w', encoding='utf-8') as file:
        json.dump({'words': ambiguous, 'validation': validation, 'valid': sorted(valid), 'examples': labeled}, file)
    sources = collections.Counter((example['source'], example['split']) for example in labeled)
    print(f'labels: {len(ambiguous)} ambiguous words ({len(words)} candidates, {len(valid)} with a validated mapping), '
          f'{len(labeled)} examples: ' + ', '.join(f'{source} {split} {count}' for (source, split), count in sorted(sources.items())))
    print('families (aligned agreement): ' + ', '.join(f'{family} {counts["agree"]}/{counts["total"]}' for family, counts in sorted(by_family.items())))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.contextlabels', description='The heteronym labels')
    commands = parser.add_subparsers(dest='command', required=True)
    aligned_parser = commands.add_parser('aligned', help='the aligned readings (LibriTTS-R MFA labels)')
    aligned_parser.add_argument('--out', required=True)
    aligned_parser.add_argument('-j', '--jobs', type=int, default=6)
    aligned_parser.set_defaults(func=aligned_command)
    books_parser = commands.add_parser('books', help='fetch the recipe\'s Project Gutenberg books')
    books_parser.add_argument('--books', help='the books directory (default: the recipe\'s)')
    books_parser.set_defaults(func=books_command)
    tag_parser = commands.add_parser('tag', help='part-of-speech tags (spaCy en_core_web_sm)')
    tag_parser.add_argument('--out', required=True)
    tag_parser.add_argument('--books', help='the books directory (default: the recipe\'s - nn.contextlabels books)')
    tag_parser.add_argument('-j', '--jobs', type=int, default=5)
    tag_parser.set_defaults(func=tag_command)
    labels_parser = commands.add_parser('labels', help='the ambiguous words, readings, and examples')
    labels_parser.add_argument('--out', required=True)
    labels_parser.set_defaults(func=labels_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
