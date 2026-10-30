# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The heteronym labels - for the G2P network's reading input and ambiguity head and the context network (context.py):
which words have more than one reading the neighboring words decide, their readings in the G2P's order, and labeled
examples of each reading in context. Training-time only: the app has no word list - the G2P's ambiguity head says when
to ask the context network. The commands (each with PYTHONPATH=train/src):

    build/venv-train/bin/python -B -m picovox_train.contextlabels books                  # 0. the recipe's books (pinned - books.py)
    build/venv-train/bin/python -B -m picovox_train.contextlabels aligned --out DIR      # 1. the measured readings (the training speech)
    build/venv-train/bin/python -B -m picovox_train.contextlabels tag --out DIR [-j]     # 2. part-of-speech tags
    build/venv-train/bin/python -B -m picovox_train.contextlabels labels --out DIR       # 3. the labels -> labels.json

(-j: tag's worker processes - its batches and books are independent; the outputs are the same for any -j.)

The label sources (train/data/context-labels.json is the curated part):

1. Aligned readings - the training speech's measured readings (measure.py - each candidate word's token in a clip whose
   reading the CTC vote or the formants measured), in its sentence: the voice's reading, heard in the audio - of the
   inventory's family words and the context gate's words only (the curated words' readings in context are the engine's
   weakest: theirs are the seeds, collocations, and their own hand-written mappings). A token a collocation decides is
   the collocation's example, not an aligned one (the corpus's disagreements with the collocations are reported - the
   engine's own reading errors).
2. Silver readings - Project Gutenberg books (pinned texts - books.py, train/data/books.json) tagged for part of speech
   (training time only), each candidate token's tag mapped to a reading (the word's own mapping, else its family's -
   the noun/verb stress pairs, the -ate words, the S/Z words), for the curated words (their mappings hand-written) and
   the words whose mapping agrees with their aligned readings (the audio) on at least MIN_AGREEMENT of at least
   MIN_VALIDATE tokens - or, for a family word of fewer, whose readings the corpus can't measure (stress alone, or a
   class whose method isn't used), its family's mapping as it is, else when its family's agrees on MIN_AGREEMENT of
   at least MIN_FAMILY_VALIDATE.
3. Seeds - hand-labeled sentences (train/data/context-labels.json "seeds") for the sense heteronyms (lead the metal, tear, bow,
   bass), the readings the dictionary lacks (does the deer, row the quarrel, house the verb), and the pairs whose tag
   mapping the audio doesn't validate (use, close, separate, increase, ...), and collocations
   ("collocations": a neighboring word that fixes the reading, reviewed by hand) over the books.

Held out: the regression set's sentences (train/data/heteronyms-test.json) and two books (every source's examples of
them - the voices share one front end, so the book is the independent unit). The settings are train/data/train.json's
context.labels.

The transfer windows - unlabeled: the ambiguous words' other windows in the books (not the held-out books), up to
"maxTransfer" a word - are where the context network's teachers' mean probabilities are distilled into it (context.py).
"""

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import re
import sys

from .frontend import context_sequence, text_items
from . import RECIPE_PATH, ROOT, books, read_json
from . import readings as readings_module
from .readings import candidate_words, load_spec, primary_vowel, variant_vowels


TEST_PATH = os.path.join(ROOT, 'train', 'data', 'heteronyms-test.json')


# The recipe's label settings - train/data/train.json's context.labels (and the context network's window)
def load_settings(path=RECIPE_PATH):
    with open(path, encoding='utf-8') as file:
        context = json.load(file)['context']
    return {**context['labels'], 'window': context['network']['window'], 'gateWords': context['gate']['words']}


SETTINGS = load_settings()

# The context window - WINDOW items each side of the word (words, and phrase ends as marks)
WINDOW = SETTINGS['window']

# The context gate's words (context.py) - their aligned readings are examples though they're curated (the gate's
# held-out accuracy is theirs)
GATE_WORDS = set(SETTINGS['gateWords'])

# The mapping validation - the least agreement with the aligned readings, and the least aligned tokens for a word's own
# check, and for its family's (a family word of too few)
MIN_AGREEMENT = SETTINGS['minAgreement']
MIN_VALIDATE = SETTINGS['minValidate']
MIN_FAMILY_VALIDATE = SETTINGS['minFamilyValidate']

# A candidate word is ambiguous (trained with readings) when its alternate readings have at least MIN_ALTERNATE examples
# and MIN_ALTERNATE_SHARE of its examples (a seeded reading always counts - the seeds say it occurs); a word whose
# mapping only its family's rule gives (readings the corpus can't measure) needs MIN_UNMEASURED_ALTERNATE examples of
# readings other than its most common - the book texts' noun/verb pairs common enough to learn, not every family word
MIN_ALTERNATE = SETTINGS['minAlternate']
MIN_ALTERNATE_SHARE = SETTINGS['minAlternateShare']
MIN_UNMEASURED_ALTERNATE = SETTINGS['minUnmeasuredAlternate']

# The books (pinned Project Gutenberg texts - books.py - fetched to the books directory), and the held-out books
BOOKS_DIR = os.path.join(ROOT, SETTINGS['books'])
PINS_PATH = os.path.join(ROOT, SETTINGS['pins'])
HELD_OUT_BOOKS = set(SETTINGS['heldOutBooks'])

# The most silver examples of a word kept (the most frequent words have thousands)
MAX_SILVER = SETTINGS['maxSilver']

# The most transfer windows of a word kept - unlabeled windows of the ambiguous words in the books (not the held-out
# books), for the context network's distillation (context.py)
MAX_TRANSFER = SETTINGS['maxTransfer']

# The part-of-speech tag groups of the families' mappings
NOUN_TAGS = {'NN', 'NNS'}
VERB_TAGS = {'VB', 'VBP', 'VBZ', 'VBD', 'VBN', 'VBG'}


# The voiced and voiceless consonants of the S/Z words' pairs
VOICED = {'Z', 'ZH', 'V', 'DH', 'D', 'G', 'B', 'JH'}
VOICELESS = {'S', 'SH', 'F', 'TH', 'T', 'K', 'P', 'CH'}


# The reading (dictionary order) a family's mapping gives a tag - None if the tag isn't mapped
def family_reading(family, tag, readings):
    if family == 'stress':
        # (a reading without a primary stress - the rule can't place it, so the tag isn't mapped)
        stresses = [primary_vowel(reading[0]) for reading in readings]
        if None in stresses:
            return None
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

def aligned_command(args):
    """The measured readings of the candidate words in the training speech - <out>/aligned.jsonl"""
    from . import tts  # pylint: disable=import-outside-toplevel
    words = candidate_words(load_spec())
    records = readings_module.read_readings(tts.corpus_dir(tts.load_settings()))
    if records is None:
        print('aligned: the corpus\'s readings aren\'t measured - python -m picovox_train.measure', file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    count = 0
    with open(os.path.join(args.out, 'aligned.jsonl'), 'w', encoding='utf-8') as file:
        for record in records:
            tokens = [token for token in record['tokens'] if token['word'] in words and token['reading'] is not None]
            if tokens:
                offsets = [offset for item, offset in text_items(record['text']) if isinstance(item, str)]
                file.write(json.dumps({'id': record['id'], 'book': os.path.splitext(record['book'])[0], 'voice': record['key'],
                                       'text': record['text'],
                                       'tokens': [{'position': token['position'], 'offset': offsets[token['position']],
                                                   'word': token['word'], 'reading': token['reading']} for token in tokens]}) + '\n')
                count += len(tokens)
    print(f'aligned: {len(records)} measured clips, {count} candidate tokens with a measured reading ({len(words)} candidate words)')
    return 0


#
# 2. Part-of-speech tags (spaCy - training time only)
#

# The recipe's books - their text names, in order (each the book's name in the labels - pg<id>)
def book_names():
    return sorted(SETTINGS['texts'])


# A book's text - the pinned text (books.py: fetched if missing, verified), the Project Gutenberg header and footer removed
def book_text(books_dir, name, pins):
    with open(books.pinned_path(books_dir, name, pins), encoding='utf-8-sig', errors='replace') as file:
        text = file.read()
    start = re.search(r'\*\*\* ?START OF.*?\*\*\*', text)
    end = re.search(r'\*\*\* ?END OF', text)
    return text[start.end() if start else 0:end.start() if end else len(text)]


def books_command(args):
    """Fetch the recipe's books (pinned texts - books.py) into the books directory, each verified against its pin"""
    books_dir = args.books or BOOKS_DIR
    try:
        paths = books.pinned_paths(books_dir, book_names(), PINS_PATH)
    except books.PinError as error:
        print(f'books: {error}', file=sys.stderr)
        return 1
    print(f'books: {len(paths)} pinned texts in {books_dir}')
    return 0


# A book's paragraphs - blank-line separated, each joined to one line
def paragraphs(text):
    return [' '.join(part.split()) for part in re.split(r'\n\s*\n', text) if part.strip()]


# spaCy's batch - the texts are tagged in batches of TAG_BATCH, as one process tags them, and the aligned utterances are
# split among the workers in chunks of whole batches (the batches - and so the tags - are the same for any -j)
TAG_BATCH = 64
TAG_CHUNK = 8 * TAG_BATCH

_TAGGER = None


# Set a tagging worker's spaCy pipeline (the part-of-speech tagger - en_core_web_sm) and the candidate words
def set_tagger(words):
    import spacy  # pylint: disable=import-outside-toplevel,import-error # (build/venv-train only - not the test environment's)
    global _TAGGER  # pylint: disable=global-statement
    _TAGGER = (spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer']), words)


# The spaCy tags of texts at the candidate words' offsets (a worker task) - [{offset: tag}]
def tag_texts(texts):
    nlp, words = _TAGGER
    results = []
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=TAG_BATCH)):
        offsets = {offset for item, offset in text_items(text) if isinstance(item, str) and item in words}
        results.append({str(token.idx): token.tag_ for token in doc if token.idx in offsets})
    return results


# A book's tagged paragraphs (a worker task) - (name, [(paragraph, {offset: tag})]): its paragraphs under 5,000
# characters with a candidate word
def tag_book(task):
    books_dir, name, pins = task
    words = _TAGGER[1]
    chosen = [paragraph for paragraph in paragraphs(book_text(books_dir, name, pins)) if len(paragraph) < 5000 and
              any(isinstance(item, str) and item in words for item, _ in text_items(paragraph))]
    return name, list(zip(chosen, tag_texts(chosen)))


def tag_command(args):
    """Tag the aligned clips and the books' paragraphs with the candidate words - <out>/aligned-tags.jsonl,
    <out>/books.jsonl (-j worker processes, each tagging whole batches of utterances and whole books)"""
    words = candidate_words(load_spec())
    books_dir = args.books or BOOKS_DIR
    pins = books.load_pins(PINS_PATH)
    try:
        for name in book_names():
            books.pinned_path(books_dir, name, pins)
    except books.PinError as error:
        print(f'tag: {error}', file=sys.stderr)
        return 1
    with open(os.path.join(args.out, 'aligned.jsonl'), encoding='utf-8') as file:
        records = [json.loads(line) for line in file]
    texts = [record['text'] for record in records]
    with multiprocessing.Pool(args.jobs, initializer=set_tagger, initargs=(words,)) as pool:
        chunks = [texts[index:index + TAG_CHUNK] for index in range(0, len(texts), TAG_CHUNK)]
        tags = [result for chunk in pool.imap(tag_texts, chunks) for result in chunk]
        with open(os.path.join(args.out, 'aligned-tags.jsonl'), 'w', encoding='utf-8') as file:
            for record, record_tags in zip(records, tags):
                file.write(json.dumps({'id': record['id'], 'tags': record_tags}) + '\n')
        print(f'tag: {len(records)} aligned clips', flush=True)
        count = 0
        with open(os.path.join(args.out, 'books.jsonl'), 'w', encoding='utf-8') as file:
            for name, tagged in pool.imap(tag_book, [(books_dir, name, pins) for name in book_names()]):
                for paragraph, paragraph_tags in tagged:
                    file.write(json.dumps({'book': name, 'text': paragraph, 'tags': paragraph_tags}) + '\n')
                count += len(tagged)
                print(f'tag: {name} {len(tagged)} paragraphs ({count} in all)', flush=True)
    return 0


#
# 3. The labels
#

# The held-out examples' windows - the regression set's
def regression_windows():
    items = read_json(TEST_PATH)
    windows = set()
    for item in items:
        sequence = context_sequence(text_items(item['text']))
        positions = [index for index, entry in enumerate(sequence) if entry == item['word']]
        windows.add(tuple(context_window(sequence, positions[item.get('occurrence', 1) - 1])))
    return windows


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


# The reading a collocation gives a word's token at a text's offset (dictionary order), or None
def collocation_reading(collocations, words, text, offset, word):
    context = text[max(0, offset - 40):offset + len(word) + 40].lower()
    for pattern, variant in collocations.get(word, []):
        if pattern.search(context):
            return readings_module.reading_of(variant, words[word]['readings'])
    return None


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
# MIN_AGREEMENT of at least MIN_VALIDATE tokens; a family word of fewer whose readings the corpus can't measure (stress
# alone, or a class whose method its controls didn't pass - measurable False) takes its family's mapping as it is (the
# noun/verb stress pairs, the -ate words); another family word of fewer, its family's on MIN_AGREEMENT of at least
# MIN_FAMILY_VALIDATE (0: none)
def mapping_valid(word, by_word, family=None, by_family=None, measurable=True):
    own = by_word.get(word, {'agree': 0, 'total': 0})
    if own['total'] >= MIN_VALIDATE:
        return own['agree'] / own['total'] >= MIN_AGREEMENT
    if family not in (None, 'own') and not measurable:
        return True
    group = (by_family or {}).get(family, {'agree': 0, 'total': 0})
    return MIN_FAMILY_VALIDATE > 0 and group['total'] >= MIN_FAMILY_VALIDATE and group['agree'] / group['total'] >= MIN_AGREEMENT


# The classes of readings the corpus measures - {class: used} (measure.py's summary; none when it isn't measured)
def measured_classes():
    from . import tts  # pylint: disable=import-outside-toplevel
    path = os.path.join(tts.corpus_dir(tts.load_settings()), 'readings', 'summary.json')
    return read_json(path)['classes'] if os.path.exists(path) else {}


# The G2P's default reading of each word (dictionary order) - the reading of its G2P training target (g2p.py's data
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
    classes = measured_classes()
    valid = {word for word, entry in words.items()
             if entry['curated'] or mapping_valid(word, by_word, entry['family'], by_family,
                                                  classes.get(readings_module.word_class(entry['readings']), False))}
    seeded = collections.defaultdict(set)
    for seed in spec['seeds']:
        seeded[seed['word']].add(readings_module.reading_of(seed['phones'], words[seed['word']]['readings']))
    unmeasured = {word for word in valid if not words[word]['curated'] and word not in seeded and
                  not mapping_valid(word, by_word, words[word]['family'], by_family)}
    eligible = valid | {word for word, entry in words.items() if entry['curated']} | set(seeded)

    # The examples (readings in dictionary order) - aligned (a collocation's token aside), silver, seeds, and collocations
    collocations = {word: [(re.compile(pattern), variant) for pattern, variant in entries]
                    for word, entries in spec['collocations'].items()}
    examples = []
    disagree = collections.Counter()
    for record in aligned:
        split = 'heldout' if record['book'] in HELD_OUT_BOOKS else 'train'
        labeled = []
        for token in record['tokens']:
            reading = collocation_reading(collocations, words, record['text'], token['offset'], token['word'])
            if reading is not None:
                disagree[reading != token['reading']] += 1
            elif token['word'] in eligible and (not words[token['word']]['curated'] or token['word'] in GATE_WORDS):
                labeled.append((token['offset'], token['word'], token['reading']))
        for example in text_examples(record['text'], labeled):
            examples.append({**example, 'source': 'aligned', 'split': split, 'group': record['book']})
    silver = collections.Counter()
    transfer = collections.defaultdict(dict)
    with open(os.path.join(args.out, 'books.jsonl'), encoding='utf-8') as file:
        for record in map(json.loads, file):
            split = 'heldout' if record['book'] in HELD_OUT_BOOKS else 'train'
            if split == 'train' and MAX_TRANSFER:
                sequence = context_sequence(text_items(record['text']))
                for index, item in enumerate(sequence):
                    if isinstance(item, str) and item in words:
                        window = context_window(sequence, index)
                        transfer[item][hashlib.sha1(json.dumps(window).encode('utf-8')).hexdigest()] = window
            labeled, colloc = [], []
            previous = previous_items(record['text'])
            for item, offset in text_items(record['text']):
                if not isinstance(item, str) or item not in words:
                    continue
                entry = words[item]
                tag = record['tags'].get(str(offset))
                surface = record['text'][offset:offset + len(item)]
                # (a collocation decides its token - no silver example of it, which may disagree)
                reading = collocation_reading(collocations, words, record['text'], offset, item)
                if reading is not None:
                    colloc.append((offset, item, reading))
                    continue
                reading = tag_reading(entry, tag, surface, previous.get(offset, '')) if tag and item in valid else None
                if reading is not None:
                    labeled.append((offset, item, reading))
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
        common = {reading for reading, count in counts[word].items() if count >= max(MIN_ALTERNATE, MIN_ALTERNATE_SHARE * total)}
        if word not in eligible or len(common | seeded[word]) < 2:
            continue
        if word in unmeasured and total - max(counts[word].values()) < MIN_UNMEASURED_ALTERNATE:
            continue
        others = sorted((reading for reading in range(len(entry['readings'])) if reading != default),
                        key=lambda reading, word=word: (-aligned_counts[word][reading], reading))
        order = [default] + others
        ambiguous[word] = {'readings': [entry['readings'][reading] for reading in order], 'order': order,
                           'family': entry['family'], 'counts': [counts[word][reading] for reading in order]}
    labeled = []
    for example in examples:
        entry = ambiguous.get(example['word'])
        if entry is not None:
            labeled.append({'window': example['window'], 'word': example['word'], 'slot': entry['order'].index(example['reading']),
                            'source': example['source'], 'split': example['split'], 'group': example['group']})
    # The transfer windows - each ambiguous word's book windows that aren't examples (or the regression set's), the first
    # MAX_TRANSFER by a hash order
    known = held_out | {tuple(example['window']) for example in examples}
    transfer_windows = []
    for word in sorted(ambiguous):
        windows = [transfer[word][key] for key in sorted(transfer[word]) if tuple(transfer[word][key]) not in known]
        transfer_windows.extend(windows[:MAX_TRANSFER])
    validation = {'words': {word: by_word[word] for word in sorted(by_word)}, 'families': by_family}
    with open(os.path.join(args.out, 'labels.json'), 'w', encoding='utf-8') as file:
        json.dump({'words': ambiguous, 'validation': validation, 'valid': sorted(valid), 'examples': labeled,
                   'transfer': transfer_windows}, file)
    sources = collections.Counter((example['source'], example['split']) for example in labeled)
    print(f'labels: {len(ambiguous)} ambiguous words ({len(words)} candidates, {len(valid)} with a validated mapping), '
          f'{len(labeled)} examples: ' + ', '.join(f'{source} {split} {count}' for (source, split), count in sorted(sources.items())) +
          f'; {len(transfer_windows)} transfer windows')
    print(f'collocations: the corpus\'s readings disagree on {disagree[True]} of {disagree[True] + disagree[False]} of their tokens')
    print('families (aligned agreement): ' +
          ', '.join(f'{family} {counts["agree"]}/{counts["total"]}' for family, counts in sorted(by_family.items())))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.contextlabels', description='The heteronym labels')
    commands = parser.add_subparsers(dest='command', required=True)
    aligned_parser = commands.add_parser('aligned', help='the measured readings (the training speech - readings.py)')
    aligned_parser.add_argument('--out', required=True)
    aligned_parser.set_defaults(func=aligned_command)
    books_parser = commands.add_parser('books', help='fetch the recipe\'s books (pinned Project Gutenberg texts)')
    books_parser.add_argument('--books', help='the books directory (default: the recipe\'s)')
    books_parser.set_defaults(func=books_command)
    tag_parser = commands.add_parser('tag', help='part-of-speech tags (spaCy en_core_web_sm)')
    tag_parser.add_argument('--out', required=True)
    tag_parser.add_argument('--books', help='the books directory (default: the recipe\'s - picovox_train.contextlabels books)')
    tag_parser.add_argument('-j', '--jobs', type=int, default=5)
    tag_parser.set_defaults(func=tag_command)
    labels_parser = commands.add_parser('labels', help='the ambiguous words, readings, and examples')
    labels_parser.add_argument('--out', required=True)
    labels_parser.set_defaults(func=labels_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
