# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training speech - book sentences spoken by a local TTS model (Ollama's speech endpoint, POST /v1/audio/speech
{model, voice, input} - 24 kHz WAV), one engine voice for each of the package's voices, measured with our own tools
(ctc.py) and built in data.py's format. The engine, its pinned digest, the voices, and the corpus's size are
train/data/train.json's tts section; another local model is the recipe's change alone (python is
PYTHONPATH=train/src build/venv-train/bin/python -B):

    python -m picovox_train.tts generate              # the selection and the clips (Ollama - resumable)
    python -m picovox_train.tts align                 # the clips' CTC alignment (GPU - the changed clips only)
    python -m picovox_train.tts build --out DIR       # the voices' training data (data.py's format)

The corpus (downloads/tts/<model>-<digest>/ - make clean keeps it, make superclean removes it): selection-<key>.json
(the clips: id, voice key, engine voice, sex, book, text, split), wav/<voice>/<id>.wav, alignments.jsonl (the latest
record of a clip wins), and readings/ (measure.py's measures of its heteronym tokens). A clip's id is a hash of its
engine voice and text, so a bigger corpus or another selection only adds clips; the engine isn't deterministic, so a
clip is generated once. A new model (or digest) is a new corpus.

The selection - book sentences through our front end (each sentence's text is its normalized words and phrase marks,
so the engine speaks exactly the words our tokens spell; a sentence that doesn't tokenize back to its tokens is
skipped), the panel's and the samples' texts excluded: first the heteronym coverage (readings.coverage_quotas - the
gate's words, the context words, and the other measured words, each in up to its quota of sentences from the
context labels' books, the rarest word first), then sentences at random from the distillation's books to the
recipe's hours; each clip's voice and split are by a hash of its text (another selection keeps the clips of the texts
it shares).

The checks per clip (build) - the recognizer's greedy phones against the sentence's: a clip cut short (its last phones
missing) or garbled (the greedy phone error above maxPhoneError - a misread or hallucinated clip) is excluded, as is
one clipped at full scale or with a phone of no frames.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

from . import RECIPE_PATH, ROOT, read_json


TTS_DIR = os.path.join(ROOT, 'downloads', 'tts')
PANEL_TEXTS_PATH = os.path.join(ROOT, 'tools', 'panel-texts.json')

# The front end's sentence ends, and its phrase marks' punctuation (a sentence's engine text)
SENTENCE_ENDS = ('<period>', '<question>', '<wh>', '<exclamation>')
MARKS = {'<comma>': ',', '<period>': '.', '<question>': '?', '<wh>': '?', '<exclamation>': '!'}

# Paragraphs skipped - a Roman numeral (the front end reads "Chapter IV" as a word), or all capitals (headings)
ROMAN = re.compile(r'\b[IVXLC]{2,}\b')

# The candidate sentences of a paragraph (to the front end's check), and the lowercase words of a text
SENTENCE = re.compile(r'[^.!?]+[.!?]+["\'”’)\]]*')
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")

# The sentences the front end checks at once (a worker's task)
CHECK_BATCH = 256

# The clips between progress lines - generate's and build's
GENERATE_PROGRESS = 500
BUILD_PROGRESS = 2000


# Print a progress line to stderr
def log(message):
    print(message, file=sys.stderr, flush=True)


# The recipe's tts section
def load_settings(path=RECIPE_PATH):
    return read_json(path)['tts']


# The corpus directory of the recipe's engine - downloads/tts/<model>-<digest>/
def corpus_dir(settings):
    engine = settings['engine']
    return os.path.join(TTS_DIR, f'{engine["model"].replace(":", "-")}-{engine["digest"]}')


# A clip's id - a hash of its engine voice and text
def clip_id(voice, text):
    return hashlib.sha1(f'{voice}\n{text}'.encode('utf-8')).hexdigest()[:16]


# A clip's WAV path
def wav_path(directory, clip):
    return os.path.join(directory, 'wav', clip['voice'], f'{clip["id"]}.wav')


# The split of a text - 'test', 'val', or 'train' by a hash (the recipe's test and val percents)
def text_split(text, split):
    bucket = int(hashlib.sha1(f'tts/{text}'.encode('utf-8')).hexdigest(), 16) % 100
    return 'test' if bucket < split['test'] else ('val' if bucket < split['test'] + split['val'] else 'train')


# A text's voice - the recipe's voice of a hash of the text (so another selection keeps the clips of the texts it shares)
def text_voice(text, voices):
    keys = list(voices)
    return keys[int(hashlib.sha1(f'voice/{text}'.encode('utf-8')).hexdigest(), 16) % len(keys)]


# The voices in dealing order - the recipe's, the sexes alternating (so sentences dealt in turn have both)
def dealing_order(voices):
    by_sex = {}
    for key, voice in voices.items():
        by_sex.setdefault(voice['sex'], []).append(key)
    queues, order = list(by_sex.values()), []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return order


#
# The selection
#

# A sentence's engine text from its tokens and words - our front end's normalized words with its phrase marks
def engine_text(tokens, words):
    parts = []
    word_at = {word['start']: word['text'] for word in words}
    for index, token in enumerate(tokens):
        if index in word_at:
            parts.append(word_at[index])
        elif token in MARKS and parts:
            parts[-1] += MARKS[token]
    text = re.sub(r'\bi\b', 'I', ' '.join(parts))
    return text[:1].upper() + text[1:]


# The candidate sentences of a book - [(paragraph index, sentence)] (paragraphs with a Roman numeral or all capitals
# skipped)
def book_candidates(path):
    from .distill import book_paragraphs  # pylint: disable=import-outside-toplevel
    candidates = []
    for index, paragraph in enumerate(book_paragraphs(path)):
        if ROMAN.search(paragraph) or paragraph.isupper():
            continue
        for match in SENTENCE.finditer(paragraph):
            candidates.append((index, match.group(0).strip()))
    return candidates


# Check sentences with the front end (a worker task: (sentences, minWords, maxWords)) - [engine text or None]: a
# sentence of minWords to maxWords words, one sentence end (its last token), whose engine text tokenizes back to its
# tokens
def check_sentences(task):
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    sentences, min_words, max_words = task
    results = []
    for sentence in sentences:
        result = tokenize(sentence)
        tokens, words = result['tokens'], result['words']
        ends = [index for index, token in enumerate(tokens) if token in SENTENCE_ENDS]
        if ends != [len(tokens) - 1] or not min_words <= len(words) <= max_words:
            results.append(None)
            continue
        text = engine_text(tokens, words)
        results.append(text if tokenize(text)['tokens'] == tokens else None)
    return results


# Check sentences in parallel - {sentence: engine text or None}
def checked(sentences, settings, jobs):
    sentences = list(dict.fromkeys(sentences))
    batches = [sentences[start:start + CHECK_BATCH] for start in range(0, len(sentences), CHECK_BATCH)]
    texts = settings['texts']
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        results = pool.map(check_sentences, [(batch, texts['minWords'], texts['maxWords']) for batch in batches])
        return {sentence: text for batch, batch_results in zip(batches, results) for sentence, text in zip(batch, batch_results)}


# The normalized words of the texts never spoken - the panel's and the samples' (a sentence of one, or containing one, is
# skipped)
def excluded_texts():
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    texts = [item['text'] for item in read_json(PANEL_TEXTS_PATH)['texts']]
    texts += [item['text'] for item in read_json(RECIPE_PATH)['samples']['texts']]
    return [' '.join(WORD.findall(' '.join(word['text'] for word in tokenize(text)['words']).lower())) for text in texts]


# The book paths of a recipe section's texts (books.py's pinned texts)
def book_paths(section):
    from . import books  # pylint: disable=import-outside-toplevel
    books_dir = section['books'] if os.path.isabs(section['books']) else os.path.join(ROOT, section['books'])
    pins = section['pins'] if os.path.isabs(section['pins']) else os.path.join(ROOT, section['pins'])
    return books.pinned_paths(books_dir, section['texts'], pins)


# The selection's key - a hash of the recipe's settings it depends on, the books' pins, and the selection's rules
def selection_key(settings, recipe):
    keys = ('voices', 'hoursPerVoice', 'wordsPerSecond', 'seed', 'texts', 'coverage', 'split')
    with open(os.path.join(ROOT, settings['texts']['pins']), 'rb') as file:
        pins = hashlib.sha256(file.read()).hexdigest()
    basis = {'tts': {key: settings[key] for key in keys}, 'coverage': recipe['context']['labels']['texts'],
             'fill': recipe['distill']['targets']['texts'], 'gate': recipe['context']['gate']['words'],
             'pins': pins, 'rules': [SENTENCE_ENDS, MARKS, ROMAN.pattern, SENTENCE.pattern, WORD.pattern]}
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode('utf-8')).hexdigest()[:12]


# Select the corpus's sentences - [{book, text, words (lowercase), coverage}] in order: the heteronym coverage,
# then the random fill
def select_sentences(settings, recipe, jobs):
    from . import readings  # pylint: disable=import-outside-toplevel
    excluded = excluded_texts()
    quotas = readings.coverage_quotas(settings['coverage'], recipe['context']['gate']['words'])
    texts = settings['texts']
    coverage_section = {'books': texts['books'], 'pins': texts['pins'], 'texts': recipe['context']['labels']['texts']}
    fill_section = {'books': texts['books'], 'pins': texts['pins'], 'texts': recipe['distill']['targets']['texts']}

    # The coverage's candidates - each quota word's sentences (in hash order), checked by the front end
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        coverage_paths = book_paths(coverage_section)
        coverage_books = dict(zip(coverage_paths, pool.map(book_candidates, coverage_paths)))
    found = collections.defaultdict(list)
    for path in sorted(coverage_books):
        for index, sentence in coverage_books[path]:
            for word in set(WORD.findall(sentence.lower())) & quotas.keys():
                found[word].append((hashlib.sha1(f'{word}/{os.path.basename(path)}/{index}/{sentence}'.encode('utf-8')).hexdigest(),
                                    os.path.basename(path), sentence))
    wanted = []
    for word, items in found.items():
        items.sort()
        wanted.extend(items[:2 * quotas[word]])
    texts_of = checked([sentence for _, _, sentence in wanted], settings, jobs)
    log(f'tts select: {len(found)} of {len(quotas)} quota words in the coverage books, {len(texts_of)} candidate sentences checked')

    # The coverage - the rarest word first, its sentences in hash order until its quota (a sentence counts for all its words)
    chosen, seen, counts = [], set(), collections.Counter()

    # (a sentence chosen once - its words counted)
    def choose(book, text, coverage):
        key = text.lower()
        words = WORD.findall(key)
        if key in seen or any(' '.join(words) in other or other in ' '.join(words) for other in excluded):
            return False
        seen.add(key)
        counts.update(set(words) & quotas.keys())
        chosen.append({'book': book, 'text': text, 'coverage': coverage})
        return True
    for word in sorted(found, key=lambda item: (len(found[item]), item)):
        for _, book, sentence in sorted(found[word]):
            if counts[word] >= quotas[word]:
                break
            text = texts_of.get(sentence)
            if text is not None and word in WORD.findall(text.lower()):
                choose(book, text, True)
    coverage_words = sum(len(item['text'].split()) for item in chosen)

    # The fill - the distillation books' sentences at random (in order of a seeded shuffle) to the recipe's hours
    target_words = settings['hoursPerVoice'] * 3600 * settings['wordsPerSecond'] * len(settings['voices'])
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        fill_paths = book_paths(fill_section)
        fill = [(os.path.basename(path), sentence) for path, items in zip(fill_paths, pool.map(book_candidates, fill_paths))
                for _, sentence in items]
    random.Random(settings['seed']).shuffle(fill)
    total, position = coverage_words, 0
    while total < target_words and position < len(fill):
        group = fill[position:position + 8 * CHECK_BATCH * max(1, jobs or os.cpu_count())]
        position += len(group)
        texts_of = checked([sentence for _, sentence in group], settings, jobs)
        for book, sentence in group:
            text = texts_of[sentence]
            if total < target_words and text is not None and choose(book, text, False):
                total += len(text.split())
    short = {word: f'{counts[word]}/{quota}' for word, quota in quotas.items() if counts[word] < quota and word in found}
    log(f'tts select: {sum(item["coverage"] for item in chosen)} coverage sentences ({coverage_words} words), '
        f'{sum(not item["coverage"] for item in chosen)} fill sentences, {total / settings["wordsPerSecond"] / 3600:.1f} hours '
        f'estimated; {len(short)} quota words short of their quota, {len(quotas) - len(found)} not in the books')
    return chosen


# The selection - the corpus's selection-<key>.json, made if missing - [{id, key, voice, sex, book, text, split, coverage}]
def selection(settings, jobs=None, recipe_path=RECIPE_PATH):
    recipe = read_json(recipe_path)
    directory = corpus_dir(settings)
    path = os.path.join(directory, f'selection-{selection_key(settings, recipe)}.json')
    if os.path.exists(path):
        return read_json(path)['clips']
    sentences = select_sentences(settings, recipe, jobs)
    clips = []
    for item in sentences:
        key = text_voice(item['text'], settings['voices'])
        voice = settings['voices'][key]
        clips.append({'id': clip_id(voice['voice'], item['text']), 'key': key, 'voice': voice['voice'], 'sex': voice['sex'],
                      'book': item['book'], 'text': item['text'], 'split': text_split(item['text'], settings['split']),
                      'coverage': item['coverage']})
    os.makedirs(directory, exist_ok=True)
    with open(path + '.part', 'w', encoding='utf-8') as file:
        json.dump({'engine': settings['engine'], 'clips': clips}, file, indent=0)
    os.replace(path + '.part', path)
    return clips


#
# The engine - Ollama's API
#

class EngineError(Exception):
    """The engine can't be used - not running, the model missing or another digest, or a request refused"""


# A request to the engine - the response body (bytes); a refused connection or an HTTP error is an EngineError
def engine_request(engine, path, body=None, timeout=60):
    url = engine['url'].rstrip('/') + path
    data = None if body is None else json.dumps(body).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace').strip()
        error.close()
        raise EngineError(f'{url}: HTTP {error.code} {detail}') from error
    except urllib.error.URLError as error:
        raise EngineError(f'Ollama isn\'t answering at {engine["url"]} ({error.reason}) - start it (the app, or ollama serve), '
                          'then run make train again') from error


# The model's digest (the first 12 hex digits) in Ollama's model list, or None if it isn't there
def model_digest(engine):
    name = engine['model'] if ':' in engine['model'] else f'{engine["model"]}:latest'
    models = json.loads(engine_request(engine, '/api/tags'))['models']
    return next((item['digest'][:12] for item in models if item['name'] == name), None)


# Check the engine's model - pulled if missing, its digest the recipe's pin
def check_engine(engine):
    digest = model_digest(engine)
    if digest is None:
        log(f'tts: pulling {engine["model"]} (ollama pull)')
        engine_request(engine, '/api/pull', {'model': engine['model'], 'stream': False}, timeout=3600)
        digest = model_digest(engine)
    if digest != engine['digest']:
        raise EngineError(f'{engine["model"]} is digest {digest}, not the recipe\'s {engine["digest"]} - a new model is a new '
                          'corpus: set train/data/train.json tts.engine.digest')


# Speak a text in a voice - the WAV bytes (a transient failure retried; an HTTP error - a refused voice or option - isn't)
def speak(engine, voice, text, attempts=4):
    body = {'model': engine['model'], 'voice': voice, 'input': text}
    for attempt in range(attempts):
        try:
            return engine_request(engine, '/v1/audio/speech', body, timeout=300)
        except EngineError as error:
            if 'HTTP 4' in str(error) or attempt == attempts - 1:
                raise
            log(f'  retry ({error})')
            time.sleep(2 ** attempt)
    return None


# Unload the model (keep_alive 0) - its memory back for the training; best effort
def unload(engine):
    try:
        engine_request(engine, '/api/generate', {'model': engine['model'], 'keep_alive': 0})
    except EngineError as error:
        log(f'tts: the model wasn\'t unloaded ({error})')


def generate_command(args):
    """The selection and its clips - every clip's WAV (only the missing ones generated)"""
    from .wav import read_wav  # pylint: disable=import-outside-toplevel
    settings = load_settings(args.recipe)
    engine = settings['engine']
    directory = corpus_dir(settings)
    clips = selection(settings, args.jobs, args.recipe)
    missing = [clip for clip in clips if not os.path.exists(wav_path(directory, clip))]
    log(f'tts generate: {len(clips)} clips in {directory}, {len(missing)} to generate')
    if not missing:
        return 0
    check_engine(engine)
    for voice in {clip['voice'] for clip in missing}:
        os.makedirs(os.path.join(directory, 'wav', voice), exist_ok=True)

    # (one clip - its WAV checked: 16-bit mono)
    def one(clip):
        path = wav_path(directory, clip)
        audio = speak(engine, clip['voice'], clip['text'])
        with open(path + '.part', 'wb') as file:
            file.write(audio)
        samples, _ = read_wav(path + '.part')
        if samples.shape[1] != 1 or len(samples) == 0:
            raise EngineError(f'{clip["id"]}: not a mono WAV')
        os.replace(path + '.part', path)
        return len(samples)
    start, total = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(settings['workers']) as pool:
        for number, count in enumerate(pool.map(one, missing), 1):
            total += count
            if number % GENERATE_PROGRESS == 0 or number == len(missing):
                log(f'  {number}/{len(missing)} clips, {total / 24000 / 3600:.2f} hours, {time.time() - start:.0f} s')
    unload(engine)
    return 0


#
# The alignment
#

# The aligner's identity - a hash of its settings (a new aligner realigns every clip)
def aligner_key():
    from . import ctc  # pylint: disable=import-outside-toplevel
    basis = [ctc.CTC_REPO, ctc.CTC_SHIFT, ctc.CTC_SPLIT, ctc.EDGE_EXTEND, ctc.IPA_TOKENS]
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode('utf-8')).hexdigest()[:12]


# Texts through the front end (a worker task: [(text, readings {position: phones})]) - [{tokens, words (the words'
# texts)}]
def tokenize_texts(items):
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    results = []
    for text, readings in items:
        result = tokenize(text, readings=readings or None)
        results.append({'tokens': result['tokens'], 'words': [word['text'] for word in result['words']]})
    return results


# The clips through the front end, in parallel - {id: {tokens, words}}, with the measured readings that differ from
# ours (overrides - readings.py; none for the front end's own)
def clip_fronts(clips, overrides, jobs):
    items = [(clip['text'], {int(position): phones.split() for position, phones in overrides.get(clip['id'], {}).items()})
             for clip in clips]
    batches = [items[start:start + CHECK_BATCH] for start in range(0, len(items), CHECK_BATCH)]
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        results = [result for batch in pool.map(tokenize_texts, batches) for result in batch]
    return {clip['id']: result for clip, result in zip(clips, results)}


# Tokens' alignment keys - the phones without stress (all the alignment sees)
def phone_keys(tokens):
    from . import ctc  # pylint: disable=import-outside-toplevel
    return ' '.join(key for _, key in ctc.phone_positions(tokens))


# The corpus's alignments - {id: record} (the latest of a clip's records)
def read_alignments(directory):
    path = os.path.join(directory, 'alignments.jsonl')
    records = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                records[record['id']] = record
    return records


def align_command(args):
    """The clips' CTC alignment - alignments.jsonl (the clips whose phones or aligner changed, appended)"""
    from . import ctc, readings  # pylint: disable=import-outside-toplevel
    settings = load_settings(args.recipe)
    directory = corpus_dir(settings)
    clips = [clip for clip in selection(settings, args.jobs, args.recipe) if os.path.exists(wav_path(directory, clip))]
    overrides = readings.overrides(directory)
    records = read_alignments(directory)
    key = aligner_key()
    fronts = clip_fronts(clips, overrides, args.jobs)
    todo = []
    for clip in clips:
        tokens = fronts[clip['id']]['tokens']
        record = records.get(clip['id'])
        if record is None or record['aligner'] != key or record['keys'] != phone_keys(tokens):
            todo.append((clip, tokens))
    log(f'tts align: {len(clips)} clips, {len(todo)} to align')
    if not todo:
        return 0
    model, vocab, device = ctc.load_ctc(args.device)
    start, failures = time.time(), 0
    with open(os.path.join(directory, 'alignments.jsonl'), 'a', encoding='utf-8') as file:
        for offset in range(0, len(todo), args.batch):
            group = todo[offset:offset + args.batch]
            samples_list = [ctc.read_16k(wav_path(directory, clip)) for clip, _ in group]
            posteriors = ctc.ctc_log_posteriors(model, device, samples_list)
            for (clip, tokens), samples, log_posteriors in zip(group, samples_list, posteriors):
                record = {'id': clip['id'], 'aligner': key, 'keys': phone_keys(tokens), 'tokens': tokens,
                          'seconds': round(len(samples) / ctc.CTC_SAMPLE_RATE, 4), 'labels': None}
                result = ctc.ctc_labels(tokens, log_posteriors, vocab, samples) if len(samples) > 0.3 * ctc.CTC_SAMPLE_RATE else None
                if result is None:
                    failures += 1
                else:
                    labels, confidences, score = result
                    record.update({'labels': [[label, round(begin, 4), round(end, 4)] for label, begin, end in labels],
                                   'confidences': [round(value, 4) for value in confidences], 'score': round(score, 4),
                                   'greedy': ctc.greedy_decode(log_posteriors, vocab)})
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
            file.flush()
            log(f'  {offset + len(group)}/{len(todo)} clips, {time.time() - start:.0f} s')
    log(f'tts align: aligned {len(todo) - failures} of {len(todo)} clips ({failures} failed) in {time.time() - start:.0f} s')
    return 0


#
# The build - the voices' training data in data.py's format
#

# A clip's checks - its exclusion reasons (no alignment, cut short, garbled) and its greedy phone error
def clip_checks(record, checks):
    from . import ctc  # pylint: disable=import-outside-toplevel
    if not record or not record['labels']:
        return ['no alignment'], None
    reasons = []
    error, tail = ctc.greedy_phone_error(record['greedy'], record['tokens'])
    last = record['labels'][-1]
    trailing = last[2] - last[1] if last[0] == 'pau' else 0.0
    if tail >= checks['truncatedAlways'] or (tail >= checks['truncatedTail'] and trailing < checks['truncatedSilence']):
        reasons.append(f'cut short ({tail} final phones missing)')
    if error > checks['maxPhoneError']:
        reasons.append(f'garbled (greedy phone error {error:.2f} > {checks["maxPhoneError"]})')
    return reasons, error


# Process one clip at its own rate (24 kHz) - durations from its labels, the edge silences cut to data.py's
# MAX_EDGE_SILENCE, its audio and frame tracks (a worker task)
def process_clip(task):
    import numpy as np  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    from . import align, ctc, data  # pylint: disable=import-outside-toplevel
    from .wav import read_wav  # pylint: disable=import-outside-toplevel
    path, tokens, labels, mean_seconds, settings = task
    samples, rate = read_wav(path)
    samples = samples.mean(axis=1)
    if rate != data.SAMPLE_RATE:
        samples = soxr.resample(samples, rate, data.SAMPLE_RATE, quality='VHQ')
    samples = np.ascontiguousarray(samples)
    seconds, _ = ctc.label_durations(tokens, [tuple(label) for label in labels], samples, data.SAMPLE_RATE, mean_seconds)
    wav_seconds = len(samples) / data.SAMPLE_RATE
    speech_start, speech_end = seconds[0], wav_seconds - seconds[-1]
    cut_start = max(0.0, speech_start - data.MAX_EDGE_SILENCE)
    cut_end = min(wav_seconds, speech_end + data.MAX_EDGE_SILENCE)
    seconds[0] = speech_start - cut_start
    seconds[-1] = cut_end - speech_end
    frames = align.durations_to_frames(seconds, data.FRAME_SECONDS, tokens)
    frame_count = sum(frames)
    start = int(round(cut_start * data.SAMPLE_RATE))
    audio = samples[start:start + frame_count * data.HOP]
    audio = np.pad(audio, (0, frame_count * data.HOP - len(audio)))
    return {'seconds': seconds, 'frames': frames, 'cutStart': cut_start, 'tracks': data.frame_tracks(audio, frame_count, *settings),
            'audio': np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16),
            'clipped': float(np.mean(np.abs(samples) >= 0.999))}


# A voice's pitch range (Hirst 2011's two passes - floor 0.75 x Q1, ceiling 1.5 x Q3 of the first pass) and median F0
def pitch_range(values):
    import numpy as np  # pylint: disable=import-outside-toplevel
    return (round(max(50.0, 0.75 * float(np.percentile(values, 25))), 1), round(min(900.0, 1.5 * float(np.percentile(values, 75))), 1),
            round(float(np.median(values)), 1))


def build_command(args):
    """The voices' training data - <out>/ in data.py's format (index.json, the arrays, alignment.jsonl)"""
    import multiprocessing  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    from . import align, ctc, data, readings  # pylint: disable=import-outside-toplevel
    from .frontend import BOUNDARIES, TOKEN_IDS, vocabulary  # pylint: disable=import-outside-toplevel
    start_time = time.time()
    settings = load_settings(args.recipe)
    directory = corpus_dir(settings)
    clips = [clip for clip in selection(settings, args.jobs, args.recipe) if os.path.exists(wav_path(directory, clip))]
    if args.limit:
        clips = clips[::max(1, len(clips) // args.limit)][:args.limit]
    records = read_alignments(directory)
    overrides = readings.overrides(directory)
    os.makedirs(args.out, exist_ok=True)

    # The checks, and the clips' tokens against their alignments (the front end changed: align again)
    checks = {}
    fronts = clip_fronts(clips, overrides, args.jobs)
    for clip in clips:
        record = records.get(clip['id'])
        if record and record['labels'] and phone_keys(fronts[clip['id']]['tokens']) != record['keys']:
            raise ValueError(f'{clip["id"]}: the front end\'s phones differ from its alignment\'s - run tts align again')
        checks[clip['id']] = clip_checks(record, settings['checks'])

    # Typical phone durations (align.py splits a label block by them), and each voice's pitch range
    totals = collections.defaultdict(lambda: [0.0, 0])
    for clip in clips:
        for label, begin, end in (records.get(clip['id']) or {}).get('labels') or []:
            totals[align.label_key(label)][0] += end - begin
            totals[align.label_key(label)][1] += 1
    mean_seconds = {key: total / count for key, (total, count) in totals.items()}
    with multiprocessing.Pool(args.jobs) as pool:
        first_f0 = pool.map(ctc.first_pass_f0, [wav_path(directory, clip) for clip in clips], chunksize=8)
    pooled = collections.defaultdict(list)
    for clip, values in zip(clips, first_f0):
        pooled[clip['key']].append(values)
    ranges = {key: pitch_range(np.concatenate(arrays)) for key, arrays in pooled.items()}
    analysis = settings['analysis']

    # The clips, streamed into raw arrays (then .npy)
    tasks, kept = [], []
    for clip in clips:
        record = records.get(clip['id'])
        if record and record['labels']:
            floor, ceiling, _ = ranges[clip['key']]
            tasks.append((wav_path(directory, clip), record['tokens'], record['labels'], mean_seconds,
                          (floor, ceiling, analysis['formantCeilings'][clip['sex']])))
            kept.append(clip)
    names = ('tokens', 'durations', 'seconds', 'audio', 'frames')
    dtypes = {'tokens': np.int16, 'durations': np.int16, 'seconds': np.float32, 'audio': np.int16, 'frames': np.float32}
    raw_files = {name: open(os.path.join(args.out, name + '.raw'), 'wb') for name in names}  # pylint: disable=consider-using-with
    lengths = dict.fromkeys(names, 0)
    entries = []
    token_offset = frame_offset = 0
    with open(os.path.join(args.out, 'alignment.jsonl'), 'w', encoding='utf-8') as alignment_file, \
            multiprocessing.Pool(args.jobs) as pool:
        for count, (clip, result) in enumerate(zip(kept, pool.imap(process_clip, tasks, chunksize=4)), 1):
            if count % BUILD_PROGRESS == 0:
                log(f'  {count}/{len(tasks)} clips, {frame_offset / 100 / 3600:.1f} h, {time.time() - start_time:.0f} s')
            record = records[clip['id']]
            reasons, error = checks[clip['id']]
            reasons = list(reasons)
            if result['clipped'] > settings['checks']['maxClipped']:
                reasons.append(f'clipped ({100 * result["clipped"]:.2f}% of samples at full scale)')
            if any(frames == 0 for token, frames in zip(record['tokens'], result['frames']) if token not in BOUNDARIES):
                reasons.append('a phone with no frames')
            frame_count = sum(result['frames'])
            entries.append({
                'id': f'{clip["key"]}/{clip["id"]}', 'speaker': clip['key'], 'sentence': clip['id'], 'split': clip['split'],
                'text': clip['text'], 'book': clip['book'], 'include': not reasons, 'reason': '; '.join(reasons),
                'tokenOffset': token_offset, 'tokenCount': len(record['tokens']), 'frameOffset': frame_offset,
                'frameCount': frame_count, 'sampleOffset': frame_offset * data.HOP, 'sampleCount': frame_count * data.HOP,
                'trimStart': round(result['cutStart'], 5), 'phoneError': round(error, 3), 'clipped': round(result['clipped'], 5),
                'readings': overrides.get(clip['id'], {})
            })
            arrays = {'tokens': np.asarray([TOKEN_IDS[token] for token in record['tokens']], dtype=np.int16),
                      'durations': np.asarray(result['frames'], dtype=np.int16),
                      'seconds': np.asarray(result['seconds'], dtype=np.float32), 'audio': result['audio'],
                      'frames': result['tracks'].astype(np.float32)}
            for name in names:
                raw_files[name].write(np.ascontiguousarray(arrays[name], dtype=dtypes[name]).tobytes())
                lengths[name] += len(arrays[name])
            token_offset += len(record['tokens'])
            frame_offset += frame_count
            alignment_file.write(json.dumps({'id': entries[-1]['id'], 'tokens': ' '.join(record['tokens']), 'frames': result['frames'],
                                             'greedy': record['greedy'], 'confidences': record['confidences']}, ensure_ascii=False) + '\n')
    for name in names:
        raw_files[name].close()
        shape = (lengths[name], len(data.CHANNELS)) if name == 'frames' else (lengths[name],)
        data.raw_to_npy(os.path.join(args.out, name + '.raw'), os.path.join(args.out, name + '.npy'), dtypes[name], shape)

    # The index - the speakers keyed by voice key, their rows the recipe's order
    engine = settings['engine']
    speakers = {}
    for row, (key, voice) in enumerate(settings['voices'].items()):
        if key in ranges:
            floor, ceiling, median = ranges[key]
            speakers[key] = {'index': row, 'sex': voice['sex'], 'corpus': 'tts',
                             'engine': {'model': engine['model'], 'digest': engine['digest'], 'voice': voice['voice']},
                             'sourceSampleRate': data.SAMPLE_RATE, 'bandwidth': analysis['bandwidth'], 'pitchFloor': floor,
                             'pitchCeiling': ceiling, 'formantCeiling': analysis['formantCeilings'][voice['sex']], 'medianF0': median}
    index = {
        'version': data.FORMAT_VERSION, 'corpus': f'tts ({engine["model"]} {engine["digest"]})', 'license': engine['license'],
        'sampleRate': data.SAMPLE_RATE, 'hop': data.HOP, 'frameRate': data.SAMPLE_RATE / data.HOP, 'channels': data.CHANNELS,
        'channelNotes': data.CHANNEL_NOTES, 'vocab': vocabulary(), 'maxEdgeSilence': data.MAX_EDGE_SILENCE,
        'alignment': {'method': f'CTC forced alignment with {ctc.CTC_REPO} (picovox_train/ctc.py)', 'aligner': aligner_key()},
        'checks': settings['checks'], 'speakers': speakers, 'utterances': entries
    }
    with open(os.path.join(args.out, 'index.json'), 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=1)
        file.write('\n')
    included = [entry for entry in entries if entry['include']]
    reasons = collections.Counter((entry['speaker'], reason.split(' (')[0]) for entry in entries
                                  for reason in entry['reason'].split('; ') if reason)
    print(f'Built {args.out}: {len(entries)} clips ({len(included)} included), {frame_offset / 100 / 3600:.2f} hours '
          f'({sum(entry["frameCount"] for entry in included) / 100 / 3600:.2f} included), {time.time() - start_time:.0f} s')
    for key, speaker in speakers.items():
        excluded = {reason: count for (voice, reason), count in sorted(reasons.items()) if voice == key}
        print(f'  {key} ({speaker["engine"]["voice"]}): F0 {speaker["pitchFloor"]}-{speaker["pitchCeiling"]} Hz, median '
              f'{speaker["medianF0"]}; {sum(entry["speaker"] == key for entry in included)} included, excluded {excluded}')
    return 0


# The command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog='picovox_train.tts', description='The training speech - a local TTS model\'s')
    parser.add_argument('--recipe', default=RECIPE_PATH, help='the recipe (default train/data/train.json)')
    commands = parser.add_subparsers(dest='command', required=True)
    generate_parser = commands.add_parser('generate', help='the selection and its clips (Ollama - resumable)')
    generate_parser.add_argument('-j', '--jobs', type=int, default=None, help='the selection\'s worker processes (default: all cores)')
    generate_parser.set_defaults(func=generate_command)
    align_parser = commands.add_parser('align', help='the clips\' CTC alignment (the changed clips only)')
    align_parser.add_argument('--batch', type=int, default=256, help='clips at once')
    align_parser.add_argument('--device', default=None, help='torch device (default mps if available)')
    align_parser.add_argument('-j', '--jobs', type=int, default=None, help='the selection\'s worker processes')
    align_parser.set_defaults(func=align_command)
    build_parser = commands.add_parser('build', help='the voices\' training data (data.py\'s format)')
    build_parser.add_argument('--out', required=True, help='the data directory')
    build_parser.add_argument('--limit', type=int, help='a sample of this many clips (a quick test)')
    build_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    build_parser.set_defaults(func=build_command)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except EngineError as error:
        log(f'tts: {error}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
