# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The multi-speaker training data - American audiobook readers: LibriTTS-R (train-clean-100 and dev-clean) speakers
classified as US English, and the labeled CMU ARCTIC US speakers (slt, bdl, clb, rms), featurized exactly as
data.py featurizes ARCTIC (the same tokens, alignment, format, and auxiliary tracks), into one corpus directory that
data.py's ArcticData loads. The pool is the committed speaker table, data/nn/speakers.json (its accents and analysis
settings were measured by the former accent classification, settings, and pool commands - see git history).

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.multi build -j 8     # build/nn/data/multi/ (6 minutes, 14 cores)
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.multi verify-tokens  # lib/tokens.js == frontend.py

The corpus (build/nn/data/multi/) has data.py's format (index.json, tokens.npy, durations.npy, seconds.npy,
audio.npy, frames.npy, alignment.jsonl): load it with ArcticData(os.path.join(DATA_DIR, 'multi')). build --out DIR
builds it elsewhere, with its speaker table and split (speakers.json, multi-split.json) beside it instead of in
data/nn/ (verify-tokens takes the same --out); build --pool TABLE builds another speaker table's pool. Its speakers are
keyed "arctic-<name>" and "libritts-<id>"; each speaker's index.json entry has data.py's analysis settings (sex,
pitchFloor, pitchCeiling, formantCeiling, sourceSampleRate, bandwidth) plus its index, which is its id in
data/nn/speakers.json.

The pieces:

- Speakers - a LibriTTS-R speaker's sex is from the corpus's speakers.tsv; its pitch range is Hirst's two-pass range
  (0.75 x the first quartile to 1.5 x the third quartile of a 60-600 Hz first pass), and its formant ceiling Escudero's
  (the ceiling with the least within-vowel variance of log F1 and log F2 over the stressed vowels' midpoints, from the
  MFA labels), both over up to 60 utterances. The ARCTIC speakers keep their data/targets/ settings (measured by the
  former tools/targets.py - see git history).
- The accent filter - a LibriTTS-R speaker is kept when the CommonAccent classifier's top accent is "us" with a mean
  posterior of at least US_THRESHOLD over its utterances; the ARCTIC speakers are US English by the
  corpus's documentation (and the classifier agrees). The rhoticity measure of data.py's survey (the median F3 of
  stressed ER over the other stressed vowels') is recorded beside it.
- Text and labels - LibriTTS-R's normalized text (numbers, abbreviations, and symbols already spelled out as words)
  through our front end (every word pronounced by the app's G2P network); LibriTTSLabel's Montreal Forced Aligner
  phone labels for the times. The alignment, the edge
  silence trim, and the exclusion rules (phone match below MIN_MATCH, a word below MIN_WORD_MATCH, a phone with no frames)
  are data.py's, except that LibriTTS-R's word rule skips function words and abbreviation expansions
  (WORD_MATCH_EXEMPT - the aligner's dictionary reduces them); utterances with no MFA labels are left out.
- Splits - per speaker: an ARCTIC utterance has data.py's sentence split (sentence_split - the same
  held-out sentences for every speaker); a LibriTTS-R utterance is "test" (2%) or "val" (3%) by a hash of its utterance id, else
  "train". data/nn/multi-split.json lists the held-out utterance ids per speaker.
- data/nn/speakers.json - the speaker id table: id (the training index), key, source, corpus speaker, reader, sex,
  accent (top, US posterior, votes), hours (included, per split), median F0, analysis settings, and match rates.
"""

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import sys
import time

import numpy as np

from . import ROOT, data
from . import readings
from .data import (CORPORA_DIR, DATA_DIR, FORMAT_VERSION, HOP, MAX_EDGE_SILENCE, MIN_MATCH, MIN_WORD_MATCH, SAMPLE_RATE,
                   CHANNELS, CHANNEL_NOTES, LABEL_DIR, SUBSETS, ArcticData, match_rate)
from .frontend import (ABBREVIATIONS, BOUNDARIES, FUNCTION_WORDS, pronounce_texts, set_pronunciations, tokenize,
                       vocabulary)


OUT_DIR = os.path.join(DATA_DIR, 'multi')
SPEAKERS_PATH = os.path.join(ROOT, 'data', 'nn', 'speakers.json')
SPLIT_PATH = os.path.join(ROOT, 'data', 'nn', 'multi-split.json')
LIBRITTS_DIR = os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R')

# The labeled ARCTIC US speakers in the pool, in id order (slt and bdl first, as in the single-speaker data)
ARCTIC_SPEAKERS = {'slt': 'female', 'bdl': 'male', 'clb': 'female', 'rms': 'male'}

# Corrections of speakers.tsv's sex - speakers whose label contradicts their name and voice (median F0)
SEX_OVERRIDES = {'7976': 'female'}     # JenniferRutters, labeled M, median F0 about 200 Hz

# The accent filter's least mean US posterior of a kept speaker (recorded in the speaker table - the pool was chosen by the
# former accent classification, settings, and pool commands, in git history)
US_THRESHOLD = 0.5


# The worst-word rule's exempt words - data.py's rule is for our letter-to-sound guesses at unknown words; in
# LibriTTS-R the Montreal Forced Aligner's dictionary picks reduced or variant pronunciations of function words ("for"
# F ER, "are" ER, "our" AA R, "the" DH IY) and has no "mrs" (its <unk>), which the rule would otherwise exclude by the
# thousand although the words' timing is right; the phone-match rule still applies to every phone
WORD_MATCH_EXEMPT = FUNCTION_WORDS | set(ABBREVIATIONS.values())

# The LibriTTS-R held-out shares (percent) - test and validation
TEST_PERCENT = 2
VAL_PERCENT = 3


#
# Corpus reading
#

# The LibriTTS-R speakers' metadata from speakers.tsv - {speaker: {sex, subset, reader}}
def libritts_metadata():
    metadata = {}
    with open(os.path.join(LIBRITTS_DIR, 'speakers.tsv'), encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if fields[0].isdigit():
                metadata[fields[0]] = {'sex': {'F': 'female', 'M': 'male'}.get(fields[1]),
                                       'subset': fields[2] if len(fields) > 2 else None,
                                       'reader': fields[3] if len(fields) > 3 else ''}
    return metadata


# The utterances of a LibriTTS-R speaker - [{speaker (key), sentence (utterance id), wav, lab, labelFormat, text}]
def libritts_utterances(subset, speaker):
    utterances = []
    base = os.path.join(LIBRITTS_DIR, subset, speaker)
    for chapter in sorted(os.listdir(base)):
        chapter_dir = os.path.join(base, chapter)
        if not os.path.isdir(chapter_dir):
            continue
        for name in sorted(os.listdir(chapter_dir)):
            if not name.endswith('.wav'):
                continue
            utterance_id = name[:-len('.wav')]
            text_path = os.path.join(chapter_dir, utterance_id + '.normalized.txt')
            if not os.path.exists(text_path):
                continue
            with open(text_path, encoding='utf-8') as file:
                text = file.read().strip()
            utterances.append({'speaker': f'libritts-{speaker}', 'sentence': utterance_id,
                               'wav': os.path.join(chapter_dir, name),
                               'lab': os.path.join(LABEL_DIR, subset, speaker, chapter, utterance_id + '.lab'),
                               'labelFormat': 'mfa', 'text': text})
    return utterances


# The candidate speakers - {key: {key, corpus, source, speaker, reader, sex, sourceSampleRate, bandwidth, utterances}}
def candidate_speakers():
    speakers = {}
    for name, sex in ARCTIC_SPEAKERS.items():
        utterances = [{**utterance, 'speaker': f'arctic-{name}'} for utterance in data.arctic_utterances(name)
                      if utterance['text'] is not None]
        speakers[f'arctic-{name}'] = {'key': f'arctic-{name}', 'corpus': 'arctic', 'source': 'CMU ARCTIC 0.95',
                                      'speaker': name, 'reader': None, 'sex': sex, 'sourceSampleRate': 16000,
                                      'bandwidth': 8000, 'utterances': utterances}
    metadata = libritts_metadata()
    for subset in SUBSETS:
        base = os.path.join(LIBRITTS_DIR, subset)
        if not os.path.isdir(base):
            continue
        for speaker in sorted((item for item in os.listdir(base) if item.isdigit()), key=int):
            info = metadata.get(speaker, {})
            speakers[f'libritts-{speaker}'] = {
                'key': f'libritts-{speaker}', 'corpus': f'libritts-{subset}', 'source': f'LibriTTS-R {subset}',
                'speaker': speaker, 'reader': info.get('reader'), 'sex': SEX_OVERRIDES.get(speaker, info.get('sex')),
                'sexCorrected': speaker in SEX_OVERRIDES, 'sourceSampleRate': 24000,
                'bandwidth': 12000, 'utterances': libritts_utterances(subset, speaker)
            }
    return speakers


# The split of a pool utterance - ARCTIC by sentence (data.py), LibriTTS-R by a hash of the utterance id
def utterance_split(corpus, sentence):
    if corpus == 'arctic':
        return data.sentence_split(sentence)
    bucket = int(hashlib.sha1(sentence.encode('utf-8')).hexdigest(), 16) % 100
    return 'test' if bucket < TEST_PERCENT else ('val' if bucket < TEST_PERCENT + VAL_PERCENT else 'train')


#
# The pool
#

# The pool of a speaker table (data/nn/speakers.json, which records each kept speaker's accent and analysis settings) -
# (the kept speakers in id order, their rows)
def table_pool(path):
    table = sorted(load_json(path, None)['speakers'], key=lambda row: row['id'])
    rows = []
    for row in table:
        accent = row['accent']
        rows.append({'key': row['key'], 'keep': True,
                     'accent': {'top': accent['top'], 'posterior': {'us': accent['us'], 'canada': accent['canada']},
                                'votes': accent['votes']} if accent else None,
                     'settings': {'pitchFloor': row['pitchFloor'], 'pitchCeiling': row['pitchCeiling'],
                                  'formantCeiling': row['formantCeiling'], 'erF3Ratio': row['erF3Ratio']}})
    return [row['key'] for row in rows], rows


#
# Build
#

# The paths of a corpus directory's speaker table and split - data/nn/'s for the repository's default directory
# (build/nn/data/multi), else beside it - so a build elsewhere (make train's, with NN_DATA_DIR) never rewrites the
# committed files
def table_paths(out_dir):
    if os.path.abspath(out_dir) == os.path.abspath(os.path.join(ROOT, 'build', 'nn', 'data', 'multi')):
        return SPEAKERS_PATH, SPLIT_PATH
    return os.path.join(out_dir, 'speakers.json'), os.path.join(out_dir, 'multi-split.json')


# Build build/nn/data/multi/ (or --out), data/nn/speakers.json, and data/nn/multi-split.json
def build_command(args):
    start_time = time.time()
    out_dir = args.out or OUT_DIR
    speakers = candidate_speakers()
    kept, rows = table_pool(args.pool)
    if args.speakers:
        kept = [key for key in kept if key in args.speakers]
    row_of = {row['key']: row for row in rows}

    # The tasks - per speaker, the utterances with labels, the typical label durations, and the analysis settings
    tasks = []
    missing_labels = collections.Counter()
    for key in kept:
        speaker = speakers[key]
        utterances = [utterance for utterance in speaker['utterances'] if os.path.exists(utterance['lab'])]
        missing_labels[key] = len(speaker['utterances']) - len(utterances)
        if args.limit:
            utterances = utterances[:args.limit]
        mean_seconds = data.mean_label_seconds(utterances)
        speaker_settings = row_of[key]['settings']
        analysis = (speaker_settings['pitchFloor'], speaker_settings['pitchCeiling'], speaker_settings['formantCeiling'])
        tasks.extend((utterance, mean_seconds, analysis) for utterance in utterances)
    print(f'{len(kept)} speakers, {len(tasks)} utterances ({sum(missing_labels.values())} without MFA labels left out)', flush=True)

    # Pronounce the texts' words once (the G2P network), for every worker
    pronunciations = pronounce_texts([utterance['text'] for utterance, _, _ in tasks], args.jobs)
    # (this process too - its per-utterance checks tokenize the texts, about 0.1 s each uncached)
    set_pronunciations(pronunciations)
    print(f'{len(pronunciations)} distinct words pronounced, {time.time() - start_time:.0f} s', flush=True)

    # Heteronym readings - the aligner's reading where it differs from the front end's (readings.py)
    reading_counts = readings.attach_readings(tasks)
    reading_check = {'tokens': 0, 'disagree': 0, 'szOnly': 0}
    print(f'heteronym readings: {reading_counts["tokens"]} aligned tokens in {reading_counts["utterances"]} LibriTTS-R utterances, '
          f'{reading_counts["overrides"]} overrides, {time.time() - start_time:.0f} s', flush=True)

    # Process in parallel, streaming the arrays to raw files
    os.makedirs(out_dir, exist_ok=True)
    names = ['tokens', 'durations', 'seconds', 'audio', 'frames']
    raw_files = {name: open(os.path.join(out_dir, name + '.raw'), 'wb') for name in names}  # pylint: disable=consider-using-with
    dtypes = {'tokens': np.int16, 'durations': np.int16, 'seconds': np.float32, 'audio': np.int16, 'frames': np.float32}
    lengths = collections.Counter()
    entries = []
    token_offset = frame_offset = 0
    edits_by_speaker = collections.defaultdict(collections.Counter)
    failures = []
    with open(os.path.join(out_dir, 'alignment.jsonl'), 'w', encoding='utf-8') as alignment_file, \
            multiprocessing.Pool(args.jobs, initializer=set_pronunciations, initargs=(pronunciations,)) as pool:
        for count, result in enumerate(pool.imap(safe_process_utterance, tasks, chunksize=8), 1):
            if count % 2000 == 0:
                print(f'  {count}/{len(tasks)} utterances, {frame_offset / 100 / 3600:.1f} h, {time.time() - start_time:.0f} s', flush=True)
            if 'error' in result:
                failures.append((result['utterance']['speaker'], result['utterance']['sentence'], result['error']))
                continue
            utterance, report = result['utterance'], result['report']
            rate = match_rate(report)
            reasons = []
            if rate < MIN_MATCH:
                reasons.append(f'phone match {rate:.3f} < {MIN_MATCH}')
            corpus = speakers[utterance['speaker']]['corpus']
            worst_word, worst_word_match = report['worstWord'], report['worstWordMatch']
            if corpus != 'arctic':
                worst_word, worst_word_match = worst_content_word(utterance['text'], report['status'], utterance.get('readings'))
            if worst_word_match < MIN_WORD_MATCH:
                reasons.append(f'word "{worst_word}" match {worst_word_match:.2f} < {MIN_WORD_MATCH}')
            if any(frames == 0 for token, frames in zip(result['tokens'], result['frames']) if token not in BOUNDARIES):
                reasons.append('a phone with no frames')
            frame_count = sum(result['frames'])
            entry = {
                'id': f'{utterance["speaker"]}/{utterance["sentence"]}',
                'speaker': utterance['speaker'],
                'sentence': utterance['sentence'],
                'split': utterance_split(corpus, utterance['sentence']),
                'text': utterance['text'],
                'include': not reasons,
                'reason': '; '.join(reasons),
                'tokenOffset': token_offset,
                'tokenCount': len(result['tokens']),
                'frameOffset': frame_offset,
                'frameCount': frame_count,
                'sampleOffset': frame_offset * HOP,
                'sampleCount': frame_count * HOP,
                'trimStart': round(result['cutStart'], 5),
                'worstWord': worst_word,
                'worstWordMatch': worst_word_match,
                'worstAnyWord': report['worstWord'],
                'worstAnyWordMatch': report['worstWordMatch'],
                'match': {key: report[key] for key in ('phones', 'labelPhones', 'exact', 'near', 'substituted', 'deleted',
                                                        'inserted', 'pauseAtBoundary', 'pauseInsideWord', 'pausesRefined')}
            }
            entries.append(entry)
            edits_by_speaker[utterance['speaker']].update(report['edits'])
            arrays = {'tokens': np.asarray(report['ids'], dtype=np.int16), 'durations': np.asarray(result['frames'], dtype=np.int16),
                      'seconds': np.asarray(result['seconds'], dtype=np.float32), 'audio': result['audio'],
                      'frames': result['tracks'].astype(np.float32)}
            for name in names:
                raw_files[name].write(np.ascontiguousarray(arrays[name], dtype=dtypes[name]).tobytes())
                lengths[name] += len(arrays[name])
            token_offset += len(result['tokens'])
            frame_offset += frame_count
            alignment_file.write(json.dumps({'id': entry['id'], 'tokens': ' '.join(result['tokens']), 'frames': result['frames'],
                                             'edits': report['edits'], 'cost': report['cost'],
                                             'readings': {str(position): ' '.join(phones) for position, phones in
                                                          utterance.get('readings', {}).items()}}) + '\n')
            if entry['include']:
                readings.check_built(utterance, result['tokens'], reading_check)
    for name in names:
        raw_files[name].close()
        shape = (lengths[name], len(CHANNELS)) if name == 'frames' else (lengths[name],)
        raw_to_npy(os.path.join(out_dir, name + '.raw'), os.path.join(out_dir, name + '.npy'), dtypes[name], shape)

    # The index - data.py's format, the speakers keyed by their pool key with their id as the index
    speaker_index = {}
    for index, key in enumerate(kept):
        speaker, speaker_settings = speakers[key], row_of[key]['settings']
        speaker_index[key] = {'index': index, 'sex': speaker['sex'], 'corpus': speaker['source'],
                              'sourceSampleRate': speaker['sourceSampleRate'], 'bandwidth': speaker['bandwidth'],
                              'pitchFloor': speaker_settings['pitchFloor'], 'pitchCeiling': speaker_settings['pitchCeiling'],
                              'formantCeiling': speaker_settings['formantCeiling']}
    index = {
        'version': FORMAT_VERSION,
        'sampleRate': SAMPLE_RATE,
        'hop': HOP,
        'frameRate': SAMPLE_RATE / HOP,
        'channels': CHANNELS,
        'channelNotes': CHANNEL_NOTES,
        'vocab': vocabulary(),
        'maxEdgeSilence': MAX_EDGE_SILENCE,
        'minMatch': MIN_MATCH,
        'minWordMatch': MIN_WORD_MATCH,
        'wordMatchExempt': 'LibriTTS-R utterances: function words and abbreviation expansions (picovox_train/multi.py WORD_MATCH_EXEMPT); '
                           'worstAnyWord is picovox_train/data.py\'s worst word over all words',
        'speakers': speaker_index,
        'readings': {**reading_counts, 'checkedTokens': reading_check['tokens'], 'disagree': reading_check['disagree'],
                     'szOnly': reading_check['szOnly']},
        'utterances': entries
    }
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=1)
        file.write('\n')
    with open(os.path.join(out_dir, 'failures.json'), 'w', encoding='utf-8') as file:
        json.dump(failures, file, indent=1)
    if not args.limit and not args.speakers:
        speakers_path, split_path = table_paths(out_dir)
        write_speakers(ArcticData(out_dir), speakers, row_of, edits_by_speaker, missing_labels, speakers_path)
        write_split(entries, split_path)
    share = reading_check['disagree'] / max(1, reading_check['tokens'])
    print(f'heteronym readings: {reading_counts["overrides"]} overrides; included utterances\' heteronym tokens {reading_check["tokens"]}, '
          f'{reading_check["disagree"]} ({100 * share:.2f}%) disagree with the aligner, {reading_check["szOnly"]} differ only by S/Z '
          f'(the front end\'s reading kept)'
          f'{" - over the 1% limit (readings.py MAX_DISAGREE)" if share > readings.MAX_DISAGREE else ""}')
    print(f'Built {out_dir}: {len(entries)} utterances ({sum(1 for entry in entries if entry["include"])} included), '
          f'{frame_offset / 100 / 3600:.2f} hours, {len(failures)} failures, {time.time() - start_time:.0f} s')


# The worst word of an utterance (data.py's word agreement) over the words not in WORD_MATCH_EXEMPT - (word, match)
def worst_content_word(text, status, word_readings=None):
    worst, worst_match = None, 1.0
    for word in tokenize(text, readings=word_readings)['words']:
        if word['text'] in WORD_MATCH_EXEMPT:
            continue
        statuses = status[word['start']:word['end']]
        good = sum(1 for item in statuses if item in ('exact', 'near'))
        total = len(statuses) + sum(1 for item in statuses if item and item.endswith('+inserted'))
        word_match = good / max(1, total)
        if total >= 2 and word_match < worst_match:
            worst, worst_match = word['text'], round(word_match, 3)
    return worst, worst_match


# data.py's process_utterance, an error caught (an utterance whose labels can't be aligned is left out)
def safe_process_utterance(task):
    try:
        return data.process_utterance(task)
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {'utterance': task[0], 'error': f'{type(error).__name__}: {error}'}


# Convert a raw array file to .npy (in chunks), removing the raw file
def raw_to_npy(raw_path, npy_path, dtype, shape):
    output = np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=shape)
    flat = output.reshape(-1)
    chunk = 1 << 26
    with open(raw_path, 'rb') as file:
        offset = 0
        while True:
            buffer = file.read(chunk * np.dtype(dtype).itemsize)
            if not buffer:
                break
            values = np.frombuffer(buffer, dtype=dtype)
            flat[offset:offset + len(values)] = values
            offset += len(values)
    output.flush()
    del output
    os.remove(raw_path)


# Write data/nn/speakers.json - the speaker id table (each speaker's listening score, utmos, kept from the table there,
# or from data/nn/'s for a new table)
def write_speakers(corpus_data, speakers, row_of, edits_by_speaker, missing_labels, path=SPEAKERS_PATH):
    previous = load_json(path if os.path.exists(path) else SPEAKERS_PATH, {'speakers': []})['speakers']
    utmos = {row['key']: row['utmos'] for row in previous if 'utmos' in row}
    f0_column = CHANNELS.index('f0')
    table = []
    for key, info in corpus_data.index['speakers'].items():
        speaker, row = speakers[key], row_of[key]
        all_entries = corpus_data.entries(speakers=[key], include_excluded=True)
        included = [entry for entry in all_entries if entry['include']]
        hours = {split: round(sum(entry['frameCount'] for entry in included if entry['split'] == split) / 100 / 3600, 4)
                 for split in ('train', 'val', 'test')}
        totals = collections.Counter()
        for entry in all_entries:
            totals.update(entry['match'])
        f0_values = np.concatenate([
            np.asarray(corpus_data.frames[entry['frameOffset']:entry['frameOffset'] + entry['frameCount'], f0_column])
            for entry in included
        ]) if included else np.zeros(0)
        f0_values = f0_values[f0_values > 0]
        accent = row['accent']
        table.append({
            'id': info['index'],
            'key': key,
            'source': speaker['source'],
            'corpusSpeaker': speaker['speaker'],
            'reader': speaker['reader'],
            'sex': speaker['sex'],
            'sexCorrected': speaker.get('sexCorrected', False),
            'accent': {'top': accent['top'], 'us': accent['posterior'].get('us'), 'canada': accent['posterior'].get('canada'),
                       'votes': accent['votes']} if accent else None,
            'accentConfidence': accent['posterior'].get('us') if accent else None,
            'erF3Ratio': (row['settings'] or {}).get('erF3Ratio'),
            'hours': round(sum(hours.values()), 4),
            'hoursBySplit': hours,
            'utterances': {'all': len(all_entries), 'included': len(included), 'noLabels': missing_labels.get(key, 0),
                           **{split: sum(1 for entry in included if entry['split'] == split) for split in ('train', 'val', 'test')}},
            'medianF0': round(float(np.median(f0_values)), 1) if len(f0_values) else None,
            'pitchFloor': info['pitchFloor'],
            'pitchCeiling': info['pitchCeiling'],
            'formantCeiling': info['formantCeiling'],
            'sourceSampleRate': info['sourceSampleRate'],
            'bandwidth': info['bandwidth'],
            'phoneMatch': round((totals['exact'] + totals['near']) / max(1, totals['phones']), 4),
            'exactMatch': round(totals['exact'] / max(1, totals['phones']), 4),
            'commonestEdits': [f'{edit} {count}' for edit, count in edits_by_speaker[key].most_common(5)],
            **({'utmos': utmos[key]} if key in utmos else {})
        })
    table.sort(key=lambda item: item['id'])
    result = {
        'description': 'The multi-speaker pool of picovox_train/multi.py - American English readers (LibriTTS-R train-clean-100 and '
                       'dev-clean speakers classified "us" by the former accent classifier (git history), and the labeled CMU '
                       'ARCTIC US speakers): id is the '
                       'training speaker index (build/nn/data/multi/index.json), hours are of the included utterances, medianF0 '
                       'is Praat\'s over their voiced frames, and phoneMatch is the share of our front end\'s phones paired '
                       'exactly (or AH/AX) with the corpus labels',
        'usThreshold': US_THRESHOLD,
        'speakers': table
    }
    write_json(path, result)


# Write data/nn/multi-split.json - the held-out utterance ids per speaker
def write_split(entries, path=SPLIT_PATH):
    split = {'description': 'The multi-speaker split of picovox_train/multi.py - per speaker, the held-out utterances ("val" and "test"); '
                            'ARCTIC speakers by sentence (data.py\'s split), LibriTTS-R by a hash of the utterance id '
                            f'({TEST_PERCENT}% test, {VAL_PERCENT}% validation); the rest is training',
             'val': collections.defaultdict(list), 'test': collections.defaultdict(list)}
    for entry in entries:
        if entry['split'] in ('val', 'test') and entry['include']:
            split[entry['split']][entry['speaker']].append(entry['sentence'])
    write_json(path, {'description': split['description'], 'val': dict(split['val']), 'test': dict(split['test'])})


#
# Token parity
#

# Verify lib/tokens.js and frontend.py agree on every pool text
def verify_tokens_command(args):
    corpus_data = ArcticData(args.out or OUT_DIR)
    texts = sorted({entry['text'] for entry in corpus_data.index['utterances']})
    pronounce_texts(texts, args.jobs)
    mismatches = 0
    tokens = 0
    for start in range(0, len(texts), 5000):
        chunk = texts[start:start + 5000]
        for text, js_result in zip(chunk, data.js_tokenize(chunk)):
            tokens += len(js_result['ids'])
            if data.tokenize(text) != js_result:
                mismatches += 1
                if mismatches <= 5:
                    print(f'MISMATCH: {text!r}')
    print(f'verify-tokens: {len(texts)} pool texts, {tokens} tokens: {len(texts) - mismatches} identical, {mismatches} different')
    return 1 if mismatches else 0


#
# Utilities
#

# Load a JSON file, or a default when it doesn't exist
def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# Write a JSON file atomically
def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w', encoding='utf-8') as file:
        json.dump(value, file, indent=1)
        file.write('\n')
    os.replace(path + '.tmp', path)


# The command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog='picovox_train.multi', description='Build the multi-speaker American English training data')
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build', help='build build/nn/data/multi/')
    build_parser.add_argument('--speakers', nargs='+', help='only these pool speakers (a quick test; no speakers.json)')
    build_parser.add_argument('--limit', type=int, help='at most this many utterances per speaker (a quick test)')
    build_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    build_parser.add_argument('--out', help='the corpus directory (default: build/nn/data/multi/, its tables in data/nn/)')
    build_parser.add_argument('--pool', default=SPEAKERS_PATH, help='the pool - a speaker table (default: data/nn/speakers.json)')
    verify_parser = commands.add_parser('verify-tokens', help='check lib/tokens.js against frontend.py on the pool texts')
    verify_parser.add_argument('--out', help='the corpus directory (default: build/nn/data/multi/)')
    verify_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    args = parser.parse_args(argv)
    handler = {'build': build_command, 'verify-tokens': verify_tokens_command}
    return handler[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
