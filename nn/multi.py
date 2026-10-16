# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The multi-speaker training data - American audiobook readers: LibriTTS-R (train-clean-100 and dev-clean) speakers
classified as US English by nn/accent.py, and the labeled CMU ARCTIC US speakers (slt, bdl, clb, rms), featurized
exactly as nn/data.py featurizes ARCTIC (the same tokens, alignment, format, and auxiliary tracks), into one corpus
directory that nn/data.py's ArcticData loads.

    build/venv-listeners/bin/python -B -m nn.accent                 # 1. accents -> build/nn/multi/accent.json
    build/venv-nn/bin/python -B -m nn.multi settings                # 2. per-speaker analysis settings (about 5 minutes)
    build/venv-nn/bin/python -B -m nn.multi pool                    # 3. the accent filter and the pool (a report)
    build/venv-nn/bin/python -B -m nn.multi build -j 8              # 4. build/nn/data/multi/ (about 6 minutes on 14 cores)
    build/venv-nn/bin/python -B -m nn.multi build --pool data/nn/speakers.json   # 4 alone: the committed pool (no 1-3)
    build/venv-nn/bin/python -B -m nn.multi verify-tokens           # nn/js/tokens.js == nn/frontend.py on the pool texts
    build/venv-nn/bin/python -B -m nn.multi stats                   # the pool's speakers, hours, and match rates
    build/venv-nn/bin/python -B -m nn.multi listen                  # UTMOS per speaker -> data/nn/speakers.json

The corpus (build/nn/data/multi/) has nn/data.py's format (index.json, tokens.npy, durations.npy, seconds.npy,
audio.npy, frames.npy, alignment.jsonl): load it with ArcticData(os.path.join(DATA_DIR, 'multi')). build --out DIR
builds it elsewhere, with its speaker table and split (speakers.json, multi-split.json) beside it instead of in
data/nn/ (verify-tokens and stats take the same --out). Its speakers are
keyed "arctic-<name>" and "libritts-<id>"; each speaker's index.json entry has nn/data.py's analysis settings (sex,
pitchFloor, pitchCeiling, formantCeiling, sourceSampleRate, bandwidth) plus its index, which is its id in
data/nn/speakers.json.

The pieces:

- Speakers - a LibriTTS-R speaker's sex is from the corpus's speakers.tsv; its pitch range is Hirst's two-pass range
  (0.75 x the first quartile to 1.5 x the third quartile of a 60-600 Hz first pass, as tools/targets.py), and its formant
  ceiling Escudero's (the ceiling with the least within-vowel variance of log F1 and log F2 over the stressed vowels'
  midpoints, from the MFA labels), both over up to 60 utterances. The ARCTIC speakers keep their data/targets/ settings
  (so slt and bdl are featurized exactly as in build/nn/data/arctic/).
- The accent filter - a LibriTTS-R speaker is kept when the CommonAccent classifier's top accent is "us" with a mean
  posterior of at least US_THRESHOLD over its utterances (nn/accent.py); the ARCTIC speakers are US English by the
  corpus's documentation (and the classifier agrees). The rhoticity measure of nn/data.py's survey (the median F3 of
  stressed ER over the other stressed vowels') is recorded beside it.
- Text and labels - LibriTTS-R's normalized text (numbers, abbreviations, and symbols already spelled out as words)
  through our front end (every word pronounced by the app's G2P network); LibriTTSLabel's Montreal Forced Aligner
  phone labels for the times. The alignment, the edge
  silence trim, and the exclusion rules (phone match below MIN_MATCH, a word below MIN_WORD_MATCH, a phone with no frames)
  are nn/data.py's, except that LibriTTS-R's word rule skips function words and abbreviation expansions
  (WORD_MATCH_EXEMPT - the aligner's dictionary reduces them); utterances with no MFA labels are left out.
- Splits - per speaker: an ARCTIC utterance has nn/data.py's sentence split (the same held-out sentences as
  data/nn/arctic-split.json); a LibriTTS-R utterance is "test" (2%) or "val" (3%) by a hash of its utterance id, else
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
import subprocess
import sys
import time

import numpy as np

from . import data
from . import readings
from .data import (ARCTIC_DIR, BOUNDARIES, CORPORA_DIR, DATA_DIR, FORMAT_VERSION, HOP, MAX_EDGE_SILENCE, MIN_MATCH,
                   MIN_WORD_MATCH, SAMPLE_RATE, CHANNELS, CHANNEL_NOTES, ArcticData, match_rate, read_mfa)
from .frontend import ABBREVIATIONS, FUNCTION_WORDS, ROOT, pronounce_texts, set_pronunciations, tokenize, vocabulary


MULTI_DIR = os.path.join(ROOT, 'build', 'nn', 'multi')
OUT_DIR = os.path.join(DATA_DIR, 'multi')
ACCENT_PATH = os.path.join(MULTI_DIR, 'accent.json')
SETTINGS_PATH = os.path.join(MULTI_DIR, 'settings.json')
SPEAKERS_PATH = os.path.join(ROOT, 'data', 'nn', 'speakers.json')
SPLIT_PATH = os.path.join(ROOT, 'data', 'nn', 'multi-split.json')
LIBRITTS_DIR = os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R')
LABEL_DIR = os.path.join(CORPORA_DIR, 'libritts_label', 'lab', 'phone')

# The LibriTTS-R subsets in the pool
SUBSETS = ['train-clean-100', 'dev-clean']

# The labeled ARCTIC US speakers in the pool, in id order (slt and bdl first, as in the single-speaker data)
ARCTIC_SPEAKERS = {'slt': 'female', 'bdl': 'male', 'clb': 'female', 'rms': 'male'}

# Corrections of speakers.tsv's sex - speakers whose label contradicts their name and voice (median F0)
SEX_OVERRIDES = {'7976': 'female'}     # JenniferRutters, labeled M, median F0 about 200 Hz

# The accent filter - the least mean US posterior of a kept speaker (its top accent must be "us"; the classifier gives
# slt, an American voice talent, 0.63 - it confuses US and Canadian English, more for women), and the rhoticity guard: a
# speaker leaning non-rhotic (ER/vowel F3 ratio at least RHOTIC_RATIO - the US-classified median is 0.64, the
# England-classified 0.78-0.83) needs a US posterior of at least RHOTIC_US
US_THRESHOLD = 0.5
RHOTIC_RATIO = 0.75
RHOTIC_US = 0.8

# The least audio of a kept speaker (minutes) - enough for a speaker embedding and a validation share
MIN_SPEAKER_MINUTES = 3.0

# The worst-word rule's exempt words - nn/data.py's rule is for our letter-to-sound guesses at unknown words; in
# LibriTTS-R the Montreal Forced Aligner's dictionary picks reduced or variant pronunciations of function words ("for"
# F ER, "are" ER, "our" AA R, "the" DH IY) and has no "mrs" (its <unk>), which the rule would otherwise exclude by the
# thousand although the words' timing is right; the phone-match rule still applies to every phone
WORD_MATCH_EXEMPT = FUNCTION_WORDS | set(ABBREVIATIONS.values())

# The LibriTTS-R held-out shares (percent) - test and validation
TEST_PERCENT = 2
VAL_PERCENT = 3

# Speaker analysis - utterances sampled, the candidate formant ceilings (tools/targets.py's), and the stressed
# monophthongs used for the ceiling choice
SETTINGS_UTTERANCES = 60
FORMANT_CEILINGS = {'male': [4000, 4250, 4500, 4750, 5000, 5250, 5500], 'female': [4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250]}
CEILING_VOWELS = ['IY', 'IH', 'EH', 'AE', 'AA', 'AO', 'UH', 'UW', 'AH']


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


# The split of a pool utterance - ARCTIC by sentence (nn/data.py), LibriTTS-R by a hash of the utterance id
def utterance_split(corpus, sentence):
    if corpus == 'arctic':
        return data.sentence_split(sentence)
    bucket = int(hashlib.sha1(sentence.encode('utf-8')).hexdigest(), 16) % 100
    return 'test' if bucket < TEST_PERCENT else ('val' if bucket < TEST_PERCENT + VAL_PERCENT else 'train')


#
# Speaker analysis settings
#

# One speaker's settings - (key, {pitchFloor, pitchCeiling, formantCeiling, formantCeilingScores, firstPassMedianF0,
# erF3Ratio, erCount})
def speaker_settings_task(task):
    import parselmouth  # pylint: disable=import-outside-toplevel
    import soundfile  # pylint: disable=import-outside-toplevel
    key, sex, utterances = task
    sample = data_subset([utterance for utterance in utterances if os.path.exists(utterance['lab'])], SETTINGS_UTTERANCES)
    first_f0 = []
    vowels = []       # [(vowel, {ceiling: (F1, F2, F3)})]
    er_f3, other_f3 = [], []
    ceilings = FORMANT_CEILINGS[sex]
    for utterance in sample:
        samples, rate = soundfile.read(utterance['wav'], dtype='float64')
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        sound = parselmouth.Sound(samples, sampling_frequency=rate)
        f0 = sound.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600).selected_array['frequency']
        first_f0.extend(f0[f0 > 0].tolist())
        midpoints = [(label[:-1], (start + end) / 2) for label, start, end in read_mfa(utterance['lab'])
                     if label[-1:] in ('1', '2') and (label[:-1] in CEILING_VOWELS or label[:-1] == 'ER') and end - start >= 0.05]
        if not midpoints:
            continue
        measures = [{} for _ in midpoints]
        for ceiling in ceilings:
            formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling,
                                            window_length=0.025, pre_emphasis_from=50)
            for measure, (_vowel, center) in zip(measures, midpoints):
                measure[ceiling] = tuple(formant.get_value_at_time(number, center) for number in (1, 2, 3))
        vowels.extend((vowel, measure) for (vowel, _), measure in zip(midpoints, measures))
    if len(first_f0) < 100:
        return key, None
    quartile1, quartile3 = np.percentile(first_f0, 25), np.percentile(first_f0, 75)
    settings = {'pitchFloor': round(float(max(40.0, 0.75 * quartile1)), 1),
                'pitchCeiling': round(float(min(800.0, 1.5 * quartile3)), 1),
                'firstPassMedianF0': round(float(np.median(first_f0)), 1)}

    # Escudero's formant ceiling - the least within-vowel variance of log F1 and log F2
    scores = {}
    for ceiling in ceilings:
        score = 0.0
        for vowel in CEILING_VOWELS:
            values = [measure[ceiling] for name, measure in vowels if name == vowel]
            values = [value for value in values if np.isfinite(value[0]) and np.isfinite(value[1])]
            if len(values) >= 5:
                score += np.var(np.log([value[0] for value in values])) + np.var(np.log([value[1] for value in values]))
        scores[ceiling] = round(float(score), 4)
    best = min(ceilings, key=lambda ceiling: scores[ceiling])
    settings['formantCeiling'] = best
    settings['formantCeilingScores'] = {str(ceiling): score for ceiling, score in scores.items()}

    # Rhoticity at the chosen ceiling - stressed ER's median F3 over the other stressed vowels'
    for name, measure in vowels:
        value = measure[best][2]
        if np.isfinite(value):
            (er_f3 if name == 'ER' else other_f3).append(value)
    settings['erF3Ratio'] = round(float(np.median(er_f3) / np.median(other_f3)), 3) if len(er_f3) >= 5 and len(other_f3) >= 20 else None
    settings['erCount'] = len(er_f3)
    return key, settings


# Evenly spaced utterances (tools/targets.py's subset)
def data_subset(utterances, limit):
    if not limit or len(utterances) <= limit:
        return utterances
    step = len(utterances) / limit
    return [utterances[int(index * step)] for index in range(limit)]


# The ARCTIC speakers' settings from data/targets/ (nn/data.py's)
def arctic_settings(name):
    pitch_floor, pitch_ceiling, formant_ceiling = data.speaker_settings(name)
    return {'pitchFloor': pitch_floor, 'pitchCeiling': pitch_ceiling, 'formantCeiling': formant_ceiling, 'fromTargets': True}


# Measure every LibriTTS-R candidate's settings (incrementally) into build/nn/multi/settings.json
def settings_command(args):
    speakers = candidate_speakers()
    results = load_json(SETTINGS_PATH, {})
    tasks = [(key, speaker['sex'], speaker['utterances']) for key, speaker in speakers.items()
             if speaker['corpus'] != 'arctic' and (args.redo or key not in results) and speaker['sex']]
    start = time.time()
    with multiprocessing.Pool(args.jobs) as pool:
        for count, (key, settings) in enumerate(pool.imap_unordered(speaker_settings_task, tasks), 1):
            results[key] = settings
            if settings:
                print(f'{key:16s} pitch {settings["pitchFloor"]:.0f}-{settings["pitchCeiling"]:.0f} '
                      f'(median {settings["firstPassMedianF0"]:.0f}) ceiling {settings["formantCeiling"]} '
                      f'ER/V F3 {settings["erF3Ratio"]} ({count}/{len(tasks)}, {time.time() - start:.0f} s)', flush=True)
            if count % 20 == 0 or count == len(tasks):
                write_json(SETTINGS_PATH, results)
    write_json(SETTINGS_PATH, results)

    # Compare with tools/targets.py's settings where it measured the speaker (dev-clean)
    differences = []
    for key, settings in results.items():
        target_path = os.path.join(ROOT, 'data', 'targets', f'{key}.json')
        if settings and os.path.exists(target_path):
            with open(target_path, encoding='utf-8') as file:
                analysis = json.load(file)['analysis']
            differences.append((key, settings['pitchFloor'] - analysis['pitchFloor'], settings['pitchCeiling'] - analysis['pitchCeiling'],
                                settings['formantCeiling'] - analysis['formantCeiling']))
    if differences:
        print(f'Against data/targets/ ({len(differences)} speakers): pitch floor median difference '
              f'{np.median([item[1] for item in differences]):+.1f} Hz, ceiling {np.median([item[2] for item in differences]):+.1f} Hz; '
              f'formant ceiling equal for {sum(1 for item in differences if item[3] == 0)}, within 250 Hz for '
              f'{sum(1 for item in differences if abs(item[3]) <= 250)}')


#
# The pool - the accent filter
#

# The pool - (speakers kept in id order, every candidate's row) from the candidates, accents, settings, and hours
def select_pool(speakers, accents, settings, hours):
    rows = []
    for key, speaker in speakers.items():
        accent = accents.get(key)
        speaker_settings = arctic_settings(speaker['speaker']) if speaker['corpus'] == 'arctic' else settings.get(key)
        row = {'key': key, 'corpus': speaker['corpus'], 'sex': speaker['sex'], 'utterances': len(speaker['utterances']),
               'hours': hours[key], 'accent': accent, 'settings': speaker_settings}
        rhotic_ratio = (speaker_settings or {}).get('erF3Ratio')
        us_posterior = accent['posterior'].get('us', 0.0) if accent else 0.0
        if speaker['corpus'] == 'arctic':
            row['keep'], row['reason'] = True, 'CMU ARCTIC US English speaker'
        elif accent is None:
            row['keep'], row['reason'] = False, 'no accent classification'
        elif accent['top'] != 'us' or us_posterior < US_THRESHOLD:
            row['keep'], row['reason'] = False, f'accent {accent["top"]} (us {us_posterior:.2f})'
        elif rhotic_ratio is not None and rhotic_ratio >= RHOTIC_RATIO and us_posterior < RHOTIC_US:
            row['keep'], row['reason'] = False, f'non-rhotic leaning (ER/V F3 {rhotic_ratio:.2f}, us {us_posterior:.2f})'
        elif hours[key] * 60 < MIN_SPEAKER_MINUTES:
            row['keep'], row['reason'] = False, f'{hours[key] * 60:.1f} minutes of audio'
        elif not speaker['sex']:
            row['keep'], row['reason'] = False, 'no sex in speakers.tsv'
        elif not speaker_settings:
            row['keep'], row['reason'] = False, 'no analysis settings'
        else:
            row['keep'], row['reason'] = True, f'us {us_posterior:.2f}'
        rows.append(row)
    kept = [row['key'] for row in rows if row['keep']]
    return kept, rows


# The pool of a speaker table (data/nn/speakers.json, which records each kept speaker's accent and analysis settings) -
# (the kept speakers in id order, their rows as select_pool gives them), so a build needn't classify accents or measure
# settings again
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


# Print the accent distribution and the pool, per sex
def pool_command(_args):
    speakers = candidate_speakers()
    accents = load_json(ACCENT_PATH, {})
    settings = load_json(SETTINGS_PATH, {})
    hours = {key: speaker_hours(speaker['utterances']) for key, speaker in speakers.items()}
    kept, rows = select_pool(speakers, accents, settings, hours)
    print_pool_report(rows, kept, hours)


# The hours of a speaker's audio (from the WAV headers)
def speaker_hours(utterances):
    import soundfile  # pylint: disable=import-outside-toplevel
    return sum(soundfile.info(utterance['wav']).duration for utterance in utterances) / 3600


# Print the accent distribution (LibriTTS-R speakers), the kept speakers per sex, and the borderline speakers
def print_pool_report(rows, kept, hours):
    libritts = [row for row in rows if row['corpus'] != 'arctic' and row['accent']]
    distribution = collections.Counter((row['accent']['top'], row['sex']) for row in libritts)
    print(f'Accent (top class) of {len(libritts)} LibriTTS-R speakers:')
    accent_counts = collections.Counter(row['accent']['top'] for row in libritts)
    for accent, _count in accent_counts.most_common():
        female, male = distribution.get((accent, 'female'), 0), distribution.get((accent, 'male'), 0)
        accent_hours = sum(hours[row['key']] for row in libritts if row['accent']['top'] == accent)
        print(f'  {accent:15s} {female + male:4d} ({female} female, {male} male), {accent_hours:.1f} h')
    us_top = [row for row in libritts if row['accent']['top'] == 'us']
    print(f'  top "us" but below {US_THRESHOLD}: {sum(1 for row in us_top if row["accent"]["posterior"]["us"] < US_THRESHOLD)}')
    for sex in ('female', 'male'):
        mine = [key for key in kept if next(row for row in rows if row['key'] == key)['sex'] == sex]
        print(f'Kept {sex}: {len(mine)} speakers, {sum(hours[key] for key in mine):.1f} h '
              f'({sum(1 for key in mine if key.startswith("arctic-"))} ARCTIC)')
    print(f'Kept: {len(kept)} speakers, {sum(hours[key] for key in kept):.1f} h')
    print('Left out: ' + '; '.join(f'{row["key"]} ({row["sex"]}, {60 * row["hours"]:.0f} min) {row["reason"]}'
                                   for row in rows if not row['keep']))
    near = sorted((row for row in libritts
                   if row['accent']['top'] in ('us', 'canada') and 0.3 <= row['accent']['posterior'].get('us', 0) < 0.7),
                  key=lambda row: row['accent']['posterior']['us'])
    print('Borderline (us posterior 0.3-0.7): ' + ', '.join(
        f'{row["key"]} {row["sex"][0]} {row["accent"]["top"]} us {row["accent"]["posterior"]["us"]:.2f}' for row in near))


#
# Build
#

# The paths of a corpus directory's speaker table and split - data/nn/'s for the default directory, else beside it
def table_paths(out_dir):
    if os.path.abspath(out_dir) == os.path.abspath(OUT_DIR):
        return SPEAKERS_PATH, SPLIT_PATH
    return os.path.join(out_dir, 'speakers.json'), os.path.join(out_dir, 'multi-split.json')


# Build build/nn/data/multi/ (or --out), data/nn/speakers.json, and data/nn/multi-split.json
def build_command(args):
    start_time = time.time()
    out_dir = args.out or OUT_DIR
    speakers = candidate_speakers()
    if args.pool:
        kept, rows = table_pool(args.pool)
    else:
        accents = load_json(ACCENT_PATH, {})
        settings = load_json(SETTINGS_PATH, {})
        hours = {key: speaker_hours(speaker['utterances']) for key, speaker in speakers.items()}
        kept, rows = select_pool(speakers, accents, settings, hours)
        write_json(os.path.join(MULTI_DIR, 'pool.json'), [{key: value for key, value in row.items() if key != 'settings'} for row in rows])
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
    print(f'{len(pronunciations)} distinct words pronounced, {time.time() - start_time:.0f} s', flush=True)

    # Heteronym readings - the aligner's reading where it differs from the front end's (nn/readings.py)
    reading_counts = readings.attach_readings(tasks)
    reading_check = {'tokens': 0, 'disagree': 0}
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

    # The index - nn/data.py's format, the speakers keyed by their pool key with their id as the index
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
        'wordMatchExempt': 'LibriTTS-R utterances: function words and abbreviation expansions (nn/multi.py WORD_MATCH_EXEMPT); '
                           'worstAnyWord is nn/data.py\'s worst word over all words',
        'speakers': speaker_index,
        'readings': {**reading_counts, 'checkedTokens': reading_check['tokens'], 'disagree': reading_check['disagree']},
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
          f'{reading_check["disagree"]} ({100 * share:.2f}%) disagree with the aligner'
          f'{" - over the 1% limit (nn/readings.py MAX_DISAGREE)" if share > readings.MAX_DISAGREE else ""}')
    print(f'Built {out_dir}: {len(entries)} utterances ({sum(1 for entry in entries if entry["include"])} included), '
          f'{frame_offset / 100 / 3600:.2f} hours, {len(failures)} failures, {time.time() - start_time:.0f} s')


# The worst word of an utterance (nn/data.py's word agreement) over the words not in WORD_MATCH_EXEMPT - (word, match)
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


# nn/data.py's process_utterance, an error caught (an utterance whose labels can't be aligned is left out)
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
        'description': 'The multi-speaker pool of nn/multi.py - American English readers (LibriTTS-R train-clean-100 and '
                       'dev-clean speakers classified "us" by nn/accent.py, and the labeled CMU ARCTIC US speakers): id is the '
                       'training speaker index (build/nn/data/multi/index.json), hours are of the included utterances, medianF0 '
                       'is Praat\'s over their voiced frames, and phoneMatch is the share of our front end\'s phones paired '
                       'exactly (or AH/AX) with the corpus labels',
        'usThreshold': US_THRESHOLD,
        'speakers': table
    }
    write_json(path, result)


# Write data/nn/multi-split.json - the held-out utterance ids per speaker
def write_split(entries, path=SPLIT_PATH):
    split = {'description': 'The multi-speaker split of nn/multi.py - per speaker, the held-out utterances ("val" and "test"); '
                            'ARCTIC speakers by sentence (as data/nn/arctic-split.json), LibriTTS-R by a hash of the utterance id '
                            f'({TEST_PERCENT}% test, {VAL_PERCENT}% validation); the rest is training',
             'val': collections.defaultdict(list), 'test': collections.defaultdict(list)}
    for entry in entries:
        if entry['split'] in ('val', 'test') and entry['include']:
            split[entry['split']][entry['speaker']].append(entry['sentence'])
    write_json(path, {'description': split['description'], 'val': dict(split['val']), 'test': dict(split['test'])})


#
# Token parity and statistics
#

# Verify nn/js/tokens.js and nn/frontend.py agree on every pool text
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


# Print the pool's statistics - per sex, speakers and hours; match rates; exclusion reasons; utterance lengths
def stats_command(args):
    out_dir = args.out or OUT_DIR
    corpus_data = ArcticData(out_dir)
    table = load_json(table_paths(out_dir)[0], {'speakers': []})['speakers']
    entries = corpus_data.index['utterances']
    sex_of = {key: info['sex'] for key, info in corpus_data.index['speakers'].items()}
    for sex in ('female', 'male'):
        mine = [entry for entry in entries if sex_of[entry['speaker']] == sex]
        included = [entry for entry in mine if entry['include']]
        rows = [row for row in table if row['sex'] == sex]
        all_hours = sum(entry['frameCount'] for entry in mine) / 360000
        included_hours = sum(entry['frameCount'] for entry in included) / 360000
        print(f'{sex}: {len(rows)} speakers, {len(mine)} utterances ({len(included)} included), '
              f'{all_hours:.2f} h ({included_hours:.2f} h included: '
              + ', '.join(f'{split} {sum(entry["frameCount"] for entry in included if entry["split"] == split) / 360000:.2f}'
                          for split in ('train', 'val', 'test')) + ')')
        speaker_hours = sorted(row['hours'] for row in rows)
        if speaker_hours:
            medians = [row['medianF0'] for row in rows if row['medianF0']]
            print(f'  hours per speaker: min {speaker_hours[0]:.2f}, median {np.median(speaker_hours):.2f}, max {speaker_hours[-1]:.2f}; '
                  f'median F0 {np.median(medians):.0f} Hz (range {min(medians):.0f}-{max(medians):.0f})')
    for corpus in ('arctic', 'libritts'):
        mine = [entry for entry in entries if entry['speaker'].startswith(corpus)]
        totals = collections.Counter()
        for entry in mine:
            totals.update(entry['match'])
        phones = max(1, totals['phones'])
        identical = sum(1 for entry in mine if entry['match']['substituted'] + entry['match']['deleted'] + entry['match']['inserted'] == 0)
        print(f'{corpus}: {len(mine)} utterances, {identical} ({100 * identical / max(1, len(mine)):.1f}%) with identical phone '
              f'sequences; phones {totals["phones"]}: exact {100 * totals["exact"] / phones:.2f}%, '
              f'AH/AX {100 * totals["near"] / phones:.2f}%, '
              f'match {100 * (totals["exact"] + totals["near"]) / phones:.2f}%, substituted {100 * totals["substituted"] / phones:.2f}%, '
              f'deleted {100 * totals["deleted"] / phones:.2f}%, inserted {100 * totals["inserted"] / phones:.2f}%')
    reasons = collections.Counter()
    for entry in entries:
        if not entry['include']:
            for reason in entry['reason'].split('; '):
                reasons[reason.split(' match')[0].split('"')[0].strip() or reason] += 1
    print('Exclusions: ' + ', '.join(f'{reason} {count}' for reason, count in reasons.most_common()))
    words = collections.Counter(entry['worstWord'] for entry in entries if not entry['include'] and entry['worstWord'])
    print('Commonest worst words of excluded utterances: ' + ', '.join(f'{word} {count}' for word, count in words.most_common(30)))
    seconds = np.array([entry['frameCount'] / 100 for entry in entries if entry['include']])
    print(f'Included utterance seconds: p5 {np.percentile(seconds, 5):.1f}, median {np.median(seconds):.1f}, '
          f'p95 {np.percentile(seconds, 95):.1f}, max {seconds.max():.1f}; over 20 s: {int(np.sum(seconds > 20))} '
          f'({np.sum(seconds[seconds > 20]) / 3600:.2f} h)')


#
# Listening proxy - UTMOS per speaker
#

# Score a few included utterances per speaker (the processed 24 kHz audio) with tools/listeners.py's UTMOS, on the CPU,
# and record each speaker's mean in data/nn/speakers.json (utmos) - for choosing the voices
def listen_command(args):
    import soundfile  # pylint: disable=import-outside-toplevel
    corpus_data = ArcticData(OUT_DIR)
    out_dir = os.path.join(MULTI_DIR, 'listen')
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for key in corpus_data.index['speakers']:
        entries = [entry for entry in corpus_data.entries(speakers=[key]) if 3.0 <= entry['frameCount'] / 100 <= 12.0]
        entries.sort(key=lambda entry: hashlib.sha1(entry['id'].encode('utf-8')).hexdigest())
        for entry in entries[:args.count]:
            path = os.path.join(out_dir, entry['id'].replace('/', '_') + '.wav')
            if not os.path.exists(path):
                soundfile.write(path, corpus_data.arrays(entry)['audio'], SAMPLE_RATE, subtype='PCM_16')
            files[path] = key
    list_path = os.path.join(out_dir, 'files.txt')
    with open(list_path, 'w', encoding='utf-8') as file:
        file.write(''.join(path + '\n' for path in files))
    python = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')
    process = subprocess.run([python, '-I', os.path.join(ROOT, 'tools', 'listeners.py'), '--no-embedding', '--no-emotion',
                              '--device', 'cpu', '--files-from', list_path], capture_output=True, text=True, check=True, cwd=ROOT)
    scores = collections.defaultdict(list)
    for line in process.stdout.splitlines():
        record = json.loads(line)
        scores[files[record['file']]].append(record['mos'])
    table = load_json(SPEAKERS_PATH, None)
    for row in table['speakers']:
        values = scores.get(row['key'])
        row['utmos'] = round(float(np.mean(values)), 3) if values else None
    write_json(SPEAKERS_PATH, table)
    for sex in ('female', 'male'):
        rows = sorted((row for row in table['speakers'] if row['sex'] == sex and row['utmos']), key=lambda row: -row['utmos'])
        print(f'{sex}: UTMOS median {np.median([row["utmos"] for row in rows]):.2f}; best: ' +
              ', '.join(f'{row["key"]} {row["utmos"]:.2f} ({row["hours"]:.2f} h)' for row in rows[:12]))


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
    parser = argparse.ArgumentParser(prog='nn.multi', description='Build the multi-speaker American English training data')
    commands = parser.add_subparsers(dest='command', required=True)
    settings_parser = commands.add_parser('settings', help='per-speaker pitch range and formant ceiling')
    settings_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    settings_parser.add_argument('--redo', action='store_true', help='remeasure speakers already measured')
    commands.add_parser('pool', help='the accent filter and the pool (a report)')
    build_parser = commands.add_parser('build', help='build build/nn/data/multi/')
    build_parser.add_argument('--speakers', nargs='+', help='only these pool speakers (a quick test; no speakers.json)')
    build_parser.add_argument('--limit', type=int, help='at most this many utterances per speaker (a quick test)')
    build_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    build_parser.add_argument('--out', help='the corpus directory (default: build/nn/data/multi/, its tables in data/nn/)')
    build_parser.add_argument('--pool', help='the pool from a speaker table (data/nn/speakers.json) - no accents or settings needed')
    verify_parser = commands.add_parser('verify-tokens', help='check nn/js/tokens.js against nn/frontend.py on the pool texts')
    verify_parser.add_argument('--out', help='the corpus directory (default: build/nn/data/multi/)')
    verify_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    stats_parser = commands.add_parser('stats', help='the pool\'s statistics')
    stats_parser.add_argument('--out', help='the corpus directory (default: build/nn/data/multi/)')
    listen_parser = commands.add_parser('listen', help='UTMOS per speaker (data/nn/speakers.json utmos)')
    listen_parser.add_argument('-n', '--count', type=int, default=4, help='utterances per speaker')
    args = parser.parse_args(argv)
    handler = {'settings': settings_command, 'pool': pool_command, 'build': build_command,
               'verify-tokens': verify_tokens_command, 'stats': stats_command, 'listen': listen_command}
    return handler[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
