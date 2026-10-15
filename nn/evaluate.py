# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Evaluation of the neural voices - the exported voices (data/nn/<voice>.*) spoken by the browser path (nn/js/speak.js:
nn/js/model.js and nn/js/render.js at 44.1 kHz) against the rule voice of the same sex (lib/speak.js), the original
recordings, and the analysis-by-synthesis ceiling (nn/fit.py's fitted re-syntheses, build/nn/fit-eq-s005-va/):

The neural voice is spoken twice: as exported (with its default formant bandwidth scale - nn/export.py), and raw
(neural-raw, the network's own bandwidths). The speed is of both together.

- the listener panel (tools/listeners.py) - UTMOS mean and minimum, and ECAPA speaker similarity: the cosine of the
  group's mean embedding with the original recordings' (the panel's "similarity"), and per utterance with the same
  sentence's recording
- Whisper (whisper.cpp base.en, as tools/eval.js) - the word error rate
- prosody on the held-out ARCTIC test sentences - the predicted durations against the recordings' (phones, pauses),
  and Praat F0 statistics of the speech against the recordings'

on two text sets: the held-out ARCTIC test sentences (data/nn/arctic-split.json) and the listener panel's texts
(tools/panel-texts.json - Harvard sentences, narrative passages, questions, exclamations).

    build/venv-nn/bin/python -B -m nn.evaluate --voices slt bdl [--out build/nn/eval]
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import subprocess
import sys

import numpy
import parselmouth
import soundfile

from .bridge import cosine, listen
from .data import ArcticData
from .frontend import tokenize
from .model import FIRST_PHONE


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCTIC = os.path.join(ROOT, 'data', 'corpora', 'arctic')
FIT_DIR = os.path.join(ROOT, 'build', 'nn', 'fit-eq-s005-va')
SEX = {'slt': 'female', 'bdl': 'male'}


#
# Whisper
#

# Words for scoring (tools/eval.js words)
def words(text):
    return [word for word in re.sub(r'[^a-z0-9\' ]+', ' ', text.lower()).replace("'", '').split() if word]


# Word-level edit distance
def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for ix, ref_word in enumerate(reference, 1):
        current = [ix] + [0] * len(hypothesis)
        for jx, hyp_word in enumerate(hypothesis, 1):
            current[jx] = min(previous[jx] + 1, current[jx - 1] + 1, previous[jx - 1] + (ref_word != hyp_word))
        previous = current
    return previous[-1]


# Transcribe a WAV file with whisper.cpp (cached beside it as <name>.json) - the text
def transcribe(path):
    base = path[:-4]
    if not os.path.exists(base + '.json') or os.path.getmtime(base + '.json') < os.path.getmtime(path):
        subprocess.run(['whisper-cli', '-m', os.path.join(ROOT, 'data', 'whisper', 'ggml-base.en.bin'), '-t', '2', '-f', path,
                        '-np', '-ojf', '-of', base], check=True, capture_output=True, cwd=ROOT)
    with open(base + '.json', encoding='utf-8') as file:
        result = json.load(file)
    return ''.join(segment['text'] for segment in result['transcription']).strip()


# Transcripts of WAV files, in parallel - {path: text}
def transcribe_all(paths, workers=6):
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return dict(zip(paths, pool.map(transcribe, paths)))


#
# Prosody
#

# Praat F0 statistics of a WAV file - the voiced frames' F0 (Hz)
def f0_values(path, floor, ceiling):
    audio, rate = soundfile.read(path)
    sound = parselmouth.Sound(audio, sampling_frequency=rate)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    values = pitch.selected_array['frequency']
    return values[values > 0]


# Summary statistics of F0 values - median (Hz), standard deviation and 5-95% range (semitones)
def f0_summary(values):
    semitones = 12 * numpy.log2(values / numpy.median(values))
    return {'median': round(float(numpy.median(values)), 1), 'sdSemitones': round(float(semitones.std()), 2),
            'range90Semitones': round(float(numpy.percentile(semitones, 95) - numpy.percentile(semitones, 5)), 2)}


#
# The evaluation
#

def run_node(script, args):
    process = subprocess.run(['node', os.path.join(ROOT, script), *args], check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(process.stdout.strip().splitlines()[-1])


# Group summary - UTMOS mean/min, WER, similarity of the mean embedding to a reference group's, per-utterance similarity
def summarize(files, scores, transcripts, reference_mean=None, pairs=None):
    mos = [scores[item['wav']]['mos'] for item in files]
    errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in files)
    count = sum(len(words(item['text'])) for item in files)
    mean_embedding = numpy.mean([scores[item['wav']]['embedding'] for item in files], axis=0)
    summary = {'count': len(files), 'mos': round(float(numpy.mean(mos)), 3), 'mosMin': round(float(numpy.min(mos)), 3),
               'wer': round(100 * errors / max(1, count), 2)}
    if reference_mean is not None:
        summary['similarity'] = round(cosine(mean_embedding, reference_mean), 3)
    if pairs:
        summary['pairSimilarity'] = round(float(numpy.mean([cosine(scores[first]['embedding'], scores[second]['embedding'])
                                                            for first, second in pairs])), 3)
    return summary, mean_embedding


def evaluate_voice(voice, args, data):
    speaker = voice
    sex = SEX[speaker]
    directory = os.path.join(args.out, voice)
    os.makedirs(directory, exist_ok=True)
    settings = data.index['speakers'][speaker]
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        panel = json.load(file)['texts'][:args.panel_count]
    test_entries = data.entries('test', [speaker])[:args.test_count]

    # The texts and their files
    groups = {'test': [], 'panel': []}
    for entry in test_entries:
        name = entry['sentence']
        groups['test'].append({'id': name, 'text': entry['text'], 'entry': entry,
                               'neural': os.path.join(directory, f'test-{name}-neural.wav'),
                               'neural-raw': os.path.join(directory, f'test-{name}-neural-raw.wav'),
                               'params': os.path.join(directory, f'test-{name}-params.json'),
                               'rule': os.path.join(directory, f'test-{name}-rule.wav'),
                               'original': os.path.join(ARCTIC, f'cmu_us_{speaker}_arctic', 'wav', f'{name}.wav')})
    for item in panel:
        groups['panel'].append({'id': item['id'], 'text': item['text'], 'kind': item['kind'],
                                'neural': os.path.join(directory, f'panel-{item["id"]}-neural.wav'),
                                'neural-raw': os.path.join(directory, f'panel-{item["id"]}-neural-raw.wav'),
                                'params': os.path.join(directory, f'panel-{item["id"]}-params.json'),
                                'rule': os.path.join(directory, f'panel-{item["id"]}-rule.wav')})
    items = groups['test'] + groups['panel']

    # The neural voice (the browser path) and the rule voice
    batch_path = os.path.join(directory, 'neural-batch.json')
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump([{'text': item['text'], 'wav': item['neural'], 'params': item['params'], 'seed': ix + 1}
                   for ix, item in enumerate(items)] +
                  [{'text': item['text'], 'wav': item['neural-raw'], 'seed': ix + 1, 'bandwidthScale': 1}
                   for ix, item in enumerate(items)], file)
    timing = run_node('nn/js/speak.js', ['-m', os.path.join(args.models, voice), '--batch', batch_path])
    rule_path = os.path.join(directory, 'rule-batch.json')
    with open(rule_path, 'w', encoding='utf-8') as file:
        json.dump([{'text': item['text'], 'klatt': item['rule'], 'seed': ix + 1} for ix, item in enumerate(items)], file)
    run_node('nn/js/speakparams.js', ['-v', sex, '--batch', rule_path])

    # The analysis-by-synthesis ceiling's files (nn/fit.py - arctic_a0001-a0006, 16 kHz)
    fit_files = []
    for number in range(1, 7):
        name = f'arctic_a{number:04d}'
        fitted = os.path.join(FIT_DIR, speaker, f'{name}-fitted.wav')
        if os.path.exists(fitted):
            entry = next((entry for entry in data.entries(None, [speaker], include_excluded=True) if entry['sentence'] == name), None)
            fit_files.append({'id': name, 'text': entry['text'] if entry else '', 'fitted': fitted,
                              'original': os.path.join(FIT_DIR, speaker, f'{name}-original.wav')})

    # Score everything
    paths = [item[key] for item in items for key in ('neural', 'neural-raw', 'rule', 'original') if key in item]
    paths += [item[key] for item in fit_files for key in ('fitted', 'original')]
    scores = listen(paths)
    transcripts = transcribe_all(paths)

    def files(group, key):
        return [{'wav': item[key], 'text': item['text']} for item in group if key in item]

    results = {'voice': voice, 'speaker': speaker, 'ruleVoice': sex, 'speed': timing, 'test': {}, 'panel': {}, 'ceiling': {}}
    original_summary, original_mean = summarize(files(groups['test'], 'original'), scores, transcripts)
    original_summary['similarity'] = 1.0
    results['test']['original'] = original_summary
    for key in ('neural', 'neural-raw', 'rule'):
        pairs = [(item[key], item['original']) for item in groups['test']]
        results['test'][key], _ = summarize(files(groups['test'], key), scores, transcripts, original_mean, pairs)
        results['panel'][key], _ = summarize(files(groups['panel'], key), scores, transcripts, original_mean)
        results['panel'][key]['mosByKind'] = {
            kind: round(float(numpy.mean([scores[item[key]]['mos'] for item in groups['panel'] if item['kind'] == kind])), 3)
            for kind in sorted({item['kind'] for item in groups['panel']})}
    if fit_files:
        pairs = [(item['fitted'], item['original']) for item in fit_files]
        results['ceiling']['fitted'], _ = summarize(files(fit_files, 'fitted'), scores, transcripts, original_mean, pairs)
        results['ceiling']['fitOriginals'], _ = summarize(files(fit_files, 'original'), scores, transcripts, original_mean)

    # Prosody on the test sentences - durations (the network's, against the recordings' aligned durations) and F0
    vocab = data.index['vocab']
    phone_errors = []
    phone_ratio = []
    pauses_true = {'comma': 0, 'word': 0}
    pauses_pred = {'comma': 0, 'word': 0}
    commas = 0
    f0_neural = []
    f0_original = []
    f0_rule = []
    for item in groups['test']:
        with open(item['params'], encoding='utf-8') as file:
            predicted = [duration for sentence in json.load(file)['sentences'] for duration in sentence['durations']]
        truth = data.arrays(item['entry'])
        ids = truth['tokens']
        if len(predicted) == len(ids) and tokenize(item['text'])['ids'] == ids.tolist():
            for ix, token in enumerate(ids):
                name = vocab[token]
                if token >= FIRST_PHONE:
                    phone_errors.append(abs(math.log((predicted[ix] + 1) / (truth['durations'][ix] + 1))))
                elif name == '<comma>':
                    commas += 1
                    pauses_true['comma'] += int(truth['durations'][ix] > 0)
                    pauses_pred['comma'] += int(predicted[ix] > 0)
                elif name == '_':
                    pauses_true['word'] += int(truth['durations'][ix] > 0)
                    pauses_pred['word'] += int(predicted[ix] > 0)
            phones = ids >= FIRST_PHONE
            phone_ratio.append(sum(numpy.array(predicted)[phones]) / truth['durations'][phones].sum())
        floor, ceiling = settings['pitchFloor'], settings['pitchCeiling']
        f0_neural.append(f0_values(item['neural'], floor, ceiling))
        f0_original.append(f0_values(item['original'], floor, ceiling))
        f0_rule.append(f0_values(item['rule'], floor * 0.7, ceiling * 1.3))
    results['prosody'] = {
        'phoneDurationMeanAbsLogError': round(float(numpy.mean(phone_errors)), 3),
        'phoneDurationMeanAbsPercent': round(float(100 * (numpy.exp(numpy.mean(phone_errors)) - 1)), 1),
        'speechDurationRatio': round(float(numpy.mean(phone_ratio)), 3),
        'commas': commas, 'commaPausesRecorded': pauses_true['comma'], 'commaPausesPredicted': pauses_pred['comma'],
        'wordPausesRecorded': pauses_true['word'], 'wordPausesPredicted': pauses_pred['word'],
        'f0': {'original': f0_summary(numpy.concatenate(f0_original)), 'neural': f0_summary(numpy.concatenate(f0_neural)),
               'rule': f0_summary(numpy.concatenate(f0_rule)),
               'utteranceSdSemitones': {
                   name: round(float(numpy.mean([numpy.std(12 * numpy.log2(values / numpy.median(values))) for values in group
                                                 if len(values) > 10])), 2)
                   for name, group in [('original', f0_original), ('neural', f0_neural), ('rule', f0_rule)]}}
    }
    with open(os.path.join(directory, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump({'results': results, 'scores': {path: {'mos': scores[path]['mos'], 'transcript': transcripts[path]} for path in paths}},
                  file, indent=1)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.evaluate', description='Evaluate the neural voices')
    parser.add_argument('--voices', nargs='+', default=['slt', 'bdl'])
    parser.add_argument('--models', default=os.path.join(ROOT, 'data', 'nn'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'nn', 'eval'))
    parser.add_argument('--test-count', type=int, default=66)
    parser.add_argument('--panel-count', type=int, default=16)
    args = parser.parse_args(argv)
    data = ArcticData()
    results = {voice: evaluate_voice(voice, args, data) for voice in args.voices}
    with open(os.path.join(args.out, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    json.dump(results, sys.stdout, indent=2)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
