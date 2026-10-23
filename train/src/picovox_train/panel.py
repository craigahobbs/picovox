# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The listener panel's scoring - how every evaluation measures speech (no training framework: an exported voice model is
spoken by the browser path, nn/js/speak.js):

- listen - UTMOS and the ECAPA speaker embedding (listeners.py in the listener-model environment, build/venv-listeners)
- transcribe - whisper.cpp base.en (data/whisper/ggml-base.en.bin), cached beside each WAV; words and edit_distance give
  the word error rate as the former tools/eval.js counted it
- cosine - the speaker similarity of two embeddings
- speak - a batch of texts with an exported model (its speed report)
- score_model - the voice model's evaluation (make train's gate measure): six speakers (data/nn/train.json distill eval) by the
  panel's 16 texts (tools/panel-texts.json), each speaker's mean UTMOS, WER, and mean embedding, and against a
  reference result (the teacher's), the speaker similarity - the result JSON distill.py pick compares
"""

import concurrent.futures
import json
import os
import re
import subprocess

import numpy

from . import ROOT


LISTENERS_PYTHON = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')
LISTENERS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'listeners.py')
WHISPER_MODEL = os.path.join(ROOT, 'data', 'whisper', 'ggml-base.en.bin')
TEXTS_PATH = os.path.join(ROOT, 'tools', 'panel-texts.json')


# Score WAV files with the listener panel (on the CPU) - {path: {mos, embedding}}
def listen(paths):
    command = [LISTENERS_PYTHON, '-I', LISTENERS_SCRIPT, '--device', 'cpu', '--no-emotion', '--files-from', '-']
    process = subprocess.run(command, input='\n'.join(paths), capture_output=True, text=True, check=True)
    return {item['file']: item for item in (json.loads(line) for line in process.stdout.splitlines() if line.strip())}


# Transcribe a WAV file with whisper.cpp base.en (cached beside it as <name>.json) - the text
def transcribe(path):
    base = path[:-4]
    if not os.path.exists(base + '.json') or os.path.getmtime(base + '.json') < os.path.getmtime(path):
        subprocess.run(['whisper-cli', '-m', WHISPER_MODEL, '-t', '2', '-f', path, '-np', '-ojf', '-of', base], check=True,
                       capture_output=True, cwd=ROOT)
    with open(base + '.json', encoding='utf-8') as file:
        result = json.load(file)
    return ''.join(segment['text'] for segment in result['transcription']).strip()


# Transcripts of WAV files, workers at a time (two threads each) - {path: text}
def transcribe_all(paths, workers=6):
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return dict(zip(paths, pool.map(transcribe, paths)))


# Words for scoring (the former tools/eval.js's words)
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


# The cosine similarity of two vectors
def cosine(first, second):
    first, second = numpy.asarray(first), numpy.asarray(second)
    return float(first @ second / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))


# The panel texts - [{id, text}]
def panel_texts():
    with open(TEXTS_PATH, encoding='utf-8') as file:
        return json.load(file)['texts']


# Speak a batch ([{text, wav, seed, speaker}]) with an export (nn/js/speak.js) - its speed report
def speak(model_path, batch, batch_path, rate):
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    command = ['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', model_path, '--batch', batch_path, '-r', str(rate)]
    process = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(process.stdout.strip().splitlines()[-1])


def score_model(model_path, out_dir, settings, reference=None, whisper_workers=6):
    """Speak and score an export (model_path - its manifest and weights without the extension) - {panelSpeed, speakers,
    utterances, panel}; settings - {speakers: [[key, sex]], rate}; reference - another model's result (the similarity)"""
    with open(model_path + '.json', encoding='utf-8') as file:
        keys = [speaker['key'] for speaker in json.load(file)['speakers']]
    items = []
    for speaker, sex in settings['speakers']:
        for number, text in enumerate(panel_texts()):
            items.append({'speaker': speaker, 'sex': sex, 'text': text['text'],
                          'wav': os.path.join(out_dir, f'{speaker}-{text["id"]}.wav'), 'seed': number + 1})
    batch = [{'text': item['text'], 'wav': item['wav'], 'seed': item['seed'], 'speaker': keys.index(item['speaker'])} for item in items]
    result = {'panelSpeed': speak(model_path, batch, os.path.join(out_dir, 'batch.json'), settings['rate'])}
    paths = [item['wav'] for item in items]
    scores = listen(paths)
    transcripts = transcribe_all(paths, whisper_workers)
    speakers = {}
    for speaker, sex in settings['speakers']:
        group = [item for item in items if item['speaker'] == speaker]
        mos = [scores[item['wav']]['mos'] for item in group]
        errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in group)
        count = sum(len(words(item['text'])) for item in group)
        embedding = numpy.mean([scores[item['wav']]['embedding'] for item in group], axis=0)
        row = {'sex': sex, 'mos': float(numpy.mean(mos)), 'mosMin': float(numpy.min(mos)), 'wer': 100 * errors / max(1, count),
               'embedding': embedding.tolist()}
        if reference is not None:
            row['similarity'] = cosine(embedding, reference['speakers'][speaker]['embedding'])
            row['pairedSimilarity'] = float(numpy.mean([
                cosine(scores[item['wav']]['embedding'], reference['utterances'][os.path.basename(item['wav'])]['embedding'])
                for item in group]))
        speakers[speaker] = row
    result['speakers'] = speakers
    result['utterances'] = {os.path.basename(item['wav']): {'mos': scores[item['wav']]['mos'],
                                                            'embedding': scores[item['wav']]['embedding'],
                                                            'transcript': transcripts[item['wav']]} for item in items}
    result['panel'] = {key: float(numpy.mean([row[key] for row in speakers.values()]))
                       for key in ('mos', 'wer', 'similarity', 'pairedSimilarity') if all(key in row for row in speakers.values())}
    if reference is not None:
        result['panel']['similarityMin'] = float(min(row['similarity'] for row in speakers.values()))
    for path in paths:
        os.remove(path)
    return result
