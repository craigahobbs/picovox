# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Evaluation of the multi-speaker, multi-style model - the exported model (data/nn/multi.*) spoken by the browser path
(nn/js/speak.js: nn/js/model.js and nn/js/render.js at 44.1 kHz), into build/nn/samples-multi/:

- speakers - ten sample speakers (SAMPLE_SPEAKERS, five female and five male) on their held-out sentences (the
  test split, data/nn/multi-split.json, topped up from the validation split) and on the listener panel's texts
  (tools/panel-texts.json), against their recordings and, for slt and bdl, the single-speaker voices (data/nn/slt,
  data/nn/bdl): UTMOS, Whisper's WER, ECAPA similarity to the recordings, and the voice quality measures
- voice quality - the over-smoothing check: in strongly voiced frames, the formant bandwidths and the aspiration and
  frication levels relative to the voicing (parameters - against the analysis-by-synthesis fits of nn/fit.py,
  build/nn/fit-eq-s005-va/, on the same ARCTIC sentences with the recorded durations), and from the audio (Praat)
  the harmonics-to-noise ratio, the smoothed cepstral peak prominence, and the level of 4-8 kHz relative to 0-4 kHz
  in voiced frames, against the recordings
- styles - narrators and emotional-corpus actors in each emotion (STYLE_TEXTS - CREMA-D's sentences and new ones), at
  intensity 1 (and for the narrators 0.5, and 1.5 - extrapolated): the emotion listener's (emotion2vec+) recognition
  rates, against its rates on the real CREMA-D recordings (data/nn/cremad.json)
- transforms - whisper, soft, and shouting (nn/js/transforms.js applied to the narrators' parameters): UTMOS, WER,
  emotion, and the measured level, F0, HNR, and voiced fraction against the plain rendering

    build/venv-nn/bin/python -B -m nn.evalmulti build/nn/runs/multi-b/best.pt [--model data/nn/multi] [--out build/nn/samples-multi]
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy
import parselmouth
import soundfile
import soxr
import torch
from parselmouth.praat import call

from . import params
from .bridge import cosine
from .data import DATA_DIR, ArcticData
from .evaluate import edit_distance, transcribe_all, words
from .export import load_model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTENERS_PYTHON = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')
FIT_DIR = os.path.join(ROOT, 'build', 'nn', 'fit-eq-s005-va')

# The sample speakers - the reader pool's best-rated speakers with at least 0.25 hours (data/nn/speakers.json UTMOS)
SAMPLE_SPEAKERS = {
    'female': ['arctic-slt', 'libritts-5895', 'libritts-6345', 'libritts-4160', 'libritts-8312'],
    'male': ['arctic-bdl', 'libritts-6454', 'libritts-446', 'arctic-rms', 'libritts-8419']
}

# The style speakers - narrators, and an actor of each sex from the emotional corpus (in-domain)
STYLE_SPEAKERS = ['libritts-5895', 'libritts-6345', 'libritts-6454', 'arctic-bdl', 'cremad-1012', 'cremad-1011']
EMOTIONS = ['neutral', 'angry', 'disgusted', 'fearful', 'happy', 'sad']
STYLE_TEXTS = [
    "I think I've seen this before.", 'The airplane is almost full.', "I'm on my way to the meeting.",
    "We'll stop in a couple of minutes.", 'She opened the letter and read it twice.', 'The train leaves at seven tomorrow morning.',
    'He said that the house would be ready by spring.', 'They walked back along the river without a word.'
]

# The parameter transforms of the speaking modes (nn/js/transforms.js applyTransforms lists)
TRANSFORMS = {
    'plain': [],
    'whisper': [{'whisper': 1}],
    'soft': [{'effort': -8}],
    'shout': [{'effort': 10}, {'pitchScale': 1.25}]
}
TRANSFORM_TEXTS = STYLE_TEXTS[4:] + ['Come here at once, and bring the lantern!']


#
# Measures
#

# Score WAV files with the listener panel (UTMOS, ECAPA embedding, emotion2vec+) - {path: {mos, embedding, emotion}}
def listen(paths, device='auto'):
    process = subprocess.run([LISTENERS_PYTHON, '-I', os.path.join(ROOT, 'tools', 'listeners.py'), '--device', device,
                              '--files-from', '-'], input='\n'.join(paths), capture_output=True, text=True, check=True)
    return {item['file']: item for item in (json.loads(line) for line in process.stdout.splitlines() if line.strip())}


# Voice quality of a WAV file (Praat) - HNR (dB, voiced frames), CPPS (dB), the 4-8 kHz level relative to 0-4 kHz in
# voiced frames (dB), and above 8 kHz relative to 0-4 kHz when the file has it
def audio_quality(path, floor=60, ceiling=400):
    audio, rate = soundfile.read(path, dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 24000:
        audio = soxr.resample(audio, rate, 24000)
        rate = 24000
    sound = parselmouth.Sound(audio, sampling_frequency=rate)
    harmonicity = sound.to_harmonicity_cc(0.01, floor, 0.1, 1.0)
    hnr = harmonicity.values[harmonicity.values > -200]
    cepstrogram = call(sound, 'To PowerCepstrogram', 60, 0.002, 5000, 50)
    cpps = call(cepstrogram, 'Get CPPS', 'yes', 0.01, 0.001, 60, 330, 0.05, 'parabolic', 0.001, 0.05, 'Straight', 'Robust')
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array['frequency']
    size = 1024
    hop = 240
    window = numpy.hanning(size)
    bands = {'low': [], 'mid': [], 'high': []}
    frequency = numpy.fft.rfftfreq(size, 1 / rate)
    for ix, value in enumerate(f0):
        if value <= 0:
            continue
        center = int((pitch.xs()[ix]) * rate)
        start = center - size // 2
        if start < 0 or start + size > len(audio):
            continue
        power = numpy.abs(numpy.fft.rfft(audio[start:start + size] * window)) ** 2
        bands['low'].append(power[(frequency > 50) & (frequency < 4000)].sum())
        bands['mid'].append(power[(frequency >= 4000) & (frequency < 8000)].sum())
        bands['high'].append(power[(frequency >= 8000) & (frequency < 11000)].sum())
    low = max(1e-20, sum(bands['low']))
    voiced = f0 > 0
    level = 20 * math.log10(max(1e-9, float(numpy.sqrt(numpy.mean(audio ** 2)))))
    return {
        'hnr': round(float(hnr.mean()), 2) if len(hnr) else None, 'cpps': round(float(cpps), 2),
        'band4to8': round(10 * math.log10(max(1e-20, sum(bands['mid'])) / low), 2),
        'band8to11': round(10 * math.log10(max(1e-20, sum(bands['high'])) / low), 2),
        'f0Median': round(float(numpy.median(f0[voiced])), 1) if voiced.any() else None,
        'voicedFraction': round(float(voiced.mean()), 3), 'levelDb': round(level, 2)
    }


# Parameter statistics in strongly voiced frames (within 15 dB of the peak voicing) - median b1-b5 (Hz), ah and af
# relative to av (dB)
def parameter_quality(values):
    values = numpy.asarray(values)
    av = values[:, params.INDEX['av']]
    strong = av > av.max() - 15
    out = {f'b{number}': round(float(numpy.median(values[strong, params.INDEX[f'b{number}']])), 1) for number in range(1, 6)}
    out['ahRel'] = round(float(numpy.median(values[strong, params.INDEX['ah']] - av[strong])), 2)
    out['afRel'] = round(float(numpy.median(values[strong, params.INDEX['af']] - av[strong])), 2)
    return out


# The mean of a list of dicts' numeric values
def mean_dict(rows):
    keys = [key for key in rows[0] if isinstance(rows[0][key], (int, float))]
    return {key: round(float(numpy.mean([row[key] for row in rows if row.get(key) is not None])), 3) for key in keys}


# A group's summary - UTMOS mean and minimum, WER, and (with a reference embedding) the similarity of the mean embedding
def summarize(files, scores, transcripts, reference_mean=None):
    mos = [scores[item['wav']]['mos'] for item in files]
    errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in files)
    count = sum(len(words(item['text'])) for item in files)
    mean_embedding = numpy.mean([scores[item['wav']]['embedding'] for item in files], axis=0)
    out = {'count': len(files), 'mos': round(float(numpy.mean(mos)), 3), 'mosMin': round(float(numpy.min(mos)), 3),
           'wer': round(100 * errors / max(1, count), 2)}
    if reference_mean is not None:
        out['similarity'] = round(cosine(mean_embedding, reference_mean), 3)
    return out, mean_embedding


# Teacher-forced parameter frames of the model for ARCTIC sentences (the recorded durations) - [T, COUNT]
@torch.no_grad()
def forced_values(model, state, data, speaker, names):
    out = []
    speaker_id = state['speakers'].index(f'arctic-{speaker}')
    for name in names:
        entry = next(entry for entry in data.entries(None, [f'arctic-{speaker}'], include_excluded=True) if entry['sentence'] == name)
        arrays = data.arrays(entry)
        outputs = model(torch.tensor(arrays['tokens'])[None], torch.tensor(arrays['durations'])[None], torch.tensor([speaker_id]))
        out.append(model.parameters_from_unit(outputs['unit'])[0].numpy())
    return numpy.concatenate(out)


#
# The evaluation
#

def run_speak(model, batch, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    process = subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', model, '--batch', path], check=True,
                             capture_output=True, text=True, cwd=ROOT)
    return json.loads(process.stdout.strip().splitlines()[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.evalmulti', description='Evaluate the multi-speaker, multi-style model')
    parser.add_argument('checkpoint')
    parser.add_argument('--model', default=os.path.join(ROOT, 'data', 'nn', 'multi'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'nn', 'samples-multi'))
    parser.add_argument('--held-out', type=int, default=8, help='held-out sentences per speaker')
    parser.add_argument('--panel-count', type=int, default=16)
    parser.add_argument('--device', default='auto', help='the listeners\' device')
    parser.add_argument('--skip-synthesis', action='store_true', help='score the files already written')
    parser.add_argument('--style-count', type=int, default=len(STYLE_TEXTS), help='style texts per speaker and style')
    parser.add_argument('--transform-count', type=int, default=len(TRANSFORM_TEXTS), help='texts per transform')
    args = parser.parse_args(argv)
    with open(f'{args.model}.json', encoding='utf-8') as file:
        manifest = json.load(file)
    keys = [speaker['key'] for speaker in manifest['speakers']]
    data = ArcticData(os.path.join(DATA_DIR, 'multi'))
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        panel = json.load(file)['texts'][:args.panel_count]
    for sub in ('speakers', 'styles', 'transforms', 'originals'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # The speakers' texts - held-out sentences (test, then validation) and the panel
    items = []
    batch = []
    old_batches = {'slt': [], 'bdl': []}
    for sex, speakers in SAMPLE_SPEAKERS.items():
        for key in speakers:
            entries = data.entries('test', [key]) + data.entries('val', [key])
            for kind, texts in [('heldout', [(entry['sentence'], entry['text'], entry) for entry in entries[:args.held_out]]),
                                ('panel', [(item['id'], item['text'], None) for item in panel])]:
                for name, text, entry in texts:
                    base = os.path.join(args.out, 'speakers', f'{key}-{kind}-{name}')
                    item = {'speaker': key, 'sex': sex, 'kind': kind, 'id': name, 'text': text, 'wav': f'{base}.wav',
                            'params': f'{base}-params.json'}
                    if entry is not None:
                        item['original'] = os.path.join(args.out, 'originals', f'{key}-{name}.wav')
                        if not args.skip_synthesis:
                            audio = data.arrays(entry)['audio']
                            soundfile.write(item['original'], audio, data.index['sampleRate'], subtype='PCM_16')
                    old = key.split('-')[1] if key in ('arctic-slt', 'arctic-bdl') else None
                    if old:
                        item['old'] = f'{base}-old.wav'
                        item['oldParams'] = f'{base}-old-params.json'
                        old_batches[old].append({'text': text, 'wav': item['old'], 'params': item['oldParams'], 'seed': len(items) + 1})
                    items.append(item)
                    batch.append({'text': text, 'wav': item['wav'], 'params': item['params'], 'seed': len(items),
                                  'speaker': keys.index(key)})

    # The styles and the transforms
    style_items = []
    for key in STYLE_SPEAKERS:
        for emotion in EMOTIONS:
            for intensity in ([1.0] if emotion == 'neutral' else [1.5, 1.0, 0.5]):
                if intensity != 1.0 and key.startswith('cremad'):
                    continue
                for number, text in enumerate(STYLE_TEXTS[:args.style_count]):
                    wav = os.path.join(args.out, 'styles', f'{key}-{emotion}-{intensity:g}-{number}.wav')
                    style_items.append({'speaker': key, 'emotion': emotion, 'intensity': intensity, 'text': text, 'wav': wav,
                                        'known': number < 4})
                    batch.append({'text': text, 'wav': wav, 'seed': number + 1, 'speaker': keys.index(key),
                                  'style': {'emotion': emotion, 'intensity': intensity}})
    transform_items = []
    for key in STYLE_SPEAKERS[:4]:
        for mode, transform in TRANSFORMS.items():
            for number, text in enumerate(TRANSFORM_TEXTS[:args.transform_count]):
                wav = os.path.join(args.out, 'transforms', f'{key}-{mode}-{number}.wav')
                transform_items.append({'speaker': key, 'mode': mode, 'text': text, 'wav': wav})
                batch.append({'text': text, 'wav': wav, 'seed': number + 1, 'speaker': keys.index(key), 'transform': transform})

    timing = {}
    if not args.skip_synthesis:
        timing['multi'] = run_speak(args.model, batch, os.path.join(args.out, 'batch.json'))
        for voice, voice_batch in old_batches.items():
            timing[voice] = run_speak(os.path.join(ROOT, 'data', 'nn', voice), voice_batch, os.path.join(args.out, f'batch-{voice}.json'))
        print(f'synthesized: {json.dumps(timing)}', file=sys.stderr, flush=True)

    # Score
    paths = [item[key] for item in items for key in ('wav', 'old', 'original') if key in item]
    paths += [item['wav'] for item in style_items + transform_items]
    paths = list(dict.fromkeys(paths))
    scores = listen(paths, args.device)
    transcripts = transcribe_all([path for path in paths if '/styles/' not in path])
    print('scored', file=sys.stderr, flush=True)

    # Speakers
    results = {'model': args.model, 'checkpoint': args.checkpoint, 'speed': timing, 'speakers': {}, 'quality': {}, 'styles': {},
               'transforms': {}}
    for sex, speakers in SAMPLE_SPEAKERS.items():
        for key in speakers:
            mine = [item for item in items if item['speaker'] == key]
            held = [item for item in mine if item['kind'] == 'heldout']
            originals = [{'wav': item['original'], 'text': item['text']} for item in held]
            row = {'sex': sex}
            row['original'], original_mean = summarize(originals, scores, transcripts)
            for kind in ('heldout', 'panel'):
                group = [item for item in mine if item['kind'] == kind]
                row[kind], _ = summarize(group, scores, transcripts, original_mean)
                if group and 'old' in group[0]:
                    row[f'{kind}Old'], _ = summarize([{'wav': item['old'], 'text': item['text']} for item in group], scores,
                                                     transcripts, original_mean)
            # The emotion listener's top classes - the recordings against the neural voice (neutral style)
            row['emotionTop'] = {}
            for label, paths in [('original', [item['original'] for item in held]), ('heldout', [item['wav'] for item in held]),
                                 ('panel', [item['wav'] for item in mine if item['kind'] == 'panel'])]:
                tops = [max(EMOTIONS, key=lambda name, path=path: scores[path]['emotion'].get(name, 0)) for path in paths]
                row['emotionTop'][label] = {name: tops.count(name) for name in sorted(set(tops))}
            floor, ceiling = (60, 250) if sex == 'male' else (100, 400)
            row['quality'] = {
                'original': mean_dict([audio_quality(item['original'], floor, ceiling) for item in held]),
                'neural': mean_dict([audio_quality(item['wav'], floor, ceiling) for item in held]),
                'neuralParameters': parameter_quality(numpy.concatenate([params.read_params(item['params']) for item in mine]))
            }
            if held and 'old' in held[0]:
                row['quality']['old'] = mean_dict([audio_quality(item['old'], floor, ceiling) for item in held])
                old_values = numpy.concatenate([params.read_params(item['oldParams']) for item in mine])
                row['quality']['oldParameters'] = parameter_quality(old_values)  # with the old voices' exported 0.6 bandwidth scale
            results['speakers'][key] = row
            print(f'{key}: {json.dumps({kind: row[kind] for kind in ("original", "heldout", "panel")})}', file=sys.stderr, flush=True)

    # The over-smoothing check against the fits (teacher-forced, the fits' ARCTIC sentences)
    model, state = load_model(args.checkpoint)
    arctic = ArcticData(os.path.join(DATA_DIR, 'multi'))
    for speaker in ('slt', 'bdl'):
        names = [f'arctic_a{number:04d}' for number in range(1, 7)]
        fits = numpy.concatenate([params.read_params(os.path.join(FIT_DIR, speaker, f'{name}.json')) for name in names])
        floor, ceiling = (60, 250) if speaker == 'bdl' else (100, 400)
        results['quality'][speaker] = {
            'fitParameters': parameter_quality(fits),
            'modelParameters': parameter_quality(forced_values(model, state, arctic, speaker, names)),
            'fitAudio': mean_dict([audio_quality(os.path.join(FIT_DIR, speaker, f'{name}-fitted.wav'), floor, ceiling) for name in names]),
            'originalAudio': mean_dict([audio_quality(os.path.join(FIT_DIR, speaker, f'{name}-original.wav'), floor, ceiling)
                                        for name in names])
        }
        old_model, old_state = load_model(os.path.join(ROOT, 'build', 'nn', 'final', f'{speaker}.pt'))
        with torch.no_grad():
            old_values = []
            old_data = ArcticData()
            for name in names:
                entry = next(entry for entry in old_data.entries(None, [speaker], include_excluded=True) if entry['sentence'] == name)
                arrays = old_data.arrays(entry)
                outputs = old_model(torch.tensor(arrays['tokens'])[None], torch.tensor(arrays['durations'])[None])
                old_values.append(old_model.parameters_from_unit(outputs['unit'])[0].numpy())
        results['quality'][speaker]['oldModelParameters'] = parameter_quality(numpy.concatenate(old_values))
        del old_state

    # Styles - the emotion listener's top class among our six emotions
    for key in STYLE_SPEAKERS:
        for emotion in EMOTIONS:
            for intensity in (1.5, 1.0, 0.5):
                group = [item for item in style_items if item['speaker'] == key and item['emotion'] == emotion and
                         item['intensity'] == intensity]
                if not group:
                    continue
                probabilities = [scores[item['wav']]['emotion'] for item in group]
                top = [max(EMOTIONS, key=lambda name, probs=probs: probs.get(name, 0)) for probs in probabilities]
                results['styles'].setdefault(key, {})[f'{emotion}@{intensity:g}'] = {
                    'recognized': round(sum(1 for name in top if name == emotion) / len(top), 3),
                    'meanProbability': round(float(numpy.mean([probs.get(emotion, 0) for probs in probabilities])), 3),
                    'top': {name: top.count(name) for name in sorted(set(top))},
                    'mos': round(float(numpy.mean([scores[item['wav']]['mos'] for item in group])), 3)
                }
    summary = {}
    for group_name, speakers in [('narrators', STYLE_SPEAKERS[:4]), ('actors', STYLE_SPEAKERS[4:])]:
        for emotion in EMOTIONS:
            for intensity in (1.5, 1.0, 0.5):
                cells = [results['styles'][key].get(f'{emotion}@{intensity:g}') for key in speakers]
                cells = [cell for cell in cells if cell]
                if cells:
                    summary.setdefault(group_name, {})[f'{emotion}@{intensity:g}'] = {
                        'recognized': round(float(numpy.mean([cell['recognized'] for cell in cells])), 3),
                        'meanProbability': round(float(numpy.mean([cell['meanProbability'] for cell in cells])), 3),
                        'mos': round(float(numpy.mean([cell['mos'] for cell in cells])), 3)}
    results['stylesSummary'] = summary

    # Transforms
    for mode in TRANSFORMS:
        group = [item for item in transform_items if item['mode'] == mode]
        row, _ = summarize(group, scores, transcripts)
        emotions = [scores[item['wav']]['emotion'] for item in group]
        row['emotion'] = {name: round(float(numpy.mean([probs.get(name, 0) for probs in emotions])), 3) for name in EMOTIONS}
        row['audio'] = mean_dict([audio_quality(item['wav'], 60, 500) for item in group])
        results['transforms'][mode] = row

    with open(os.path.join(args.out, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=1)
    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as file:
        json.dump({'speakers': items, 'styles': style_items, 'transforms': transform_items}, file, indent=1)
    json.dump({key: value for key, value in results.items() if key not in ('styles',)}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
