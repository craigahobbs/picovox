# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Listening samples and the controllability demonstration - the exported neural voices (data/nn/<voice>.*) spoken by the
browser path (nn/js/speak.js) at 44.1 kHz, 16-bit, into build/nn/samples/:

- <voice>-<text id>.wav - the first --count panel texts (tools/panel-texts.json), and rule/<voice>-<text id>.wav, the
  rule voice of the same sex (lib/speak.js) on the same texts, for comparison
- control/<voice>-<transform>.wav - one text with the interpretable transforms (nn/js/transforms.js) applied to the
  network's predicted parameters: pitch scale, tract-length scale, whisper, and effort

Each transform's effect is measured with Praat against the untransformed rendering (F0 median, the voiced fraction,
the formant medians), into control/results.json.

    build/venv-nn/bin/python -B -m nn.samples --voices slt bdl
"""

import argparse
import json
import os
import subprocess
import sys

import numpy
import parselmouth
import soundfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEX = {'slt': 'female', 'bdl': 'male'}

# The transforms demonstrated (nn/js/transforms.js applyTransforms lists)
TRANSFORMS = {
    'plain': [],
    'pitch-1.3': [{'pitchScale': 1.3}],
    'pitch-0.75': [{'pitchScale': 0.75}],
    'tract-1.15': [{'tractScale': 1.15}],
    'tract-0.87': [{'tractScale': 0.87}],
    'whisper': [{'whisper': 1}],
    'effort+8': [{'effort': 8}],
    'effort-8': [{'effort': -8}],
    'child-like': [{'pitchScale': 1.6}, {'tractScale': 1.2}]
}
CONTROL_TEXT = 'The stale smell of old beer lingers. A cold dip restores health and zest.'


# Praat measures of a WAV file - F0 median, voiced fraction (of frames above the level floor), F1-F3 medians in voiced
# frames
def measure(path, floor=60, ceiling=600):
    audio, rate = soundfile.read(path)
    sound = parselmouth.Sound(audio, sampling_frequency=rate)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    intensity = sound.to_intensity(minimum_pitch=floor, time_step=0.01)
    formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=6000)
    times = pitch.xs()
    f0 = pitch.selected_array['frequency']
    level = numpy.array([intensity.get_value(time) for time in times])
    active = numpy.nan_to_num(level, nan=-100) > numpy.nanmax(level) - 30
    voiced = (f0 > 0) & active
    formants = {}
    for number in (1, 2, 3):
        values = numpy.array([formant.get_value_at_time(number, time) for time in times[voiced]])
        values = values[~numpy.isnan(values)]
        formants[f'f{number}'] = round(float(numpy.median(values)), 1) if len(values) else None
    return {'f0Median': round(float(numpy.median(f0[voiced])), 1) if voiced.any() else None,
            'voicedFraction': round(float(voiced.sum() / max(1, active.sum())), 3), **formants}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.samples', description='Listening samples and transforms')
    parser.add_argument('--voices', nargs='+', default=['slt', 'bdl'])
    parser.add_argument('--models', default=os.path.join(ROOT, 'data', 'nn'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'nn', 'samples'))
    parser.add_argument('--count', type=int, default=6)
    args = parser.parse_args(argv)
    for sub in ('', 'rule', 'control'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        texts = json.load(file)['texts']
    # Six texts of every kind - the Harvard sentences, two narrative passages, a question, an exclamation
    kinds = {}
    for item in texts:
        kinds.setdefault(item['kind'], []).append(item)
    chosen = kinds['harvard'][:2] + kinds['narrative'][:2] + kinds['question'][:1] + kinds['exclamation'][:1]
    chosen = chosen[:args.count]

    results = {}
    for voice in args.voices:
        batch = [{'text': item['text'], 'wav': os.path.join(args.out, f'{voice}-{item["id"]}.wav'), 'seed': 1} for item in chosen]
        batch += [{'text': CONTROL_TEXT, 'wav': os.path.join(args.out, 'control', f'{voice}-{name}.wav'), 'transform': transform,
                   'seed': 1} for name, transform in TRANSFORMS.items()]
        batch_path = os.path.join(args.out, f'{voice}-batch.json')
        with open(batch_path, 'w', encoding='utf-8') as file:
            json.dump(batch, file)
        subprocess.run(['node', 'nn/js/speak.js', '-m', os.path.join(args.models, voice), '--batch', batch_path],
                       check=True, cwd=ROOT, capture_output=True)
        os.remove(batch_path)
        rule = [{'text': item['text'], 'klatt': os.path.join(args.out, 'rule', f'{voice}-{item["id"]}.wav'), 'seed': 1} for item in chosen]
        with open(batch_path, 'w', encoding='utf-8') as file:
            json.dump(rule, file)
        subprocess.run(['node', 'nn/js/speakparams.js', '-v', SEX[voice], '--batch', batch_path], check=True, cwd=ROOT,
                       capture_output=True)
        os.remove(batch_path)
        results[voice] = {name: measure(os.path.join(args.out, 'control', f'{voice}-{name}.wav')) for name in TRANSFORMS}
    with open(os.path.join(args.out, 'control', 'results.json'), 'w', encoding='utf-8') as file:
        json.dump({'text': CONTROL_TEXT, 'transforms': TRANSFORMS, 'measures': results}, file, indent=1)
    json.dump(results, sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
