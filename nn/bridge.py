# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The rule voice through the parameter set - how close nn/js/render.js and nn/dsynth.py, given lib/speak.js's frames
converted by nn/js/bridge.js, sound to lib/klatt.js. For each voice, the listener-panel texts (tools/panel-texts.json)
are spoken and rendered four ways:

- klatt - lib/speak.js's own audio (lib/klatt.js)
- klatt-alt - lib/klatt.js on the same frames with another seed - the floor of the distances (realizations only)
- render - nn/js/render.js at 44.1 kHz
- dsynth - nn/dsynth.py at --dsynth-rate (22.05 kHz), the training synthesizer

and compared by log-spectral distance (nn/spectral.py, against klatt) and by the listener panel (tools/listeners.py -
UTMOS, and ECAPA speaker similarity to klatt).

    build/venv-nn/bin/python -m nn.bridge [--voices male female] [--count 16] [--no-listeners]

Results go to build/nn/bridge/ (audio, parameters, results.json).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy
import soundfile
import soxr
import torch

from . import dsynth, params, spectral


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTENERS_PYTHON = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')


# Score WAV files with the listener panel (UTMOS and ECAPA embeddings) - {path: {mos, embedding}}
def listen(paths, device='auto'):
    process = subprocess.run(
        [LISTENERS_PYTHON, '-I', os.path.join(ROOT, 'tools', 'listeners.py'), '--no-emotion', '--device', device, '--files-from', '-'],
        input='\n'.join(paths), capture_output=True, text=True, check=True
    )
    results = {}
    for line in process.stdout.splitlines():
        if line.strip():
            item = json.loads(line)
            results[item['file']] = item
    return results


# The cosine similarity of two vectors
def cosine(first, second):
    first, second = numpy.asarray(first), numpy.asarray(second)
    return float(first @ second / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))


# Group listener scores - mean MOS, and the mean embedding's similarity to a reference group's
def summarize_listeners(scores, groups, reference):
    summary = {}
    means = {name: numpy.mean([scores[path]['embedding'] for path in paths], axis=0) for name, paths in groups.items()}
    for name, paths in groups.items():
        mos = [scores[path]['mos'] for path in paths]
        pairs = [cosine(scores[path]['embedding'], scores[ref]['embedding']) for path, ref in zip(paths, groups[reference])]
        summary[name] = {
            'mos': round(float(numpy.mean(mos)), 3), 'mosMin': round(float(numpy.min(mos)), 3),
            'similarity': round(cosine(means[name], means[reference]), 3), 'pairSimilarity': round(float(numpy.mean(pairs)), 3)
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.bridge', description='The rule voice through the parameter set')
    parser.add_argument('--voices', nargs='+', default=['male', 'female'])
    parser.add_argument('--count', type=int, default=16, help='the number of panel texts')
    parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'nn', 'bridge'))
    parser.add_argument('--dsynth-rate', type=int, default=22050)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--ob1', type=float, default=0.5, help='the open-phase F1 bandwidth weight (nn/js/bridge.js)')
    parser.add_argument('--no-listeners', action='store_true')
    args = parser.parse_args(argv)

    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        texts = json.load(file)['texts'][:args.count]
    synth = dsynth.DSynth(args.dsynth_rate).to(args.device)
    results = {'voices': {}, 'dsynthRate': args.dsynth_rate, 'ob1': args.ob1}
    all_groups = {}
    for voice in args.voices:
        directory = os.path.join(args.out, voice)
        os.makedirs(directory, exist_ok=True)
        items = []
        for text in texts:
            stem = os.path.join(directory, text['id'])
            items.append({'text': text['text'], 'params': f'{stem}.json', 'klatt': f'{stem}-klatt.wav',
                          'klattAlt': f'{stem}-klatt-alt.wav', 'render': f'{stem}-render.wav'})
        batch_path = os.path.join(directory, 'batch.json')
        with open(batch_path, 'w', encoding='utf-8') as file:
            json.dump(items, file)
        process = subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'speakparams.js'), '-v', voice, '--ob1', str(args.ob1),
                                  '--batch', batch_path], capture_output=True, text=True, check=True)
        timing = json.loads(process.stdout.strip().splitlines()[-1])

        # The training synthesizer's rendering
        dsynth_seconds = 0
        for item in items:
            values = torch.tensor(params.read_params(item['params']), dtype=torch.float32, device=args.device)
            start = time.time()
            with torch.no_grad():
                audio = synth(values, seed=7).cpu().numpy()
            if args.device == 'mps':
                torch.mps.synchronize()
            dsynth_seconds += time.time() - start
            item['dsynth'] = item['render'].replace('-render.wav', '-dsynth.wav')
            soundfile.write(item['dsynth'], audio * 0.89 / max(1e-9, numpy.abs(audio).max()), args.dsynth_rate)

        # Log-spectral distances from klatt
        distances = {'klatt-alt': [], 'render': [], 'dsynth': []}
        ltas = []
        for item in items:
            klatt, rate = soundfile.read(item['klatt'])
            for name, key in [('klatt-alt', 'klattAlt'), ('render', 'render')]:
                other, _ = soundfile.read(item[key])
                distances[name].append(spectral.log_spectral_distance(klatt, other, rate))
            other, _ = soundfile.read(item['render'])
            ltas.append(spectral.ltas_difference(klatt, other, rate, high=11025))
            dsynth_audio, _ = soundfile.read(item['dsynth'])
            klatt_low = soxr.resample(klatt, rate, args.dsynth_rate)
            distances['dsynth'].append(spectral.log_spectral_distance(klatt_low, dsynth_audio, args.dsynth_rate))
        results['voices'][voice] = {
            'lsd': {name: round(float(numpy.mean(values)), 3) for name, values in distances.items()},
            'ltasRenderMinusKlatt': [round(float(value), 2) for value in numpy.mean(ltas, axis=0)[::8]],
            'audioSeconds': round(timing['audioSeconds'], 2), 'speakSeconds': round(timing['speakSeconds'], 2),
            'renderSeconds': round(timing['renderSeconds'], 2), 'dsynthSeconds': round(dsynth_seconds, 2)
        }
        all_groups[voice] = {name: [item[key] for item in items] for name, key in
                             [('klatt', 'klatt'), ('klatt-alt', 'klattAlt'), ('render', 'render'), ('dsynth', 'dsynth')]}

    if not args.no_listeners:
        paths = [path for groups in all_groups.values() for group in groups.values() for path in group]
        scores = listen(paths)
        for voice, groups in all_groups.items():
            results['voices'][voice]['listeners'] = summarize_listeners(scores, groups, 'klatt')

    with open(os.path.join(args.out, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    json.dump(results, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
