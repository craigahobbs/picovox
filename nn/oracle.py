# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
What limits the neural voice - the held-out ARCTIC test sentences rendered with parts of the prediction replaced by
the recording's own (oracle) values, scored by the listener panel and Whisper:

- oracle-timing-f0 - the recorded durations (teacher-forced) and the recorded (Praat) F0: only the network's spectral
  envelope, levels, and source shape are predicted
- oracle-timing - the recorded durations, the predicted F0
- predicted - everything predicted (the voice as it speaks; as nn/js/speak.js, rendered in PyTorch)
- original - the recordings

All are rendered by nn/dsynth.py at 44.1 kHz with the counter noise (as nn/js/render.js), from the exported
(quantized) weights.

    build/venv-nn/bin/python -B -m nn.oracle build/nn/runs/slt/best.pt [--speaker slt] [--out build/nn/oracle/slt]
"""

import argparse
import json
import os
import sys

import numpy
import soundfile
import torch

from . import params
from .data import ArcticData
from .evaluate import ARCTIC
from .panel import cosine, edit_distance, listen, transcribe_all, words
from .synthesize import Voice, render
from .losses import f0_track


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.oracle', description='Oracle prosody against predicted')
    parser.add_argument('checkpoint')
    parser.add_argument('--speaker', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--rate', type=int, default=44100)
    args = parser.parse_args(argv)
    voice = Voice(args.checkpoint, quantized=True, speaker=args.speaker)
    speaker = voice.state['speakers'][voice.speaker]
    out = args.out or os.path.join(ROOT, 'build', 'nn', 'oracle', speaker)
    os.makedirs(out, exist_ok=True)
    data = ArcticData()
    channels = data.index['channels']
    settings = data.index['speakers'][speaker]
    entries = data.entries('test', [speaker])[:args.count]
    groups = {'oracle-timing-f0': [], 'oracle-timing': [], 'predicted': [], 'original': []}
    texts = {}
    model = voice.model
    for entry in entries:
        arrays = data.arrays(entry)
        tokens = torch.tensor([[voice.remap.get(int(token), int(token)) for token in arrays['tokens']]])
        durations = torch.tensor([arrays['durations']])
        with torch.no_grad():
            unit = model(tokens, durations, torch.tensor([voice.speaker]))['unit']
            values = model.parameters_from_unit(unit)[0].numpy()
        values = numpy.concatenate([values, values[-1:]])
        tracks = arrays['frames']
        f0 = f0_track(tracks[:, channels.index('f0')], tracks[:, channels.index('voiced')] > 0.5, entry['frameCount'],
                      settings['pitchFloor'], settings['pitchCeiling'])
        oracle_f0 = values.copy()
        oracle_f0[:, params.INDEX['f0']] = f0
        predicted, _ = voice.frames(entry['text'])
        name = entry['sentence']
        for group, frames in [('oracle-timing-f0', oracle_f0), ('oracle-timing', values), ('predicted', predicted)]:
            path = os.path.join(out, f'{name}-{group}.wav')
            soundfile.write(path, render(frames, voice.eq, args.rate, seed=1), args.rate, subtype='PCM_16')
            groups[group].append(path)
        groups['original'].append(os.path.join(ARCTIC, f'cmu_us_{speaker}_arctic', 'wav', f'{name}.wav'))
        for group in groups:
            texts[groups[group][-1]] = entry['text']

    paths = [path for files in groups.values() for path in files]
    scores = listen(paths, 'auto')
    transcripts = transcribe_all(paths)
    original_mean = numpy.mean([scores[path]['embedding'] for path in groups['original']], axis=0)
    results = {'checkpoint': args.checkpoint, 'speaker': speaker, 'count': len(entries), 'groups': {}}
    for group, files in groups.items():
        errors = sum(edit_distance(words(texts[path]), words(transcripts[path])) for path in files)
        count = sum(len(words(texts[path])) for path in files)
        mean = numpy.mean([scores[path]['embedding'] for path in files], axis=0)
        results['groups'][group] = {
            'mos': round(float(numpy.mean([scores[path]['mos'] for path in files])), 3),
            'mosMin': round(float(numpy.min([scores[path]['mos'] for path in files])), 3),
            'wer': round(100 * errors / count, 2), 'similarity': round(cosine(mean, original_mean), 3),
            'pairSimilarity': round(float(numpy.mean([cosine(scores[path]['embedding'], scores[original]['embedding'])
                                                      for path, original in zip(files, groups['original'])])), 3)
        }
    with open(os.path.join(out, 'results.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    json.dump(results, sys.stdout, indent=2)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
