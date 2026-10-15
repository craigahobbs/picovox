# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Parity and speed of the browser inference - nn/js/model.js (the exported 8-bit weights, data/nn/<voice>.*) against
PyTorch with the same dequantized weights (nn/export.py's quantized_model), and the quantization's own effect (PyTorch
quantized against the float model):

- durations - the share of tokens with the same predicted duration
- frames - the largest and mean absolute difference of each parameter (normalized 0-1 as the network outputs them, and
  in its units), over the frames of utterances whose durations agree
- audio - the log-spectral distance between the renderings (nn/dsynth.py, counter noise, same seed) of the quantized
  and float models' frames, against the floor of two noise seeds
- speed - nn/js/speak.js's real-time factors (network, rendering at 44.1 kHz)

    build/venv-nn/bin/python -B -m nn.parity build/nn/runs/slt/best.pt data/nn/slt
    build/venv-nn/bin/python -B -m nn.parity build/nn/runs/multi-b/best.pt data/nn/multi --speaker libritts-6454 [--style happy]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy
import torch

from . import params, spectral
from .synthesize import Voice, render


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The evaluation texts - the panel texts and the held-out ARCTIC test sentences
def texts(count=None):
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        panel = [item['text'] for item in json.load(file)['texts']]
    with open(os.path.join(ROOT, 'data', 'nn', 'arctic-split.json'), encoding='utf-8') as file:
        split = json.load(file)
    test = [item['text'] for item in split['test']] if isinstance(split['test'], list) and split['test'] and \
        isinstance(split['test'][0], dict) else []
    out = panel + test
    return out[:count] if count else out


# A voice's parameter frames for sentences with given durations (teacher-forced) - [T, COUNT] with the end frame
def forced_values(voice, parts):
    chunks = []
    for part in parts:
        tokens = torch.tensor([[voice.remap.get(token, token) for token in part['ids']]])
        with torch.no_grad():
            outputs = voice.model(tokens, torch.tensor([part['durations']]), **voice.condition())
            chunks.append(voice.model.parameters_from_unit(outputs['unit'])[0].numpy())
    values = numpy.concatenate(chunks)
    return params.bandwidth_scale(numpy.concatenate([values, values[-1:]]), voice.bandwidth_scale)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.parity', description='nn/js/model.js against PyTorch')
    parser.add_argument('checkpoint')
    parser.add_argument('model', help='the exported voice (data/nn/<voice>, without extension)')
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--audio-count', type=int, default=8)
    parser.add_argument('--speaker', default=None, help='the speaker of a multi-speaker model (a key or an id)')
    parser.add_argument('--blend', default=None, help='a multi-speaker model\'s speaker blend (JSON {id: weight})')
    parser.add_argument('--style', default=None, help='a multi-speaker model\'s style (an emotion)')
    parser.add_argument('--intensity', type=float, default=1.0)
    args = parser.parse_args(argv)
    blend = json.loads(args.blend) if args.blend else None
    items = texts(args.count)
    with open(f'{args.model}.json', encoding='utf-8') as file:
        scale = json.load(file)['inference'].get('bandwidthScale', 1.0)
    voice = {'speaker': args.speaker, 'blend': blend, 'emotion': args.style, 'intensity': args.intensity}
    quantized = Voice(args.checkpoint, quantized=True, bandwidth_scale=scale, **voice)
    floating = Voice(args.checkpoint, quantized=False, bandwidth_scale=scale, **voice)
    js_voice = []
    if quantized.multi:
        js_voice = ['--blend', json.dumps(blend)] if blend else ['--speaker', str(quantized.speaker)]
        if args.style:
            js_voice += ['--style', args.style, '--intensity', str(args.intensity)]

    with tempfile.TemporaryDirectory() as directory:
        batch = [{'text': text, 'params': os.path.join(directory, f'{ix}.json')} for ix, text in enumerate(items)]
        with open(os.path.join(directory, 'batch.json'), 'w', encoding='utf-8') as file:
            json.dump(batch, file)
        subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', args.model, '--batch',
                        os.path.join(directory, 'batch.json'), '--frames-only', *js_voice], check=True, capture_output=True, cwd=ROOT)
        js_frames = []
        js_durations = []
        for item in batch:
            with open(item['params'], encoding='utf-8') as file:
                data = json.load(file)
            js_frames.append(numpy.array(data['frames'], dtype=numpy.float64))
            js_durations.append([duration for sentence in data['sentences'] for duration in sentence['durations']])

        # Speed - the panel texts rendered at 44.1 kHz
        speed_batch = [{'text': text, 'wav': os.path.join(directory, f'speed-{ix}.wav')} for ix, text in enumerate(items[:16])]
        with open(os.path.join(directory, 'speed.json'), 'w', encoding='utf-8') as file:
            json.dump(speed_batch, file)
        process = subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', args.model, '--batch',
                                  os.path.join(directory, 'speed.json'), *js_voice], check=True, capture_output=True, text=True, cwd=ROOT)
        speed = json.loads(process.stdout.strip().splitlines()[-1])

    same_tokens = 0
    total_tokens = 0
    differences = []
    quant_same_tokens = 0
    quant_frame_ratio = []
    quant_differences = []
    lsd_quant = []
    lsd_floor = []
    for ix, text in enumerate(items):
        values, parts = quantized.frames(text)
        durations = [duration for part in parts for duration in part['durations']]
        same_tokens += sum(1 for first, second in zip(durations, js_durations[ix]) if first == second)
        total_tokens += len(durations)
        if durations == js_durations[ix] and len(values) == len(js_frames[ix]):
            differences.append(numpy.abs(params.normalize(values) - params.normalize(js_frames[ix])))

        # The quantization - durations, and the frames with the float model's durations (teacher-forced)
        float_values, float_parts = floating.frames(text)
        float_durations = [duration for part in float_parts for duration in part['durations']]
        quant_same_tokens += sum(1 for first, second in zip(durations, float_durations) if first == second)
        quant_frame_ratio.append(len(values) / len(float_values))
        forced = forced_values(quantized, float_parts)
        quant_differences.append(numpy.abs(params.normalize(forced) - params.normalize(float_values)))
        if ix < args.audio_count:
            # The quantized model's rendering against the float model's, and the floor - the float model's frames
            # with another noise seed
            rate = 16000
            float_audio = render(float_values, floating.eq, rate, seed=1)
            lsd_quant.append(spectral.log_spectral_distance(float_audio, render(forced, quantized.eq, rate, seed=1), rate))
            lsd_floor.append(spectral.log_spectral_distance(float_audio, render(float_values, floating.eq, rate, seed=2), rate))
    stacked = numpy.concatenate(differences) if differences else numpy.zeros((1, params.COUNT))
    quant_stacked = numpy.concatenate(quant_differences) if quant_differences else numpy.zeros((1, params.COUNT))
    result = {
        'texts': len(items),
        'jsVsTorch': {
            'durationsSame': round(same_tokens / max(1, total_tokens), 5), 'utterancesCompared': len(differences),
            'maxAbsNormalized': float(stacked.max()), 'meanAbsNormalized': float(stacked.mean()),
            'worstParameter': params.NAMES[int(stacked.max(axis=0).argmax())]
        },
        'quantizedVsFloat': {
            'durationsSame': round(quant_same_tokens / max(1, total_tokens), 5),
            'frameCountRatio': round(float(numpy.mean(quant_frame_ratio)), 5), 'maxAbsNormalized': float(quant_stacked.max()),
            'meanAbsNormalized': float(quant_stacked.mean()),
            'meanAbsByParameter': {name: round(float(value), 5) for name, value in zip(params.NAMES, quant_stacked.mean(axis=0))},
            'lsdQuantized': round(float(numpy.mean(lsd_quant)), 3) if lsd_quant else None,
            'lsdSeedFloor': round(float(numpy.mean(lsd_floor)), 3)
        },
        'speed': speed
    }
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
