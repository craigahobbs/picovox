# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Export - a trained model (nn/train.py checkpoint) to the browser's format, data/nn/<voice>.bin and data/nn/<voice>.json,
which nn/js/model.js loads.

    build/venv-nn/bin/python -B -m nn.export build/nn/runs/slt/best.pt --voice slt

The weights are quantized to 8 bits: every weight matrix, convolution kernel, and embedding table as int8 with one
float32 scale per output channel (row) - symmetric, the scale the row's largest magnitude / 127; the biases and layer
norm parameters, a small fraction of the model, stay float32. The .bin is the tensors' data back to back (each
tensor's int8 data, then its float32 scales; or its float32 data), aligned to 4 bytes; the .json manifest lists each
tensor's name, shape, encoding, and byte offsets, with the model config, the vocabulary remap, the channel equalizer,
and the inference defaults (the pause threshold, and the formant bandwidth scale - a sharpening of the network's
over-smoothed bandwidths, chosen on the validation sentences).

The quantization is the only lossy step - quantized_model() gives the PyTorch model with the exported weights
(dequantized), which nn/js/model.js must match.
"""

import argparse
import collections
import copy
import json
import os
import sys

import numpy
import torch

from . import params
from .data import ArcticData
from .model import Model, parameter_count


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'data', 'nn')
FORMAT_VERSION = 1

# Tensors smaller than this stay float32
MIN_QUANTIZED = 1024


# Load a checkpoint's model (on the CPU, in eval mode) - (model, checkpoint)
def load_model(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    model = Model(state['config'], priors=state.get('priors'))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state


# The exported tensors of a model's state - {name: tensor} - for one speaker of a multi-speaker model (its rows of the
# speaker embeddings); the channel equalizer is exported in the manifest instead
def exported_tensors(model, speaker=0):
    tensors = collections.OrderedDict()
    for name, tensor in model.state_dict().items():
        if name == 'eq_free':
            continue
        if name in ('speaker.weight', 'decoder_speaker.weight'):
            tensor = tensor[speaker:speaker + 1]
        tensors[name] = tensor.detach().cpu().float()
    return tensors


# Quantize a tensor per output channel (dim 0) - (int8 values, float32 scales)
def quantize(tensor):
    rows = tensor.reshape(tensor.shape[0], -1).numpy().astype(numpy.float64)
    scales = numpy.abs(rows).max(axis=1) / 127
    scales[scales == 0] = 1.0
    values = numpy.clip(numpy.round(rows / scales[:, None]), -127, 127).astype(numpy.int8)
    return values, scales.astype(numpy.float32)


# Whether a tensor is quantized
def is_quantized(tensor):
    return tensor.dim() >= 2 and tensor.numel() >= MIN_QUANTIZED


# A copy of a model with its weights quantized and dequantized - the exported model, in PyTorch
def quantized_model(model):
    model = copy.deepcopy(model)
    state = model.state_dict()
    for name, tensor in exported_tensors(model).items():
        if is_quantized(tensor):
            values, scales = quantize(tensor)
            state[name] = torch.tensor(values.astype(numpy.float32) * scales[:, None]).reshape(tensor.shape)
    model.load_state_dict(state)
    return model


# The token remap for inference - tokens rare or unseen in training to the nearest well-trained token: phrase ends to
# <period>, a vowel's rare stress to its commonest stress
def token_remap(vocab, counts, minimum=10):
    remap = {}
    for index, token in enumerate(vocab):
        if counts.get(index, 0) >= minimum or token in ('<pad>',):
            continue
        if token in ('<question>', '<wh>', '<exclamation>'):
            remap[token] = '<period>'
        elif token[-1:].isdigit():
            variants = [f'{token[:-1]}{stress}' for stress in range(3)]
            best = max(variants, key=lambda variant: counts.get(vocab.index(variant), 0))
            if counts.get(vocab.index(best), 0) >= minimum:
                remap[token] = best
        elif token == '<unk>':
            remap[token] = '_'
    return remap


# The training token counts of a checkpoint's speakers
def training_token_counts(speakers):
    data = ArcticData()
    counts = collections.Counter()
    for entry in data.entries('train', speakers):
        counts.update(data.arrays(entry)['tokens'].tolist())
    return counts, data.index['vocab']


# Export a model - writes <out>/<voice>.bin and <voice>.json, returns the manifest
def export(model, state, voice, out_dir=OUT_DIR, extra=None, speaker=0, bandwidth_scale=1.0):
    os.makedirs(out_dir, exist_ok=True)
    chunks = []
    offset = 0
    tensors = []
    for name, tensor in exported_tensors(model, speaker).items():
        entry = {'name': name, 'shape': list(tensor.shape)}
        if is_quantized(tensor):
            values, scales = quantize(tensor)
            data = values.tobytes()
            entry.update({'encoding': 'int8', 'offset': offset, 'scaleOffset': offset + _aligned(len(data))})
            chunks.append(data + b'\0' * (_aligned(len(data)) - len(data)))
            chunks.append(scales.tobytes())
            offset += _aligned(len(data)) + len(scales.tobytes())
        else:
            data = tensor.numpy().astype('<f4').tobytes()
            entry.update({'encoding': 'float32', 'offset': offset})
            chunks.append(data)
            offset += len(data)
        tensors.append(entry)
    binary = b''.join(chunks)
    counts, vocab = training_token_counts(state['speakers'])
    with torch.no_grad():
        eq = [round(float(value), 3) for value in model.eq(torch.tensor([speaker]))[0]]
    manifest = {
        'format': 'voice-synth-nn', 'version': FORMAT_VERSION, 'voice': voice, 'speaker': state['speakers'][speaker],
        'trainingSpeakers': state['speakers'],
        'description': f'The neural voice trained on CMU ARCTIC {"/".join(state["speakers"])} (nn/train.py) - tokens to '
                       'durations to synthesizer parameter frames (nn/params.py), rendered by nn/js/render.js',
        'config': model.config, 'parameterNames': params.NAMES, 'frameRate': params.FRAME_RATE,
        'eq': eq, 'remap': token_remap(vocab, counts), 'vocab': vocab,
        'inference': {'pauseThreshold': 0.5, 'bandwidthScale': bandwidth_scale, 'f0Min': None, 'f0Max': None},
        'parameterCount': parameter_count(model), 'bytes': len(binary), 'tensors': tensors,
        'training': {'step': state.get('step'), 'val': state.get('val')}, **(extra or {})
    }
    settings = ArcticData().index['speakers'].get(state['speakers'][speaker])
    if settings:
        manifest['inference']['f0Min'] = settings['pitchFloor']
        manifest['inference']['f0Max'] = settings['pitchCeiling']
    with open(os.path.join(out_dir, f'{voice}.bin'), 'wb') as file:
        file.write(binary)
    with open(os.path.join(out_dir, f'{voice}.json'), 'w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=1)
    return manifest


def _aligned(size):
    return (size + 3) // 4 * 4


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.export', description='Export a model for nn/js/model.js')
    parser.add_argument('checkpoint')
    parser.add_argument('--voice', required=True)
    parser.add_argument('--out', default=OUT_DIR)
    parser.add_argument('--speaker', default=None, help='the speaker of a multi-speaker model (default: the first)')
    parser.add_argument('--bandwidth-scale', type=float, default=1.0,
                        help='the default formant bandwidth scale at inference (nn/js/model.js speakFrames)')
    args = parser.parse_args(argv)
    model, state = load_model(args.checkpoint)
    speaker = state['speakers'].index(args.speaker) if args.speaker else 0
    manifest = export(model, state, args.voice, args.out, speaker=speaker, bandwidth_scale=args.bandwidth_scale)
    sizes = {suffix: os.path.getsize(os.path.join(args.out, f'{args.voice}.{suffix}')) for suffix in ('bin', 'json')}
    print(json.dumps({'voice': args.voice, 'parameters': manifest['parameterCount'], 'bytes': sizes,
                      'quantized': sum(1 for tensor in manifest['tensors'] if tensor['encoding'] == 'int8'),
                      'float32': sum(1 for tensor in manifest['tensors'] if tensor['encoding'] == 'float32'),
                      'remap': manifest['remap']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
