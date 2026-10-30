# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Export - a trained model (an trainmulti.py or distill.py checkpoint) to the browser's format, <out>/<voice>.bin
and <out>/<voice>.json (default train/data/), which lib/model.js loads (and package.py packages as a module):

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.export build/nn/runs/multi-b/best.pt --voice multi

The multi-speaker model (model.py's MultiModel) exports whole: its speaker table (a row per voice - the training
speech's voices, tts.py, in the recipe's order) and its channel equalizers (eq_table, dB, a row per speaker,
float32); the manifest lists the speakers (id, key - the package's voice key, sex, source - the engine's model and
voice, median F0).

The weights are quantized to 8 bits: every weight matrix, convolution kernel, and embedding table as int8 with one
float32 scale per output channel (row) - symmetric, the scale the row's largest magnitude / 127; the biases and layer
norm parameters, a small fraction of the model, stay float32. The .bin is the tensors' data back to back (each
tensor's int8 data, then its float32 scales; or its float32 data), aligned to 4 bytes; the .json manifest lists each
tensor's name, shape, encoding, and byte offsets, with the model config, the vocabulary remap, and the default speaker's
channel equalizer (the first's, male - the package's default voice).

The quantization (quant.py) is the only lossy step - quantized_model() gives the PyTorch model with the exported
weights (dequantized), which lib/model.js must match.

The manifest's training record (the checkpoint's step, validation, and arguments) has its paths relative to the
repository root.
"""

import argparse
import collections
import copy
import json
import os
import sys

import numpy
import torch

from . import ROOT, params, quant
from .data import DATA_DIR, CorpusData
from .model import MultiModel, core_parameter_count, parameter_count


OUT_DIR = os.path.join(ROOT, 'build', 'export')
FORMAT_VERSION = 1


# Load a checkpoint's model (on the CPU, in eval mode) - (model, checkpoint); the multi-speaker model only (config
# 'kind': 'multi' - the single-speaker model and its trainer, nn/train.py, are in git history)
def load_model(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    if state['config'].get('kind') != 'multi':
        raise ValueError('only the multi-speaker model (config kind "multi") is supported')
    model = MultiModel(state['config'], priors=state.get('priors'))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state


# The exported tensors of a model's state - {name: tensor} - the whole speaker table, and the channel equalizers as a
# table of gains (dB)
def exported_tensors(model):
    tensors = collections.OrderedDict()
    for name, tensor in model.state_dict().items():
        if name == 'eq_free':
            tensor = params.EQ_RANGE_DB * torch.tanh(tensor)
            name = 'eq_table'
        tensors[name] = tensor.detach().cpu().float()
    return tensors


# A copy of a model with its weights quantized and dequantized - the exported model, in PyTorch
def quantized_model(model):
    model = copy.deepcopy(model)
    state = model.state_dict()
    for name, tensor in exported_tensors(model).items():
        if quant.is_quantized(tensor, name):
            state[name] = torch.tensor(quant.dequantize(*quant.quantize(tensor), tensor.shape))
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


# The training token counts of the training speech (the multi-speaker model's corpus) - (counts, its vocabulary)
def training_token_counts():
    counts = collections.Counter()
    data = CorpusData(os.path.join(DATA_DIR, 'tts'))
    for entry in data.entries('train'):
        counts.update(numpy.asarray(data.tokens[entry['tokenOffset']:entry['tokenOffset'] + entry['tokenCount']]).tolist())
    return counts, data.index['vocab']


# The multi-speaker model's speaker list - [{id, key, sex, source, medianF0}] from the training speech's index (each
# speaker's row its index)
def speaker_list(state):
    speakers = CorpusData(os.path.join(DATA_DIR, 'tts')).index['speakers']
    out = []
    for index, key in enumerate(state['speakers']):
        speaker = speakers[key]
        if speaker['index'] != index:
            raise ValueError(f'speaker {key}: table row {index} is not its index {speaker["index"]}')
        engine = speaker['engine']
        out.append({'id': index, 'key': key, 'sex': speaker['sex'], 'source': f'{engine["model"]} {engine["voice"]} ({engine["digest"]})',
                    'medianF0': speaker['medianF0']})
    return out


# A checkpoint's training arguments as the manifest records them - every absolute path relative to the repository root
# (a published model never records the training machine's paths)
def relative_paths(value):
    if isinstance(value, dict):
        return {key: relative_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_paths(item) for item in value]
    if isinstance(value, str) and os.path.isabs(value):
        return os.path.relpath(value, ROOT)
    return value


# Export a model - writes <out>/<voice>.bin and <voice>.json, returns the manifest
def export(model, state, voice, out_dir=OUT_DIR, extra=None, speaker=0):
    os.makedirs(out_dir, exist_ok=True)
    chunks = []
    offset = 0
    tensors = []
    for name, tensor in exported_tensors(model).items():
        entry = {'name': name, 'shape': list(tensor.shape)}
        if quant.is_quantized(tensor, name):
            values, scales = quant.quantize(tensor)
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
    counts, vocab = training_token_counts()
    with torch.no_grad():
        eq = [round(float(value), 3) for value in model.eq(torch.tensor([speaker]))[0]]
    manifest = {
        'format': 'voice-synth-nn', 'version': FORMAT_VERSION, 'voice': voice, 'kind': 'multi',
        'description': 'The multi-speaker neural model (picovox_train/trainmulti.py) - tokens to durations to synthesizer '
                       'parameter frames (picovox_train/params.py), rendered by lib/render.js; a voice is a row of the speaker table',
        'config': model.config, 'parameterNames': params.NAMES, 'frameRate': params.FRAME_RATE,
        'speakers': speaker_list(state), 'defaultSpeaker': speaker,
        'eq': eq, 'remap': token_remap(vocab, counts), 'vocab': vocab,
        'parameterCount': parameter_count(model),
        'parameterCountCore': core_parameter_count(model),
        'bytes': len(binary), 'tensors': tensors,
        'training': {'step': state.get('step'), 'val': state.get('val'), 'args': relative_paths(state.get('args'))},
        'license': f'Trained on speech generated by {CorpusData(os.path.join(DATA_DIR, "tts")).index["license"]}', **(extra or {})
    }
    with open(os.path.join(out_dir, f'{voice}.bin'), 'wb') as file:
        file.write(binary)
    with open(os.path.join(out_dir, f'{voice}.json'), 'w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=1)
    return manifest


def _aligned(size):
    return (size + 3) // 4 * 4


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.export', description='Export a model for lib/model.js')
    parser.add_argument('checkpoint')
    parser.add_argument('--voice', required=True)
    parser.add_argument('--out', default=OUT_DIR)
    args = parser.parse_args(argv)
    model, state = load_model(args.checkpoint)
    manifest = export(model, state, args.voice, args.out)
    sizes = {suffix: os.path.getsize(os.path.join(args.out, f'{args.voice}.{suffix}')) for suffix in ('bin', 'json')}
    print(json.dumps({'voice': args.voice, 'parameters': manifest['parameterCount'], 'bytes': sizes,
                      'quantized': sum(1 for tensor in manifest['tensors'] if tensor['encoding'] == 'int8'),
                      'float32': sum(1 for tensor in manifest['tensors'] if tensor['encoding'] == 'float32'),
                      'remap': manifest['remap']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
