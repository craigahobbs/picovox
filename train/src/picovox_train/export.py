# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Export - a trained model (an trainmulti.py or distill.py checkpoint) to the browser's format, <out>/<voice>.bin
and <out>/<voice>.json (default data/nn/), which lib/model.js loads (and package.py packages as a module):

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.export build/nn/runs/multi-b/best.pt --voice multi

The multi-speaker, multi-style model (model.py's MultiModel) exports whole: its speaker table (a row per speaker -
the readers' data/nn/speakers.json ids, then the emotional corpus's actors), its style table (model.py's
STYLE_NAMES; neutral is the zero vector), and its channel equalizers (eq_table, dB, a row per speaker, float32); the
manifest lists the speakers (id, key, sex, corpus, role) and the styles, so lib/model.js can blend speakers and
scale styles.

The weights are quantized to 8 bits: every weight matrix, convolution kernel, and embedding table as int8 with one
float32 scale per output channel (row) - symmetric, the scale the row's largest magnitude / 127; the biases and layer
norm parameters, a small fraction of the model, stay float32. The .bin is the tensors' data back to back (each
tensor's int8 data, then its float32 scales; or its float32 data), aligned to 4 bytes; the .json manifest lists each
tensor's name, shape, encoding, and byte offsets, with the model config, the vocabulary remap, the channel equalizer,
and the inference defaults (the pause threshold, and the formant bandwidth scale - a sharpening of the network's
over-smoothed bandwidths, chosen on the validation sentences).

The quantization is the only lossy step - quantized_model() gives the PyTorch model with the exported weights
(dequantized), which lib/model.js must match.

The manifest's training record (the checkpoint's step, validation, and arguments) has its paths relative to the
repository root.

Lower precision (--bits 6, 5, 4, and --group for per-group scales) uses quant.py's 'packed' encoding: the values
packed at the bit width, float16 scales per row or per group (the manifest entry's bits and group). The default (8 bits
per row) is the 'int8' format above, unchanged.
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
from .data import DATA_DIR, ArcticData
from .model import STYLE_NAMES, build_model, parameter_count


OUT_DIR = os.path.join(ROOT, 'data', 'nn')
FORMAT_VERSION = 1

# Tensors smaller than this stay float32
MIN_QUANTIZED = 1024


# Load a checkpoint's model (on the CPU, in eval mode) - (model, checkpoint)
def load_model(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_model(state['config'], priors=state.get('priors'))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state


# The tensors that stay float32 whatever their size
FLOAT_TENSORS = ('eq_table',)


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


# Quantize a tensor per output channel (dim 0) - (int8 values, float32 scales)
def quantize(tensor):
    rows = tensor.reshape(tensor.shape[0], -1).numpy().astype(numpy.float64)
    scales = numpy.abs(rows).max(axis=1) / 127
    scales[scales == 0] = 1.0
    values = numpy.clip(numpy.round(rows / scales[:, None]), -127, 127).astype(numpy.int8)
    return values, scales.astype(numpy.float32)


# Whether a tensor is quantized
def is_quantized(tensor, name=None):
    return tensor.dim() >= 2 and tensor.numel() >= MIN_QUANTIZED and name not in FLOAT_TENSORS


# A copy of a model with its weights quantized and dequantized - the exported model, in PyTorch (bits and group - the
# export's precision, quant.py)
def quantized_model(model, bits=8, group=0):
    model = copy.deepcopy(model)
    state = model.state_dict()
    for name, tensor in exported_tensors(model).items():
        if is_quantized(tensor, name):
            if quant.is_int8(bits, group):
                values, scales = quantize(tensor)
                state[name] = torch.tensor(values.astype(numpy.float32) * scales[:, None]).reshape(tensor.shape)
            else:
                values, scales = quant.quantize_array(tensor.numpy(), bits, group)
                state[name] = torch.tensor(quant.dequantize_array(values, scales, tuple(tensor.shape), group))
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


# The training token counts of a checkpoint (its state) - of the multi-speaker model's corpora
def training_token_counts(state):
    counts = collections.Counter()
    corpora = ['multi'] + (['cremad'] if state.get('args', {}).get('cremad', 0) > 0 else [])
    for corpus in corpora:
        data = ArcticData(os.path.join(DATA_DIR, corpus))
        for entry in data.entries('train'):
            counts.update(numpy.asarray(data.tokens[entry['tokenOffset']:entry['tokenOffset'] + entry['tokenCount']]).tolist())
    return counts, data.index['vocab']


# The multi-speaker model's speaker list - [{id, key, sex, corpus, role ('reader' - the American reader pool, or
# 'style' - an actor of the emotional corpus), and for readers the reader's name, median F0, and UTMOS}]
def speaker_list(state):
    with open(os.path.join(ROOT, 'data', 'nn', 'speakers.json'), encoding='utf-8') as file:
        readers = {speaker['key']: speaker for speaker in json.load(file)['speakers']}
    cremad = ArcticData(os.path.join(DATA_DIR, 'cremad')).index['speakers']
    out = []
    for index, key in enumerate(state['speakers']):
        if key in readers:
            reader = readers[key]
            if reader['id'] != index:
                raise ValueError(f'speaker {key}: table row {index} is not its data/nn/speakers.json id {reader["id"]}')
            out.append({'id': index, 'key': key, 'sex': reader['sex'], 'corpus': reader['source'], 'role': 'reader',
                        'reader': reader.get('reader'), 'medianF0': reader.get('medianF0'), 'utmos': reader.get('utmos')})
        else:
            actor = cremad.get(key, {})
            out.append({'id': index, 'key': key, 'sex': actor.get('sex'), 'corpus': 'CREMA-D', 'role': 'style'})
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


# Export a model - writes <out>/<voice>.bin and <voice>.json, returns the manifest (bits and group - the weights'
# precision, quant.py; the default is the 8-bit per-row 'int8' format)
def export(model, state, voice, out_dir=OUT_DIR, extra=None, speaker=0, bandwidth_scale=1.0, bits=8, group=0):
    os.makedirs(out_dir, exist_ok=True)
    if not quant.is_int8(bits, group):
        extra = {**(extra or {}), 'quantization': {'bits': bits, 'group': group}}
    chunks = []
    offset = 0
    tensors = []
    for name, tensor in exported_tensors(model).items():
        entry = {'name': name, 'shape': list(tensor.shape)}
        if is_quantized(tensor, name) and not quant.is_int8(bits, group):
            values, scales = quant.quantize_array(tensor.numpy(), bits, group)
            data = quant.pack(values, bits)
            scale_data = scales.astype('<f2').tobytes()
            entry.update({'encoding': 'packed', 'bits': bits, 'group': quant.group_size(values.shape[1], group),
                          'offset': offset, 'scaleOffset': offset + _aligned(len(data))})
            chunks.append(data + b'\0' * (_aligned(len(data)) - len(data)))
            chunks.append(scale_data + b'\0' * (_aligned(len(scale_data)) - len(scale_data)))
            offset += _aligned(len(data)) + _aligned(len(scale_data))
        elif is_quantized(tensor, name):
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
    counts, vocab = training_token_counts(state)
    with torch.no_grad():
        eq = [round(float(value), 3) for value in model.eq(torch.tensor([speaker]))[0]]
    manifest = {
        'format': 'voice-synth-nn', 'version': FORMAT_VERSION, 'voice': voice, 'kind': 'multi',
        'description': 'The multi-speaker, multi-style neural model (nn/trainmulti.py) - tokens to durations to '
                       'synthesizer parameter frames (nn/params.py), rendered by lib/render.js; a voice is any blend of '
                       'the speaker table\'s rows, a style any blend of the style table\'s rows (neutral is zero)',
        'config': model.config, 'parameterNames': params.NAMES, 'frameRate': params.FRAME_RATE,
        'speakers': speaker_list(state), 'styles': list(STYLE_NAMES), 'defaultSpeaker': speaker,
        'eq': eq, 'remap': token_remap(vocab, counts), 'vocab': vocab,
        'inference': {'pauseThreshold': 0.5, 'bandwidthScale': bandwidth_scale},
        'parameterCount': parameter_count(model),
        'parameterCountCore': parameter_count(model) - model.speaker.weight.numel() - model.eq_free.numel(),
        'bytes': len(binary), 'tensors': tensors,
        'training': {'step': state.get('step'), 'val': state.get('val'), 'args': relative_paths(state.get('args'))},
        'license': 'Trained on LibriTTS-R (CC BY 4.0), CMU ARCTIC (free use with attribution), and CREMA-D (a Produced '
                   'Work of the Open Database License 1.0 database - Cao, Cooper, Keutmann, Gur, Nenkova, and Verma, '
                   '"CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", IEEE Transactions on Affective '
                   'Computing 5(4), 2014)', **(extra or {})
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
    parser.add_argument('--speaker', default=None, help='the default speaker and its channel equalizer (default: arctic-bdl)')
    parser.add_argument('--bandwidth-scale', type=float, default=1.0,
                        help='the default formant bandwidth scale at inference (lib/model.js speakFrames)')
    parser.add_argument('--bits', type=int, default=8, help='the weights\' bit width (quant.py; default 8)')
    parser.add_argument('--group', type=int, default=0, help='per-group scales of this many weights (default 0: per row)')
    args = parser.parse_args(argv)
    model, state = load_model(args.checkpoint)
    speaker = state['speakers'].index(args.speaker if args.speaker else 'arctic-bdl')
    manifest = export(model, state, args.voice, args.out, speaker=speaker, bandwidth_scale=args.bandwidth_scale, bits=args.bits,
                      group=args.group)
    sizes = {suffix: os.path.getsize(os.path.join(args.out, f'{args.voice}.{suffix}')) for suffix in ('bin', 'json')}
    print(json.dumps({'voice': args.voice, 'parameters': manifest['parameterCount'], 'bytes': sizes,
                      'quantized': sum(1 for tensor in manifest['tensors'] if tensor['encoding'] in ('int8', 'packed')),
                      'float32': sum(1 for tensor in manifest['tensors'] if tensor['encoding'] == 'float32'),
                      'remap': manifest['remap']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
