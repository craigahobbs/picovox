# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Inference in PyTorch - text to parameter frames with a trained model (a checkpoint, or an exported voice's weights
via export.py's quantized_model), the same steps as lib/model.js: tokens (frontend.py), split into sentences
(each with its own <bos>), rare tokens remapped (the export manifest's remap), durations, frames.

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.synthesize build/nn/runs/slt/best.pt "Hello, world." \
        -o hello.wav [--rate 44100]
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.synthesize build/nn/runs/multi-b/best.pt "Hello." \
        -o hello.wav --speaker libritts-6454 [--blend '{"5": 0.5, "12": 0.5}'] [--style angry --intensity 0.7]

A multi-speaker model's voice is a speaker (a key or an id of its table) or a blend ({id: weight}, the weights
normalized to sum to 1), and its style an emotion (model.py's STYLE_NAMES, or neutral) at an intensity (0-1).
"""

import argparse
import json
import os
import sys

import numpy
import soundfile
import torch

from . import dsynth, params
from .export import load_model, quantized_model, token_remap, training_token_counts
from .frontend import tokenize
from .model import STYLE_NAMES


SENTENCE_ENDS = {'<period>', '<question>', '<wh>', '<exclamation>'}


# Split token ids into sentences, each starting with <bos>
def sentences(ids, vocab):
    out = []
    current = [2]
    for token_id in ids[1:] if ids and ids[0] == 2 else ids:
        current.append(token_id)
        if vocab[token_id] in SENTENCE_ENDS:
            out.append(current)
            current = [2]
    if len(current) > 1:
        out.append(current)
    return out


class Voice:
    """A model ready for inference - the model, its remap, and its channel equalizer"""

    def __init__(self, checkpoint, quantized=True, device='cpu', speaker=None, bandwidth_scale=1.0, blend=None, emotion=None,
                 intensity=1.0, bits=8, group=0):
        model, state = load_model(checkpoint, device)
        self.bandwidth_scale = bandwidth_scale
        self.speaker = speaker_id(state, speaker) if speaker is not None else 0
        self.model = quantized_model(model, bits, group) if quantized else model
        self.model.eval()
        self.state = state
        counts, self.vocab = training_token_counts(state)
        remap = token_remap(self.vocab, counts)
        self.remap = {self.vocab.index(source): self.vocab.index(target) for source, target in remap.items()}
        self.device = device
        self.set_voice(self.speaker, blend, emotion, intensity)

    def set_voice(self, speaker=0, blend=None, emotion=None, intensity=1.0):
        """Set the multi-speaker model's voice - a speaker id (or key) or a blend {id or key: weight}, and a style (an
        emotion name, or None for neutral, at an intensity)"""
        if blend:
            weights = torch.zeros(1, self.model.config['speakers'], device=self.device)
            for key, weight in blend.items():
                weights[0, speaker_id(self.state, key)] += float(weight)
            self.speaker_input = weights / weights.sum()
        else:
            self.speaker = speaker_id(self.state, speaker)
            self.speaker_input = torch.tensor([self.speaker], device=self.device)
        self.emotion = torch.tensor([STYLE_NAMES.index(emotion) if emotion in STYLE_NAMES else -1], device=self.device)
        self.intensity = torch.tensor([float(intensity)], device=self.device)
        with torch.no_grad():
            self.eq = self.model.eq(self.speaker_input)[0].cpu().numpy()

    # The model's conditioning arguments
    def condition(self):
        return {'speaker': self.speaker_input, 'emotion': self.emotion, 'intensity': self.intensity}

    def frames(self, text, pause_threshold=0.5, rate=1.0):
        """Text to parameter frames [T, COUNT] (numpy, the end frame included) and the sentences' token ids and
        durations"""
        ids = tokenize(text)['ids']
        chunks = []
        parts = []
        for sentence in sentences(ids, self.vocab):
            mapped = [self.remap.get(token_id, token_id) for token_id in sentence]
            tokens = torch.tensor([mapped], device=self.device)
            with torch.no_grad():
                result = self.model.infer(tokens, **self.condition(), pause_threshold=pause_threshold, rate=rate)
            values = result['values'][0, :-1].cpu().numpy()
            parts.append({'ids': sentence, 'durations': result['durations'][0].cpu().numpy().tolist()})
            chunks.append(values)
        if not chunks:
            return numpy.zeros((0, params.COUNT)), parts
        values = numpy.concatenate(chunks)
        values = numpy.concatenate([values, values[-1:]])
        if self.bandwidth_scale != 1:
            values = params.bandwidth_scale(values, self.bandwidth_scale)
        return values, parts


# A speaker's row in a checkpoint's speaker table - from an id (int or numeric string) or a key
def speaker_id(state, speaker):
    if isinstance(speaker, int) or (isinstance(speaker, str) and speaker.isdigit()):
        return int(speaker)
    return state['speakers'].index(speaker)


# Render parameter frames [T, COUNT] with dsynth.py (counter noise, as lib/render.js), peak-normalized
def render(values, eq, sample_rate=44100, seed=1, peak=0.89):
    synth = dsynth.DSynth(sample_rate, noise='counter')
    values = numpy.asarray(values)
    with torch.no_grad():
        audio = synth(torch.tensor(values[:, :params.COUNT], dtype=torch.float64), seed=seed,
                      eq=torch.tensor(eq, dtype=torch.float64)).numpy()
    return audio * peak / max(1e-9, numpy.abs(audio).max())


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.synthesize', description='Speak text with a trained model')
    parser.add_argument('checkpoint')
    parser.add_argument('text')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--rate', type=int, default=44100)
    parser.add_argument('--params', default=None)
    parser.add_argument('--float', action='store_true', help='the unquantized model')
    parser.add_argument('--speaker', default=None, help='the speaker of a multi-speaker model')
    parser.add_argument('--bandwidth-scale', type=float, default=1.0)
    parser.add_argument('--blend', default=None, help='a multi-speaker model\'s speaker blend (JSON {id or key: weight})')
    parser.add_argument('--style', default=None, help='a multi-speaker model\'s style (an emotion)')
    parser.add_argument('--intensity', type=float, default=1.0)
    args = parser.parse_args(argv)
    voice = Voice(args.checkpoint, quantized=not args.float, speaker=args.speaker, bandwidth_scale=args.bandwidth_scale,
                  blend=json.loads(args.blend) if args.blend else None, emotion=args.style, intensity=args.intensity)
    values, _ = voice.frames(args.text)
    if args.params:
        params.write_params(args.params, values, eq=voice.eq)
    soundfile.write(args.output, render(values, voice.eq, args.rate), args.rate, subtype='PCM_16')
    print(f'{os.path.basename(args.output)}: {len(values) / params.FRAME_RATE:.2f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
