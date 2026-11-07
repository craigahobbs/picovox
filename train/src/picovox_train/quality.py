# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The gap - how far each of the package's voices is from its source, the training speech's clips of its engine voice
(tts.py), in a pretrained speech model's features - the pinned wav2vec2 phoneme recognizer's (ctc.py): voice
training's quality measure, and make train's gate (panel.py).

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality gap --out build/gap \\
        [--per-voice 40] [--model build/train/voiceModel.js] [-j 8]

Over per_voice held-out clips of each voice (the test split's, 3-9 s, by id), each synthesized by the package from its
text (tools/renderLines.js - the voices, gain, and limiter as the package speaks; --model another voice model module,
default the package's) and its source clip, both trimmed of silence:

- fdL12 - each voice's Frechet distance between its synthesized clips' frames and its source clips' in the
  recognizer's layer 12 - the gap's measure; fdL6 - the same in layer 6

The measure was chosen and validated in November 2026 (paper Section 13.9): of 20 candidates over 480 held-out clips, fd
on layer 12 had the largest gap to Phonon relative to the spread between trained seeds (about 400 to 1), with a second
take of the engine's own clips far closer than any voice of ours (the floor - every voice), each step of 8-6-5-4-bit
rounding farther, stage A alone (without the critic) farther, and the shipped 2.0 voice closer than the simpler recipe's
retrained voice in all twelve voices - as two blind A/B rounds (8 to 0) and the listener panel judged them. It compares
sets of clips of the same texts, so it suits evaluation, not a training loss. (The spectral-balance, pitch, and
movement measures that guided the October 2026 listening - picovox_train.quality measure - and the paired DTW distances
are in git history: the gap superseded them.)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy
import torch

from . import ROOT
from .wav import read_wav, resample, READ_ZEROS


# The gap measure's clips (seconds), its layers (the transformer's), and the trim's silence (dB below the loudest 20 ms
# frame) and margin (s)
GAP_SECONDS = (3.0, 9.0)
GAP_LAYERS = {'L12': 12, 'L6': 6}
GAP_MEASURES = ['fdL12', 'fdL6']
TRIM_DB = 40
TRIM_MARGIN = 0.05


# Render jobs ([{text, voice, wav}]) with the package (tools/renderLines.js - its voice model, or a voice model module's),
# in processes Node processes at once
def render(jobs, processes, model=None):
    groups = [jobs[index::processes] for index in range(processes)]
    command = ['node', os.path.join('tools', 'renderLines.js'), *([os.path.abspath(model)] if model else [])]
    children = [subprocess.Popen(command, cwd=ROOT,  # pylint: disable=consider-using-with
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True) for group in groups if group]
    for child, group in zip(children, [group for group in groups if group]):
        child.stdin.write(''.join(json.dumps(job) + '\n' for job in group))
        child.stdin.close()
    failed = [child.wait() for child in children]
    if any(failed):
        raise RuntimeError('tools/renderLines.js failed')


# Samples trimmed of leading and trailing silence (20 ms frames TRIM_DB below the loudest), TRIM_MARGIN kept
def trim(samples, rate):
    size = rate // 50
    energy = numpy.array([numpy.mean(samples[index:index + size] ** 2) + 1e-12 for index in range(0, len(samples), size)])
    loud = numpy.nonzero(10 * numpy.log10(energy / energy.max()) > -TRIM_DB)[0]
    margin = int(TRIM_MARGIN * rate)
    return samples[max(0, loud[0] * size - margin):min(len(samples), (loud[-1] + 1) * size + margin)]


# A WAV file's features - {layer: [frames, dims] float64}: trimmed, at 16 kHz, normalized as the recognizer expects
def gap_features(model, device, path):
    samples, rate = read_wav(path)
    samples = resample(torch.from_numpy(numpy.ascontiguousarray(trim(samples.mean(axis=1), rate))[None]), rate, 16000, READ_ZEROS)[0]
    samples = ((samples - samples.mean()) / torch.sqrt(samples.var(unbiased=False) + 1e-7)).float()
    with torch.no_grad():
        outputs = model.wav2vec2(samples[None].to(device), output_hidden_states=True)
    return {name: outputs.hidden_states[index][0].cpu().double().numpy() for name, index in GAP_LAYERS.items()}


# The Frechet distance of two sets of frames [count, dims]
def frechet(first, second):
    cov_first, cov_second = numpy.cov(first, rowvar=False), numpy.cov(second, rowvar=False)
    values, vectors = numpy.linalg.eigh(cov_first)
    root = (vectors * numpy.sqrt(numpy.clip(values, 0, None))) @ vectors.T
    product = numpy.linalg.eigvalsh(root @ cov_second @ root)
    return float(numpy.sum((first.mean(0) - second.mean(0)) ** 2) + numpy.trace(cov_first) + numpy.trace(cov_second)
                 - 2 * numpy.sum(numpy.sqrt(numpy.clip(product, 0, None))))


# A voice's gap measures - its synthesized clips' features against their source clips' ([(synth, source)])
def gap_measures(pairs):
    return {f'fd{layer}': frechet(numpy.concatenate([synth[layer] for synth, _ in pairs]),
                                  numpy.concatenate([source[layer] for _, source in pairs])) for layer in GAP_LAYERS}


# Each voice's gap clips - {voice key: [clip with its wav]}: the test split's clips of GAP_SECONDS (by id), per_voice a voice
def gap_clips(per_voice):
    from . import tts  # pylint: disable=import-outside-toplevel
    settings = tts.load_settings()
    directory = tts.corpus_dir(settings)
    clips = {}
    for clip in sorted(tts.selection(settings), key=lambda clip: clip['id']):
        path = tts.wav_path(directory, clip)
        if clip['split'] == 'test' and len(clips.get(clip['key'], [])) < per_voice and os.path.exists(path):
            samples, rate = read_wav(path)
            if GAP_SECONDS[0] <= len(samples) / rate <= GAP_SECONDS[1]:
                clips.setdefault(clip['key'], []).append({**clip, 'wav': path})
    return clips


def gap(model, out_dir, per_voice, jobs):
    """The gap of a voice model module (model - None, the package's) - each voice's measures over per_voice held-out
    clips, synthesized into out_dir (jobs Node renderers; the renderings removed after) - {model, perVoice, clips,
    voices: {key: {fdL12, fdL6}}, fdL12, fdL6 (the voices' means)}"""
    from . import ctc  # pylint: disable=import-outside-toplevel
    clips = gap_clips(per_voice)
    os.makedirs(out_dir, exist_ok=True)

    # (a voice's clip's rendering)
    def synth_path(key, clip):
        return os.path.join(out_dir, f'{key}-{clip["id"]}.wav')
    render([{'text': clip['text'], 'voice': key, 'wav': synth_path(key, clip)} for key, voice_clips in clips.items()
            for clip in voice_clips], jobs, model)
    recognizer, _, device = ctc.load_ctc()
    voices = {}
    for key, voice_clips in clips.items():
        pairs = [(gap_features(recognizer, device, synth_path(key, clip)), gap_features(recognizer, device, clip['wav']))
                 for clip in voice_clips]
        voices[key] = gap_measures(pairs)
        for clip in voice_clips:
            os.remove(synth_path(key, clip))
    result = {'model': model, 'perVoice': per_voice, 'clips': sum(len(voice_clips) for voice_clips in clips.values()), 'voices': voices}
    for name in GAP_MEASURES:
        result[name] = sum(values[name] for values in voices.values()) / max(1, len(voices))
    return result


# The gap's table - a line per voice, then the mean
def gap_lines(result):
    lines = [f'{"voice":16}' + ''.join(f'{name:>12}' for name in GAP_MEASURES)]
    for key, values in result['voices'].items():
        lines.append(f'{key:16}' + ''.join(f'{values[name]:12.4g}' for name in GAP_MEASURES))
    lines.append(f'{"mean":16}' + ''.join(f'{result[name]:12.4g}' for name in GAP_MEASURES))
    return lines


def gap_command(args):
    result = gap(args.model, os.path.join(args.out, 'gap'), args.per_voice, args.jobs)
    path = os.path.join(args.out, 'gap.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    print(f'quality gap: {result["clips"]} clips - {path}')
    print('\n'.join(gap_lines(result)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='picovox_train.quality', description="The voices' gap to their source clips")
    commands = parser.add_subparsers(dest='command', required=True)
    gap_parser = commands.add_parser('gap', help="the voices' distance from their source in wav2vec2 features")
    gap_parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'quality'), help='the output directory')
    gap_parser.add_argument('--per-voice', type=int, default=40, help='held-out clips per voice')
    gap_parser.add_argument('--model', help='a voice model module to render with (default: the package\'s)')
    gap_parser.add_argument('-j', '--jobs', type=int, default=8, help='Node renderers')
    gap_parser.set_defaults(func=gap_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
