# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The architecture experiments' evaluation of a multi-speaker checkpoint at an export precision - the measures of the
fine-tune comparisons, quantization-aware training, and the size search (nn/sizesearch.py):

- val - the validation losses (nn/trainmulti.py validate: the spectral loss, durations, F0) of the exported model
  (nn/export.py quantized_model at --bits and --group) on the first --val-count reader validation utterances
- panel - the exported model spoken by the browser path (nn/js/speak.js - nn/js/model.js and nn/js/render.js, at 16 kHz)
  for each sample speaker (tools/dashboard/pipeline.json's six: three female, three male) on the listener panel's texts
  (tools/panel-texts.json), scored on the CPU: UTMOS (tools/listeners.py), Whisper's WER (whisper-cli, base.en), and
  the ECAPA speaker similarity of the mean embedding to the speaker's recordings (--reference-count validation and
  test utterances, cached under build/nn/arch/refs/)
- size - the export's bytes, and the generated JavaScript module's bytes and gzip bytes (nn/package.py)

    build/venv-nn/bin/python -B -m nn.archeval build/nn/runs/arch-ft-base/last.pt --name ft-base [--bits 4 --group 32]
        [--device mps] [--out build/nn/arch/eval]

It writes <out>/<name>/result.json (and the WAV files and the export beside it) and prints the summary JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy
import soundfile
import torch

from . import dsynth
from .data import DATA_DIR, ArcticData
from .panel import cosine, edit_distance, listen, transcribe_all, words
from .export import export, load_model, quantized_model
from .package import package
from .trainmulti import Corpus, default_settings, sonorant_mask, validate


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The experiments' directory (the evaluations, and the reference recordings' embeddings) - NN_ARCH_DIR, if set
ARCH_DIR = os.environ.get('NN_ARCH_DIR', os.path.join(ROOT, 'build', 'nn', 'arch'))

# The sample speakers - tools/dashboard/pipeline.json's (three female, three male)
SAMPLE_SPEAKERS = [('arctic-slt', 'female'), ('libritts-5895', 'female'), ('libritts-6345', 'female'),
                   ('arctic-bdl', 'male'), ('libritts-5750', 'male'), ('libritts-2843', 'male')]

# The evaluation's sample rate (Whisper's and the listeners' rate)
RATE = 16000

# The listening samples' texts (tools/panel-texts.json ids) - the Moby-Dick opening, and a dialogue passage
SAMPLE_TEXTS = ('melville-moby', 'austen-pride')


# The mean ECAPA embedding of a speaker's recordings (cached) - numpy [192]
def reference_embedding(speaker, count=8):
    directory = os.path.join(ARCH_DIR, 'refs')
    cache = os.path.join(directory, f'{speaker}-{count}.json')
    if os.path.exists(cache):
        with open(cache, encoding='utf-8') as file:
            return numpy.array(json.load(file)['embedding'])
    os.makedirs(directory, exist_ok=True)
    data = ArcticData(os.path.join(DATA_DIR, 'multi'))
    entries = (data.entries('test', [speaker]) + data.entries('val', [speaker]))[:count]
    paths = []
    for entry in entries:
        path = os.path.join(directory, f'{speaker}-{entry["id"]}.wav'.replace('/', '_'))
        soundfile.write(path, data.arrays(entry)['audio'], data.index['sampleRate'], subtype='PCM_16')
        paths.append(path)
    scores = listen(paths)
    embedding = numpy.mean([scores[path]['embedding'] for path in paths], axis=0)
    with open(cache, 'w', encoding='utf-8') as file:
        json.dump({'speaker': speaker, 'files': paths, 'embedding': embedding.tolist()}, file)
    return embedding


# The validation losses of a model on the first count reader validation utterances (the export's weights)
def validation(model, device, count, corpus=None):
    settings = default_settings(val_count=count)
    corpus = corpus or Corpus('multi', 'val', max_items=count)
    sonorants = torch.tensor(sonorant_mask(corpus.data.index['vocab']), device=device)
    synth = dsynth.DSynth(settings.rate).to(device)
    result = validate(model.to(device), synth, corpus, settings, device, sonorants)
    return {key: round(float(value), 5) for key, value in result.items()}


# Speak the panel texts for the sample speakers with an export (nn/js/speak.js at RATE) - the items
def speak_panel(model_path, out_dir, speakers, texts, manifest):
    keys = [speaker['key'] for speaker in manifest['speakers']]
    items = []
    for speaker, sex in speakers:
        for number, text in enumerate(texts):
            wav = os.path.join(out_dir, f'{speaker}-{number:02d}.wav')
            items.append({'speaker': speaker, 'sex': sex, 'text': text, 'wav': wav, 'seed': number + 1,
                          'speakerId': keys.index(speaker)})
    batch = [{'text': item['text'], 'wav': item['wav'], 'seed': item['seed'], 'speaker': item['speakerId']} for item in items]
    batch_path = os.path.join(out_dir, 'batch.json')
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    process = subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', model_path, '--batch', batch_path,
                              '-r', str(RATE)], check=True, capture_output=True, text=True, cwd=ROOT)
    return items, json.loads(process.stdout.strip().splitlines()[-1])


# The panel's scores per speaker - {speaker: {mos, mosMin, wer, similarity}} and their means
def score_panel(items, reference_count=8):
    paths = [item['wav'] for item in items]
    scores = listen(paths)
    transcripts = transcribe_all(paths)
    out = {}
    for speaker in dict.fromkeys(item['speaker'] for item in items):
        group = [item for item in items if item['speaker'] == speaker]
        mos = [scores[item['wav']]['mos'] for item in group]
        errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in group)
        count = sum(len(words(item['text'])) for item in group)
        mean_embedding = numpy.mean([scores[item['wav']]['embedding'] for item in group], axis=0)
        out[speaker] = {'sex': group[0]['sex'], 'mos': round(float(numpy.mean(mos)), 4), 'mosMin': round(float(numpy.min(mos)), 3),
                        'wer': round(100 * errors / max(1, count), 2),
                        'similarity': round(cosine(mean_embedding, reference_embedding(speaker, reference_count)), 4)}
    means = {key: round(float(numpy.mean([row[key] for row in out.values()])), 4) for key in ('mos', 'wer', 'similarity')}
    return out, means, {item['wav']: transcripts[item['wav']] for item in items}


# Render the listening samples (SAMPLE_TEXTS for the speakers, 44.1 kHz, the browser path) of an export to flat WAV files
# <out_dir>/<name>-<speaker>-<text id>.wav - the paths
def render_samples(model_path, name, out_dir, manifest, speakers=None):
    os.makedirs(out_dir, exist_ok=True)
    keys = [speaker['key'] for speaker in manifest['speakers']]
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        texts = {item['id']: item['text'] for item in json.load(file)['texts']}
    batch = []
    for speaker, _ in speakers or SAMPLE_SPEAKERS:
        for text_id in SAMPLE_TEXTS:
            batch.append({'text': texts[text_id], 'wav': os.path.join(out_dir, f'{name}-{speaker}-{text_id}.wav'), 'seed': 1,
                          'speaker': keys.index(speaker)})
    batch_path = os.path.join(os.path.dirname(model_path), 'samples-batch.json')
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'speak.js'), '-m', model_path, '--batch', batch_path, '-r', '44100'],
                   check=True, capture_output=True, text=True, cwd=ROOT)
    return [item['wav'] for item in batch]


def evaluate(checkpoint, name, out_root, bits=8, group=0, device='cpu', val_count=96, text_count=16, speakers=None,
             listen_panel=True, reference_count=8, keep_wavs=True, samples_dir=None):
    """Evaluate a checkpoint at an export precision - the result dict (also written to <out_root>/<name>/result.json)"""
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    model, state = load_model(checkpoint)
    result = {'name': name, 'checkpoint': checkpoint, 'step': state.get('step'), 'bits': bits, 'group': group,
              'config': {key: value for key, value in model.config.items() if key in ('dim', 'decoderDim', 'encoderLayers',
                                                                                     'decoderLayers', 'encoder', 'correctionBands')},
              'parameters': sum(parameter.numel() for parameter in model.parameters())}

    # The export, and its package
    speaker_index = state['speakers'].index('arctic-bdl')
    manifest = export(model, state, 'model', out_dir, speaker=speaker_index, bits=bits or 8, group=group if bits else 0)
    _, sizes = package(os.path.join(out_dir, 'model'))
    result['size'] = {'binBytes': sizes['binBytes'], 'moduleBytes': sizes[sizes['chosen']]['bytes'],
                      'moduleGzipBytes': sizes[sizes['chosen']]['gzipBytes'], 'encoding': sizes['chosen'],
                      'rawModuleBytes': sizes['raw']['bytes'], 'rawModuleGzipBytes': sizes['raw']['gzipBytes']}

    # The listening samples
    if samples_dir:
        result['samples'] = render_samples(os.path.join(out_dir, 'model'), name, samples_dir, manifest, speakers)

    # The validation losses of the exported weights
    if val_count:
        result['val'] = validation(quantized_model(model, bits, group) if bits else model, torch.device(device), val_count)

    # The panel
    if listen_panel:
        with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
            texts = [item['text'] for item in json.load(file)['texts']][:text_count]
        items, speed = speak_panel(os.path.join(out_dir, 'model'), out_dir, speakers or SAMPLE_SPEAKERS, texts, manifest)
        result['speed'] = speed
        result['speakers'], result['panel'], result['transcripts'] = score_panel(items, reference_count)
        if not keep_wavs:
            for item in items:
                for path in (item['wav'], item['wav'][:-4] + '.json'):
                    if os.path.exists(path):
                        os.remove(path)
    result['seconds'] = round(time.time() - start, 1)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.archeval', description='Evaluate a checkpoint at an export precision')
    parser.add_argument('checkpoint')
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', default=os.path.join(ARCH_DIR, 'eval'))
    parser.add_argument('--bits', type=int, default=8, help='the export\'s bit width (0: the float model\'s validation, an 8-bit export)')
    parser.add_argument('--group', type=int, default=0)
    parser.add_argument('--device', default='cpu', help='the validation\'s device')
    parser.add_argument('--val-count', type=int, default=96, help='reader validation utterances (0: none)')
    parser.add_argument('--texts', type=int, default=16, help='panel texts per speaker')
    parser.add_argument('--no-panel', action='store_true', help='no listening (validation and size only)')
    parser.add_argument('--samples', default=None, help='render the listening samples (44.1 kHz WAV files) into this directory')
    args = parser.parse_args(argv)
    result = evaluate(args.checkpoint, args.name, args.out, args.bits, args.group, args.device, args.val_count, args.texts,
                      listen_panel=not args.no_panel, samples_dir=args.samples)
    json.dump({key: value for key, value in result.items() if key != 'transcripts'}, sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
