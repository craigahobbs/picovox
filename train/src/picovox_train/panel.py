# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The listener panel - how make train measures a voice model and gates it (no training framework: an exported voice
model is spoken by the browser path, tools/speak.js):

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.panel COMMAND ...

    eval NAME=CHECKPOINT [--reference NAME] [--whisper-workers 3]    # a checkpoint or a packaged module
    gate NAME --against NAME                                         # within the tolerance of another

- listen - UTMOS and the ECAPA speaker embedding (listeners.py, in a process of its own)
- transcribe - whisper.cpp base.en (downloads/whisper/ggml-base.en.bin), cached beside each WAV; words and edit_distance give
  the word error rate as the former tools/eval.js counted it
- cosine - the speaker similarity of two embeddings
- speak - a batch of texts with an exported model (its speed report)
- score_model - an export's scores: the recipe's speakers (train/data/train.json panel) by the panel's texts
  (tools/panel-texts.json), each speaker's mean UTMOS, WER, and mean embedding, and against a reference result, the
  speaker similarity
- evaluate_model - a checkpoint (exported, 8-bit - export.py - and packaged, package.py: its module's bytes and gzip
  bytes) or a packaged module (unpacked - tools/unpack.js) spoken and scored - NN_EVAL_DIR/<name>/result.json
- eval and gate - make train's: the shipped model (the baseline) and the trained voice evaluated, the voice's
  similarity to the baseline's, and the gate - the voice within the recipe's tolerance of the baseline
  (NN_EVAL_DIR/summary.json; the exit status 1 if not)
"""

import argparse
import concurrent.futures
import datetime
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy

from .export import export, load_model
from .model import core_parameter_count, parameter_count
from .package import package
from . import RECIPE_PATH, ROOT, corpora


LISTENERS_PYTHON = os.path.join(ROOT, 'build', 'venv-train', 'bin', 'python')
LISTENERS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'listeners.py')
WHISPER_MODEL = corpora.WHISPER_MODEL
TEXTS_PATH = os.path.join(ROOT, 'tools', 'panel-texts.json')
EVAL_DIR = os.environ.get('NN_EVAL_DIR', os.path.join(ROOT, 'build', 'nn', 'eval'))


# Score WAV files with the listener panel (on the CPU) - {path: {mos, embedding}}
def listen(paths):
    command = [LISTENERS_PYTHON, '-I', LISTENERS_SCRIPT, '--device', 'cpu', '--files-from', '-']
    process = subprocess.run(command, input='\n'.join(paths), capture_output=True, text=True, check=True)
    return {item['file']: item for item in (json.loads(line) for line in process.stdout.splitlines() if line.strip())}


# Transcribe a WAV file with whisper.cpp base.en (cached beside it as <name>.json) - the text
def transcribe(path):
    base = path[:-4]
    if not os.path.exists(base + '.json') or os.path.getmtime(base + '.json') < os.path.getmtime(path):
        subprocess.run(['whisper-cli', '-m', WHISPER_MODEL, '-t', '2', '-f', path, '-np', '-ojf', '-of', base], check=True,
                       capture_output=True, cwd=ROOT)
    with open(base + '.json', encoding='utf-8') as file:
        result = json.load(file)
    return ''.join(segment['text'] for segment in result['transcription']).strip()


# Transcripts of WAV files, workers at a time (two threads each) - {path: text}
def transcribe_all(paths, workers=6):
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return dict(zip(paths, pool.map(transcribe, paths)))


# Words for scoring (the former tools/eval.js's words)
def words(text):
    return [word for word in re.sub(r'[^a-z0-9\' ]+', ' ', text.lower()).replace("'", '').split() if word]


# Word-level edit distance
def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for ix, ref_word in enumerate(reference, 1):
        current = [ix] + [0] * len(hypothesis)
        for jx, hyp_word in enumerate(hypothesis, 1):
            current[jx] = min(previous[jx] + 1, current[jx - 1] + 1, previous[jx - 1] + (ref_word != hyp_word))
        previous = current
    return previous[-1]


# The cosine similarity of two vectors
def cosine(first, second):
    first, second = numpy.asarray(first), numpy.asarray(second)
    return float(first @ second / (numpy.linalg.norm(first) * numpy.linalg.norm(second)))


# The panel texts - [{id, text}]
def panel_texts():
    with open(TEXTS_PATH, encoding='utf-8') as file:
        return json.load(file)['texts']


# Speak a batch ([{text, wav, seed, speaker}]) with an export (tools/speak.js) - its speed report
def speak(model_path, batch, batch_path, rate):
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    command = ['node', os.path.join(ROOT, 'tools', 'speak.js'), '-m', model_path, '--batch', batch_path, '-r', str(rate)]
    process = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(process.stdout.strip().splitlines()[-1])


def score_model(model_path, out_dir, settings, reference=None, whisper_workers=6):
    """Speak and score an export (model_path - its manifest and weights without the extension) - {panelSpeed, speakers,
    utterances, panel}; settings - the recipe's panel ({speakers: [[key, sex]], rate}); reference - another model's result
    (the similarity)"""
    with open(model_path + '.json', encoding='utf-8') as file:
        keys = [speaker['key'] for speaker in json.load(file)['speakers']]
    items = []
    for speaker, sex in settings['speakers']:
        for number, text in enumerate(panel_texts()):
            items.append({'speaker': speaker, 'sex': sex, 'text': text['text'],
                          'wav': os.path.join(out_dir, f'{speaker}-{text["id"]}.wav'), 'seed': number + 1})
    batch = [{'text': item['text'], 'wav': item['wav'], 'seed': item['seed'], 'speaker': keys.index(item['speaker'])} for item in items]
    result = {'panelSpeed': speak(model_path, batch, os.path.join(out_dir, 'batch.json'), settings['rate'])}
    paths = [item['wav'] for item in items]
    scores = listen(paths)
    transcripts = transcribe_all(paths, whisper_workers)
    speakers = {}
    for speaker, sex in settings['speakers']:
        group = [item for item in items if item['speaker'] == speaker]
        mos = [scores[item['wav']]['mos'] for item in group]
        errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in group)
        count = sum(len(words(item['text'])) for item in group)
        embedding = numpy.mean([scores[item['wav']]['embedding'] for item in group], axis=0)
        row = {'sex': sex, 'mos': float(numpy.mean(mos)), 'mosMin': float(numpy.min(mos)), 'wer': 100 * errors / max(1, count),
               'embedding': embedding.tolist()}
        if reference is not None:
            row['similarity'] = cosine(embedding, reference['speakers'][speaker]['embedding'])
            row['pairedSimilarity'] = float(numpy.mean([
                cosine(scores[item['wav']]['embedding'], reference['utterances'][os.path.basename(item['wav'])]['embedding'])
                for item in group]))
        speakers[speaker] = row
    result['speakers'] = speakers
    result['utterances'] = {os.path.basename(item['wav']): {'mos': scores[item['wav']]['mos'],
                                                            'embedding': scores[item['wav']]['embedding'],
                                                            'transcript': transcripts[item['wav']]} for item in items}
    result['panel'] = {key: float(numpy.mean([row[key] for row in speakers.values()]))
                       for key in ('mos', 'wer', 'similarity', 'pairedSimilarity') if all(key in row for row in speakers.values())}
    if reference is not None:
        result['panel']['similarityMin'] = float(min(row['similarity'] for row in speakers.values()))
    for path in paths:
        os.remove(path)
    return result


#
# make train's evaluation and gate
#

# The settings - the recipe's (train/data/train.json) panel
def load_config(path=RECIPE_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)['panel']


# A path relative to the repository root as an absolute path
def root_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# Append a progress line to EVAL_DIR/eval.log (and stderr)
def log_line(text):
    os.makedirs(EVAL_DIR, exist_ok=True)
    line = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {text}'
    with open(os.path.join(EVAL_DIR, 'eval.log'), 'a', encoding='utf-8') as file:
        file.write(line + '\n')
    print(line, file=sys.stderr, flush=True)


def evaluate_model(name, checkpoint, settings, reference=None, whisper_workers=6):
    """Export, package, speak, and score a checkpoint, or speak and score a packaged module (a .js path - unpacked to its
    export, tools/unpack.js) as packaged - its result (also EVAL_DIR/<name>/result.json); reference - another model's
    result (the similarity)"""
    out_dir = os.path.join(EVAL_DIR, name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    start = time.time()
    model_path = os.path.join(out_dir, 'model')
    if checkpoint.endswith('.js'):
        subprocess.run(['node', os.path.join(ROOT, 'tools', 'unpack.js'), root_path(checkpoint), model_path], check=True)
        with open(model_path + '.json', encoding='utf-8') as file:
            manifest = json.load(file)
        with open(root_path(checkpoint), 'rb') as file:
            module = file.read()
        # (the parameter count - package.py's header comment)
        result = {'name': name, 'checkpoint': checkpoint, 'step': None, 'config': manifest['config'],
                  'parameters': int(re.search(rb'\((\d+) parameters', module).group(1)),
                  'size': {'binBytes': os.path.getsize(model_path + '.bin'),
                           'manifestBytes': len(json.dumps(manifest, separators=(',', ':')).encode('utf-8')),
                           'encoding': 'module', 'moduleBytes': len(module), 'moduleGzipBytes': len(gzip.compress(module, 9))}}
    else:
        model, state = load_model(checkpoint)
        result = {'name': name, 'checkpoint': checkpoint, 'step': state.get('step'),
                  'config': {key: model.config[key]
                             for key in ('dim', 'decoderDim', 'encoderLayers', 'decoderLayers', 'heads', 'ffnMultiple')},
                  'parameters': parameter_count(model), 'parametersCore': core_parameter_count(model)}
        export(model, state, 'model', out_dir, speaker=0)
        _, sizes = package(model_path)
        result['size'] = {'binBytes': sizes['binBytes'], 'manifestBytes': sizes['manifestBytes'], 'encoding': sizes['chosen'],
                          'moduleBytes': sizes[sizes['chosen']]['bytes'], 'moduleGzipBytes': sizes[sizes['chosen']]['gzipBytes']}
    result.update(score_model(model_path, out_dir, settings, reference, whisper_workers))
    result['seconds'] = round(time.time() - start, 1)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    return result


# An evaluated model's result (EVAL_DIR/<name>/result.json)
def load_result(name):
    with open(os.path.join(EVAL_DIR, name, 'result.json'), encoding='utf-8') as file:
        return json.load(file)


# The eval subcommand - a model (NAME=CHECKPOINT) through the panel, against a reference's result
def evaluate(args):
    settings = load_config(args.config)
    name, _, checkpoint = args.model.partition('=')
    reference = load_result(args.reference) if args.reference else None
    log_line(f'{name}: evaluating {checkpoint}')
    result = evaluate_model(name, checkpoint, settings, reference, whisper_workers=args.whisper_workers)
    measures = result['panel']
    log_line(f'{name}: {result["parameters"]} parameters, module {result["size"]["moduleBytes"]} bytes '
             f'({result["size"]["moduleGzipBytes"]} gzip); UTMOS {measures["mos"]:.3f}, WER {measures["wer"]:.2f}%' +
             (f', similarity to {args.reference} {measures["similarity"]:.3f}' if reference else '') + f' ({result["seconds"]:.0f} s)')
    return 0


# A result's summary row
def summary_row(result):
    measures = result['panel']
    return {'name': result['name'], 'parameters': result['parameters'], 'moduleBytes': result['size']['moduleBytes'],
            'moduleGzipBytes': result['size']['moduleGzipBytes'], 'mos': measures['mos'], 'wer': measures['wer'],
            'similarity': measures.get('similarity'), 'realTimeTotal': result['panelSpeed']['realTimeTotal']}


# The gate subcommand - an evaluated model within the recipe's tolerance of another (EVAL_DIR/summary.json); the exit
# status 1 if not
def gate(args):
    tolerance = load_config(args.config)['tolerance']
    voice, basis = summary_row(load_result(args.model)), summary_row(load_result(args.against))
    voice['within'] = {'mos': voice['mos'] >= basis['mos'] - tolerance['mos'], 'wer': voice['wer'] <= basis['wer'] + tolerance['wer'],
                       'similarity': voice['similarity'] is not None and voice['similarity'] >= tolerance['similarity']}
    passed = all(voice['within'].values())
    summary = {'name': args.model, 'against': args.against, 'tolerance': tolerance, 'rows': [voice, basis], 'passed': passed}
    with open(os.path.join(EVAL_DIR, 'summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=1)
    for row in summary['rows']:
        log_line(f'{row["name"]}: {row["parameters"]} parameters, {row["moduleBytes"]} bytes ({row["moduleGzipBytes"]} gzip), '
                 f'UTMOS {row["mos"]:.3f}, WER {row["wer"]:.2f}%, real time {row["realTimeTotal"]:.1f}x')
    failed = [key for key, within in voice['within'].items() if not within]
    log_line(f'gate: {args.model} {"passed" if passed else "FAILED (" + ", ".join(failed) + ")"} - UTMOS {voice["mos"]:.3f} '
             f'({args.against} {basis["mos"]:.3f}), WER {voice["wer"]:.2f}% ({args.against} {basis["wer"]:.2f}%), similarity '
             f'{voice["similarity"] or 0:.3f} (tolerance: UTMOS -{tolerance["mos"]}, WER +{tolerance["wer"]}, similarity '
             f'>= {tolerance["similarity"]})')
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.panel', description="The listener panel and make train's gate")
    parser.add_argument('--config', default=RECIPE_PATH, help='the recipe (default train/data/train.json - its panel)')
    subparsers = parser.add_subparsers(dest='command', required=True)
    sub = subparsers.add_parser('eval', help='a model through the panel')
    sub.add_argument('model', help='NAME=CHECKPOINT - a checkpoint, or a packaged module (.js); its result in <eval>/NAME')
    sub.add_argument('--reference', default=None, help="an evaluated model's name - the speaker similarity's reference")
    sub.add_argument('--whisper-workers', type=int, default=6, help='whisper.cpp processes at a time (two threads each)')
    sub.set_defaults(func=evaluate)
    sub = subparsers.add_parser('gate', help="an evaluated model within the tolerance of another's scores")
    sub.add_argument('model', help="the evaluated model's name")
    sub.add_argument('--against', required=True, help="the basis - an evaluated model's name (make train's: baseline)")
    sub.set_defaults(func=gate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
