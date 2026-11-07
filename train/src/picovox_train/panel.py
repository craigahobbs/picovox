# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The listener panel - how make train measures a voice model and gates it (no training framework: an exported voice
model is spoken by the browser path, tools/speak.js, and by the package's, tools/renderLines.js):

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.panel COMMAND ...

    eval NAME=CHECKPOINT [--whisper-workers 3] [-j 8]    # a checkpoint or a packaged module
    gate NAME --against NAME                             # within the tolerance of another

- transcribe - whisper.cpp base.en (downloads/whisper/ggml-base.en.bin), cached beside each WAV; words and edit_distance give
  the word error rate - the panel's intelligibility, and its one judge independent of the training's recognizer
- speak - a batch of texts with an exported model (its speed report)
- score_model - an export's word error rate: the recipe's voices (train/data/train.json tts voices) by the panel's texts
  (tools/panel-texts.json), each voice's and their mean
- evaluate_model - a checkpoint (exported, 7-bit - export.py, as NN_EVAL_DIR/<name>/multi.json and multi.bin, which make
  train's package stage packages - and packaged, package.py, as NN_EVAL_DIR/<name>/voiceModel.js) or a packaged module
  (unpacked - tools/unpack.js) spoken and scored, and the module's gap to its source (quality.py - each voice's fdL12
  over the training speech's held-out clips, the package speaking) - NN_EVAL_DIR/<name>/result.json
- eval and gate - make train's: the shipped model (the baseline - a module's result is cached in the training speech's
  directory, keyed by the module and the evaluation's inputs, so a shipped model is evaluated once) and the trained
  voice evaluated, and the gate - the voice within the recipe's tolerance of the baseline (NN_EVAL_DIR/summary.json;
  the exit status 1 if not)
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy

from .books import display_path
from .export import export, load_model
from .package import module_size, package
from . import RECIPE_PATH, ROOT, corpora, g2p, quality, read_json, tts
from .tts import PANEL_TEXTS_PATH


WHISPER_MODEL = corpora.WHISPER_MODEL
EVAL_DIR = os.environ.get('NN_EVAL_DIR', os.path.join(ROOT, 'build', 'nn', 'eval'))

# The export's name - the voice make train's package stage packages (its manifest's voice) - and the module's
EXPORT_VOICE = 'multi'
MODULE_NAME = 'voiceModel.js'


# Transcribe a WAV file with whisper.cpp base.en (cached beside it as <name>.json) - the text
def transcribe(path):
    base = path[:-4]
    if not os.path.exists(base + '.json') or os.path.getmtime(base + '.json') < os.path.getmtime(path):
        subprocess.run(['whisper-cli', '-m', WHISPER_MODEL, '-t', '2', '-f', path, '-np', '-ojf', '-of', base], check=True,
                       capture_output=True, cwd=ROOT)
    result = read_json(base + '.json')
    return ''.join(segment['text'] for segment in result['transcription']).strip()


# Transcripts of WAV files, workers at a time (two threads each) - {path: text}
def transcribe_all(paths, workers=6):
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return dict(zip(paths, pool.map(transcribe, paths)))


# Words for scoring - lowercase, apostrophes dropped, other punctuation a word break
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


# The panel texts - [{id, text}]
def panel_texts():
    return read_json(PANEL_TEXTS_PATH)['texts']


# Speak a batch ([{text, wav, seed, speaker}]) with an export (tools/speak.js) - its speed report
def speak(model_path, batch, batch_path, rate):
    with open(batch_path, 'w', encoding='utf-8') as file:
        json.dump(batch, file)
    command = ['node', os.path.join(ROOT, 'tools', 'speak.js'), '-m', model_path, '--batch', batch_path, '-r', str(rate)]
    process = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(process.stdout.strip().splitlines()[-1])


def score_model(model_path, out_dir, settings, whisper_workers=6):
    """Speak and score an export (model_path - its manifest and weights without the extension) - {panelSpeed, speakers
    ({key: {wer}}), utterances, panel ({wer})}; settings - load_config's ({speakers: [key], rate})"""
    keys = [speaker['key'] for speaker in read_json(model_path + '.json')['speakers']]
    items = []
    for speaker in settings['speakers']:
        for number, text in enumerate(panel_texts()):
            items.append({'speaker': speaker, 'text': text['text'], 'wav': os.path.join(out_dir, f'{speaker}-{text["id"]}.wav'),
                          'seed': number + 1})
    batch = [{'text': item['text'], 'wav': item['wav'], 'seed': item['seed'], 'speaker': keys.index(item['speaker'])} for item in items]
    result = {'panelSpeed': speak(model_path, batch, os.path.join(out_dir, 'batch.json'), settings['rate'])}
    paths = [item['wav'] for item in items]
    transcripts = transcribe_all(paths, whisper_workers)
    speakers = {}
    for speaker in settings['speakers']:
        group = [item for item in items if item['speaker'] == speaker]
        errors = sum(edit_distance(words(item['text']), words(transcripts[item['wav']])) for item in group)
        count = sum(len(words(item['text'])) for item in group)
        speakers[speaker] = {'wer': 100 * errors / max(1, count)}
    result['speakers'] = speakers
    result['utterances'] = {os.path.basename(item['wav']): {'transcript': transcripts[item['wav']]} for item in items}
    result['panel'] = {'wer': float(numpy.mean([row['wer'] for row in speakers.values()]))}
    for path in paths:
        os.remove(path)
    return result


#
# make train's evaluation and gate
#

# The settings - the recipe's (train/data/train.json) panel, and its speakers: the training speech's voices' keys (tts voices)
def load_config(path=RECIPE_PATH):
    recipe = read_json(path)
    return {**{key: value for key, value in recipe['panel'].items() if key != 'description'}, 'speakers': list(recipe['tts']['voices'])}


# A path relative to the repository root as an absolute path
def root_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# Append a progress line to EVAL_DIR/eval.log (and stderr)
def log_line(text):
    os.makedirs(EVAL_DIR, exist_ok=True)
    g2p.log_line(os.path.join(EVAL_DIR, 'eval.log'), text)


def evaluate_model(name, checkpoint, settings, whisper_workers=6, jobs=8):
    """Export, package, speak, and score a checkpoint (its export EVAL_DIR/<name>/EXPORT_VOICE.json and .bin, its module
    EVAL_DIR/<name>/MODULE_NAME), or speak and score a packaged module (a .js path - unpacked to its export, tools/unpack.js)
    as packaged, and measure the module's gap to its source (quality.gap - jobs renderers) - its result (also
    EVAL_DIR/<name>/result.json)"""
    out_dir = os.path.join(EVAL_DIR, name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    start = time.time()
    model_path = os.path.join(out_dir, EXPORT_VOICE)
    if checkpoint.endswith('.js'):
        module_path = root_path(checkpoint)
        subprocess.run(['node', os.path.join(ROOT, 'tools', 'unpack.js'), module_path, model_path], check=True)
        with open(module_path, encoding='utf-8') as file:
            module = file.read()
    else:
        model, state = load_model(checkpoint)
        export(model, state, EXPORT_VOICE, out_dir)
        module = package(model_path)
        module_path = os.path.join(out_dir, MODULE_NAME)
        with open(module_path, 'w', encoding='utf-8') as file:
            file.write(module)
    parameters = read_json(model_path + '.json')['parameterCount']
    size = module_size(module)
    result = {'name': name, 'checkpoint': checkpoint, 'parameters': parameters,
              'size': {'moduleBytes': size['bytes'], 'moduleGzipBytes': size['gzipBytes']}}
    result.update(score_model(model_path, out_dir, settings, whisper_workers))
    result['gap'] = quality.gap(module_path, os.path.join(out_dir, 'gap'), settings['gap']['perVoice'], jobs)
    result['panel']['gap'] = result['gap']['fdL12']
    result['seconds'] = round(time.time() - start, 1)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    return result


# An evaluated model's result (EVAL_DIR/<name>/result.json)
def load_result(name):
    return read_json(os.path.join(EVAL_DIR, name, 'result.json'))


# A packaged module's cached evaluation - its path in the training speech's directory (panel-<key>.json), the key a hash
# of the module, the panel's settings, the panel texts, the corpus's selection (the gap's clips), and the panel's code
def cached_result_path(module_path, settings):
    tts_settings = tts.load_settings()
    recipe = read_json(RECIPE_PATH)
    directory = tts.corpus_dir(tts_settings)
    files = [os.path.join(directory, f'selection-{tts.selection_key(tts_settings, recipe)}.json'), PANEL_TEXTS_PATH, module_path]
    key = tts.inputs_key({'settings': settings}, files + tts.code_files('panel'))
    return os.path.join(directory, f'panel-{key}.json')


# The eval subcommand - a model (NAME=CHECKPOINT) through the panel; a packaged module's result from the cache when it's
# there (make train's baseline - the shipped model, evaluated once)
def evaluate(args):
    settings = load_config(args.config)
    name, _, checkpoint = args.model.partition('=')
    cached = cached_result_path(root_path(checkpoint), settings) if checkpoint.endswith('.js') else None
    if cached is not None and os.path.exists(cached):
        result = {**read_json(cached), 'name': name}
        os.makedirs(os.path.join(EVAL_DIR, name), exist_ok=True)
        with open(os.path.join(EVAL_DIR, name, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=1)
        log_line(f'{name}: {checkpoint} is evaluated ({display_path(cached)})')
    else:
        log_line(f'{name}: evaluating {checkpoint}')
        result = evaluate_model(name, checkpoint, settings, whisper_workers=args.whisper_workers, jobs=args.jobs)
        if cached is not None:
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            with open(cached + '.part', 'w', encoding='utf-8') as file:
                json.dump(result, file, indent=1)
            os.replace(cached + '.part', cached)
    measures = result['panel']
    log_line(f'{name}: {result["parameters"]} parameters, module {result["size"]["moduleBytes"]} bytes '
             f'({result["size"]["moduleGzipBytes"]} gzip); WER {measures["wer"]:.2f}%, gap {measures["gap"]:.0f} '
             f'({result["seconds"]:.0f} s)')
    return 0


# A result's summary row
def summary_row(result):
    measures = result['panel']
    return {'name': result['name'], 'parameters': result['parameters'], 'moduleBytes': result['size']['moduleBytes'],
            'moduleGzipBytes': result['size']['moduleGzipBytes'], 'wer': measures['wer'], 'gap': measures['gap'],
            'realTimeTotal': result['panelSpeed']['realTimeTotal']}


# The gate subcommand - an evaluated model within the recipe's tolerance of another (EVAL_DIR/summary.json): its word
# error rate at most tolerance wer points above the basis's, and its gap at most the tolerance's share above; the exit
# status 1 if not
def gate(args):
    tolerance = load_config(args.config)['tolerance']
    voice, basis = summary_row(load_result(args.model)), summary_row(load_result(args.against))
    voice['within'] = {'wer': voice['wer'] <= basis['wer'] + tolerance['wer'], 'gap': voice['gap'] <= basis['gap'] * (1 + tolerance['gap'])}
    passed = all(voice['within'].values())
    summary = {'name': args.model, 'against': args.against, 'tolerance': tolerance, 'rows': [voice, basis], 'passed': passed}
    with open(os.path.join(EVAL_DIR, 'summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=1)
    for row in summary['rows']:
        log_line(f'{row["name"]}: {row["parameters"]} parameters, {row["moduleBytes"]} bytes ({row["moduleGzipBytes"]} gzip), '
                 f'WER {row["wer"]:.2f}%, gap {row["gap"]:.0f}, real time {row["realTimeTotal"]:.1f}x')
    failed = [key for key, within in voice['within'].items() if not within]
    log_line(f'gate: {args.model} {"passed" if passed else "FAILED (" + ", ".join(failed) + ")"} - WER {voice["wer"]:.2f}% '
             f'({args.against} {basis["wer"]:.2f}%), gap {voice["gap"]:.0f} ({args.against} {basis["gap"]:.0f}, '
             f'{100 * (voice["gap"] / basis["gap"] - 1):+.1f}%) (tolerance: WER +{tolerance["wer"]}, gap +{100 * tolerance["gap"]:g}%)')
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.panel', description="The listener panel and make train's gate")
    parser.add_argument('--config', default=RECIPE_PATH, help='the recipe (default train/data/train.json - its panel)')
    subparsers = parser.add_subparsers(dest='command', required=True)
    sub = subparsers.add_parser('eval', help='a model through the panel')
    sub.add_argument('model', help='NAME=CHECKPOINT - a checkpoint, or a packaged module (.js); its result in <eval>/NAME')
    sub.add_argument('--whisper-workers', type=int, default=6, help='whisper.cpp processes at a time (two threads each)')
    sub.add_argument('-j', '--jobs', type=int, default=8, help="the gap's Node renderers")
    sub.set_defaults(func=evaluate)
    sub = subparsers.add_parser('gate', help="an evaluated model within the tolerance of another's scores")
    sub.add_argument('model', help="the evaluated model's name")
    sub.add_argument('--against', required=True, help="the basis - an evaluated model's name (make train's: baseline)")
    sub.set_defaults(func=gate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
