# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
make train's trained models - the report that ends make train, and make train-install. make train trains both models to
their gates in TRAIN_DIR (TRAIN_DIR/g2pModel.js and voiceModel.js) and leaves lib/models/ unchanged; the report says
what each model is, how the voice scored against the shipped one, and whether each differs from the committed module.
make train-install replaces the committed modules with the trained ones that differ - the G2P below its comment header
(its provenance differs run to run), the voice byte for byte; make train-install-voice and train-install-g2p, one model.

    PYTHONPATH=train/src python3 -B -m picovox_train.trained report build/train                  # make train's report
    PYTHONPATH=train/src python3 -B -m picovox_train.trained install build/train [--model voice]  # make train-install
"""

import argparse
import gzip
import json
import os
import re
import shutil
import sys

from . import ROOT


# The committed modules
MODELS_DIR = os.path.join(ROOT, 'lib', 'models')


# The trained models in TRAIN_DIR - [(name, its stamp - the stage that writes it, its module)]
MODELS = [('voice', 'package.done', 'voiceModel.js'), ('G2P', 'g2p.done', 'g2pModel.js')]


# A generated module's lines below its comment header - its leading comment lines (its provenance - the checkpoints'
# digests - differs run to run; a line of base64 weights below it may start with // too)
def module_body(path):
    with open(path, encoding='utf-8') as file:
        lines = file.read().split('\n')
    return lines[next((index for index, line in enumerate(lines) if not line.startswith('//')), len(lines)):]


# Whether a trained module differs from the committed one - the G2P below its comment header, the voice (its provenance
# in its manifest) byte for byte
def differs(name, path, committed):
    if name == 'G2P':
        return module_body(path) != module_body(committed)
    with open(path, 'rb') as file, open(committed, 'rb') as committed_file:
        return file.read() != committed_file.read()


# A path for display - relative to the repository when it's in it
def display_path(path):
    relative = os.path.relpath(path, ROOT)
    return path if relative.startswith('..') else relative


# The trained models present in TRAIN_DIR - [(name, path, committed path)]
def trained_models(train_dir):
    return [(name, os.path.join(train_dir, module), os.path.join(MODELS_DIR, module))
            for name, stamp, module in MODELS if os.path.exists(os.path.join(train_dir, stamp))]


# A model module's size - "N bytes (N KB gzipped)"
def module_size(path):
    with open(path, 'rb') as file:
        data = file.read()
    return f'{len(data):,} bytes ({len(gzip.compress(data, compresslevel=9, mtime=0)) / 1000:.0f} KB gzipped)'


# The voice's lines - its size and the gate's panel scores against the shipped model (TRAIN_DIR/distill/summary.json)
def voice_lines(train_dir, path):
    with open(path, encoding='utf-8') as file:
        parameters = re.search(r'"parameterCount":(\d+)', file.read())
    lines = [f'{int(parameters.group(1)):,} parameters, {module_size(path)}' if parameters else module_size(path)]
    summary_path = os.path.join(train_dir, 'distill', 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as file:
            summary = json.load(file)
        rows = {row['name']: row for row in summary['rows']}
        student, shipped = rows.get('student'), rows.get(summary['against'])
        if student and shipped:
            lines.append(f'the listener panel: UTMOS {student["mos"]:.3f} (shipped {shipped["mos"]:.3f}, '
                         f'{student["mos"] - shipped["mos"]:+.3f}), WER {student["wer"]:.2f}% (shipped {shipped["wer"]:.2f}%, '
                         f'{student["wer"] - shipped["wer"]:+.2f}), similarity {student["similarity"]:.3f}')
            tolerance = summary['tolerance']
            lines.append(f'the gate: {"passed" if student.get("withinTolerance") else "FAILED"} (UTMOS at most {tolerance["mos"]} '
                         f'below the shipped model\'s, WER at most {tolerance["wer"]} points above, similarity at least '
                         f'{tolerance["similarity"]})')
    return lines


# The G2P's lines - its gate's check (TRAIN_DIR/logs/g2p.log's last "check:" line) when it differs
def g2p_lines(train_dir):
    log_path = os.path.join(train_dir, 'logs', 'g2p.log')
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding='utf-8') as file:
        checks = [line.strip() for line in file if line.startswith('check:')]
    return checks[-1:]


def command_report(args):
    models = trained_models(args.train_dir)
    print(f'make train: the trained models in {display_path(args.train_dir)} (lib/models/ unchanged)')
    for name, path, committed in models:
        changed = differs(name, path, committed)
        state = f'differs from {display_path(committed)}' if changed else \
            f'identical to {display_path(committed)}' + (' (below its header)' if name == 'G2P' else '')
        print(f'  The {name} - {display_path(path)}: {state}')
        lines = voice_lines(args.train_dir, path) if name == 'voice' else (g2p_lines(args.train_dir) if changed else [])
        for line in lines:
            print(f'    {line}')
    if not models:
        print('  None - no model has finished its training')
    elif len(models) < len(MODELS):
        print('Next: make train - make train-check and make train-install need both models')
    else:
        print('Next: the training dashboard (python3 -m http.server, then http://localhost:8000/tools/dashboard/), then make train-check '
              '(the package\'s tests with the trained models, and the voice lab staged with them to listen), then make train-install')
    return 0


def command_install(args):
    models = [model for model in trained_models(args.train_dir) if args.model in (None, model[0].lower())]
    installed = []
    for name, path, committed in models:
        if differs(name, path, committed):
            shutil.copyfile(path, committed)
            installed.append(name)
            print(f'install: {display_path(committed)} is now {display_path(path)}')
        else:
            print(f'install: {display_path(path)} is identical to {display_path(committed)}' +
                  (' (below its header)' if name == 'G2P' else '') + f' - {display_path(committed)} kept')
    if 'G2P' in installed:
        print('install: the voice\'s data, built with the earlier G2P, is rebuilt by the next make train')
    if installed:
        print('install: next, make commit, then commit lib/models/')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.trained', description='make train\'s trained models')
    commands = parser.add_subparsers(dest='command', required=True)
    report_parser = commands.add_parser('report', help='make train\'s report - each trained model, and whether it differs')
    report_parser.add_argument('train_dir', help='make train\'s TRAIN_DIR')
    report_parser.set_defaults(func=command_report)
    install_parser = commands.add_parser('install', help='make train-install - the trained models that differ into lib/models/')
    install_parser.add_argument('train_dir', help='make train\'s TRAIN_DIR')
    install_parser.add_argument('--model', choices=('voice', 'g2p'), help='one model (default: both)')
    install_parser.set_defaults(func=command_install)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
