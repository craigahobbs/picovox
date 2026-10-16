# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The training pipeline's smoke test - every make-train entry point run for a few steps on the CPU, into a scratch
directory, each checked to finish and write its outputs:

    build/venv-nn/bin/python -B -m nn.smoke [--out build/nn/smoke] [--only g2p,data,teacher,distill,package]

- g2p - nn.g2p train (each stage 3 steps) and export
- data - nn.multi build --out on a few utterances of two speakers
- teacher - nn.trainmulti stage A (3 steps), then stage B from it (the critic, the emotional speech)
- distill - nn.distill targets (32 items, one short book) from the stage-B teacher, train (3 steps), polish (2 steps)
- package - nn.export and nn.package of the polished student

The voice stages read the training data (build/nn/data/multi and cremad); everything is written under --out (runs go
to <out>/runs - NN_RUNS_DIR).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from .frontend import ROOT


PYTHON = sys.executable
STAGES = ('g2p', 'data', 'teacher', 'distill', 'package')

# A short public-domain text for the distillation targets' book sentences
BOOK_TEXT = '''Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and
nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.
It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about
the mouth; whenever it is a damp, drizzly November in my soul, then, I account it high time to get to sea as soon as I
can.
'''


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.smoke', description='Smoke-test the training pipeline on the CPU')
    parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'nn', 'smoke'))
    parser.add_argument('--only', default=','.join(STAGES), help='the stages to run (comma-separated)')
    parser.add_argument('--keep', action='store_true', help='keep the output directory from a previous run')
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    runs = os.path.join(out, 'runs')
    env = {**os.environ, 'NN_RUNS_DIR': runs, 'NN_DISTILL_DIR': os.path.join(out, 'distill'), 'OMP_NUM_THREADS': '2',
           'PYTHONPATH': ROOT}
    failures = []

    # Run an entry point; check its outputs
    def run(name, command, outputs):
        start = time.time()
        with open(os.path.join(out, f'{name}.log'), 'w', encoding='utf-8') as log:
            status = subprocess.run([PYTHON, '-B', '-m'] + command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    check=False).returncode
        missing = [path for path in outputs if not os.path.exists(path)]
        ok = status == 0 and not missing
        print(f'{"ok" if ok else "FAILED"} {name} ({time.time() - start:.0f} s)' +
              ('' if ok else f' - exit {status}, missing {missing}, see {os.path.join(out, name + ".log")}'), flush=True)
        if not ok:
            failures.append(name)
        return ok

    stages = args.only.split(',')
    if 'g2p' in stages:
        g2p_dir = os.path.join(out, 'g2p')
        if run('g2p-train', ['nn.g2p', 'train', '--out', g2p_dir, '--steps', '3'], [os.path.join(g2p_dir, 'done.json')]):
            run('g2p-export', ['nn.g2p', 'export', g2p_dir, '--out', os.path.join(out, 'g2pModel.js')],
                [os.path.join(out, 'g2pModel.js')])

    if 'data' in stages:
        corpus = os.path.join(out, 'multi')
        run('data-build', ['nn.multi', 'build', '--out', corpus, '--speakers', 'arctic-slt', 'libritts-5895', '--limit', '3',
                           '-j', '2'], [os.path.join(corpus, 'index.json')])

    common = ['--device', 'cpu', '--batch', '4', '--val-count', '4', '--keep-every', '0', '--log-every', '1']
    teacher = os.path.join(runs, 'smoke-b', 'last.pt')
    if 'teacher' in stages:
        if run('teacher-a', ['nn.trainmulti', '--name', 'smoke-a', '--recipe', 'teacher.a', '--steps', '3', '--val-every', '3'] + common,
               [os.path.join(runs, 'smoke-a', 'last.pt')]):
            run('teacher-b', ['nn.trainmulti', '--name', 'smoke-b', '--recipe', 'teacher.b', '--init', os.path.join(runs, 'smoke-a', 'last.pt'),
                              '--steps', '3', '--val-every', '3', '--adversarial-start', '1'] + common,
                [teacher])

    student = os.path.join(runs, 'distill-s32-polish', 'best.pt')
    if 'distill' in stages and os.path.exists(teacher):
        books = os.path.join(out, 'books')
        os.makedirs(books, exist_ok=True)
        with open(os.path.join(books, 'book.txt'), 'w', encoding='utf-8') as file:
            file.write(BOOK_TEXT)
        with open(os.path.join(ROOT, 'data', 'nn', 'train.json'), encoding='utf-8') as file:
            recipe = json.load(file)
        recipe['distill']['targets'].update({'books': books, 'valItems': 4})
        recipe['distill']['train'].update({'batch': 8, 'valEvery': 3})
        recipe['distill']['polish'].update({'valEvery': 2})
        config_path = os.path.join(out, 'train.json')
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(recipe, file)
        targets = os.path.join(out, 'targets')
        distill = ['--config', config_path, '--device', 'cpu']
        if run('distill-targets', ['nn.distill', 'targets', '--teacher', teacher, '--out', targets, '--limit', '32'] + distill,
               [os.path.join(targets, 'index.json')]):
            if run('distill-train', ['nn.distill', 'train', '--student', 's32', '--steps', '3', '--targets', targets] + distill,
                   [os.path.join(runs, 'distill-s32', 'best.pt')]):
                run('distill-polish', ['nn.distill', 'polish', '--student', 's32', '--steps', '2'] + distill, [student])

    if 'package' in stages and os.path.exists(student):
        export_dir = os.path.join(out, 'export')
        if run('export', ['nn.export', student, '--voice', 'smoke', '--out', export_dir], [os.path.join(export_dir, 'smoke.json')]):
            run('package', ['nn.package', os.path.join(export_dir, 'smoke'), '-o', os.path.join(out, 'voiceModel.js')],
                [os.path.join(out, 'voiceModel.js')])

    print('smoke: ' + (f'FAILED - {", ".join(failures)}' if failures else 'ok'), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
