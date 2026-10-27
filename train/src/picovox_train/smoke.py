# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training pipeline's smoke test - every make-train entry point run for a few steps on the CPU, into a scratch
directory, each checked to finish and write its outputs:

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.smoke [--out build/nn/smoke] \
        [--only g2p,context,data,teacher,distill,package]

- g2p - picovox_train.g2p train (each stage 3 steps)
- context - picovox_train.context train-g2p and train-context (3 steps) and export, from the g2p stage (when build/nn/context has the
  context labels - picovox_train.contextlabels)
- data - picovox_train.multi build --out on a few utterances of two speakers
- teacher - picovox_train.trainmulti stage A (3 steps), then stage B from it (the critic, the emotional speech)
- distill - picovox_train.distill targets (32 items, one short book) from the stage-B teacher, train (3 steps), and polish
  (2 steps)
- package - picovox_train.export and picovox_train.package of the polished student

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

from . import ROOT, SOURCE_DIR


PYTHON = sys.executable
STAGES = ('g2p', 'context', 'data', 'teacher', 'distill', 'package')

# A short public-domain text for the distillation targets' book sentences
BOOK_TEXT = '''Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and
nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.
It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about
the mouth; whenever it is a damp, drizzly November in my soul, then, I account it high time to get to sea as soon as I
can.
'''


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.smoke', description='Smoke-test the training pipeline on the CPU')
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
           'PYTHONPATH': SOURCE_DIR}
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
        if run('g2p-train', ['picovox_train.g2p', 'train', '--out', g2p_dir, '--steps', '3'], [os.path.join(g2p_dir, 'done.json')]):
            # The heteronym models (context.py) on the context labels, if built (picovox_train.contextlabels - build/nn/context)
            labels = os.path.join(ROOT, 'build', 'nn', 'context', 'labels.json')
            if 'context' in stages and os.path.exists(labels):
                g2p_checkpoint, context_checkpoint = os.path.join(out, 'context-g2p.pt'), os.path.join(out, 'context.pt')
                if run('context-g2p', ['picovox_train.context', 'train-g2p', '--labels', labels, '--init',
                                       os.path.join(g2p_dir, 'focus.pt'), '--out', g2p_checkpoint, '--settings', '{"steps": 3}'],
                       [g2p_checkpoint]) and \
                        run('context-train', ['picovox_train.context', 'train-context', '--labels', labels, '--out', context_checkpoint,
                                              '--settings', '{"steps": 3, "quantSteps": 2}'], [context_checkpoint]):
                    run('context-export', ['picovox_train.context', 'export', '--g2p', g2p_checkpoint, '--context', context_checkpoint,
                                           '--out', os.path.join(out, 'g2pModel.js')], [os.path.join(out, 'g2pModel.js')])

    if 'data' in stages:
        corpus = os.path.join(out, 'multi')
        run('data-build', ['picovox_train.multi', 'build', '--out', corpus, '--speakers', 'arctic-slt', 'libritts-5895', '--limit', '3',
                           '-j', '2'], [os.path.join(corpus, 'index.json')])

    common = ['--device', 'cpu', '--batch', '4', '--val-count', '4', '--keep-every', '0', '--log-every', '1']
    teacher = os.path.join(runs, 'smoke-b', 'last.pt')
    if 'teacher' in stages:
        if run('teacher-a', ['picovox_train.trainmulti', '--name', 'smoke-a', '--recipe', 'teacher.a', '--steps', '3',
                             '--val-every', '3'] + common, [os.path.join(runs, 'smoke-a', 'last.pt')]):
            run('teacher-b', ['picovox_train.trainmulti', '--name', 'smoke-b', '--recipe', 'teacher.b', '--init',
                              os.path.join(runs, 'smoke-a', 'last.pt'), '--steps', '3', '--val-every', '3', '--adversarial-start',
                              '1'] + common, [teacher])

    student = os.path.join(runs, 'distill-s32-polish', 'best.pt')
    if 'distill' in stages and os.path.exists(teacher):
        books = os.path.join(out, 'books')
        os.makedirs(books, exist_ok=True)
        with open(os.path.join(books, 'book.txt'), 'w', encoding='utf-8') as file:
            file.write(BOOK_TEXT)
        with open(os.path.join(ROOT, 'train', 'data', 'train.json'), encoding='utf-8') as file:
            recipe = json.load(file)
        recipe['distill']['targets'].update({'books': books, 'valItems': 4})
        del recipe['distill']['targets']['texts']
        recipe['distill']['train'].update({'batch': 8, 'valEvery': 3})
        recipe['distill']['polish'].update({'val-every': 2})
        config_path = os.path.join(out, 'train.json')
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(recipe, file)
        targets = os.path.join(out, 'targets')
        distill = ['--config', config_path, '--device', 'cpu']
        if run('distill-targets', ['picovox_train.distill', 'targets', '--teacher', teacher, '--out', targets, '--limit', '32'] + distill,
               [os.path.join(targets, 'index.json')]):
            if run('distill-train', ['picovox_train.distill', 'train', '--student', 's32', '--steps', '3', '--targets', targets] + distill,
                   [os.path.join(runs, 'distill-s32', 'best.pt')]):
                run('distill-polish', ['picovox_train.distill', 'polish', '--student', 's32', '--steps', '2'] + distill, [student])

    if 'package' in stages and os.path.exists(student):
        export_dir = os.path.join(out, 'export')
        if run('export', ['picovox_train.export', student, '--voice', 'smoke', '--out', export_dir],
               [os.path.join(export_dir, 'smoke.json')]):
            run('package', ['picovox_train.package', os.path.join(export_dir, 'smoke'), '-o', os.path.join(out, 'voiceModel.js')],
                [os.path.join(out, 'voiceModel.js')])

    print('smoke: ' + (f'FAILED - {", ".join(failures)}' if failures else 'ok'), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
