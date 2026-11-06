# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training pipeline's smoke test - every make-train entry point run for a few steps on the CPU, into a scratch
directory, each checked to finish and write its outputs:

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.smoke [--out build/nn/smoke] \
        [--only g2p,context,data,voice,package]

- g2p - picovox_train.g2p train (each stage 3 steps)
- context - picovox_train.context train-g2p and train-context (3 steps) and export, from the g2p stage (when build/train/context has the
  context labels - picovox_train.contextlabels)
- data - picovox_train.tts build --out on a few clips of the training speech (when its corpus is generated and aligned -
  picovox_train.tts generate, align; the smoke test never runs the engine)
- voice - picovox_train.trainmulti stage A (3 steps), then stage B from it (the critic)
- package - picovox_train.export and picovox_train.package of stage B

The voice stages read the training data the data stage built (<out>/data/tts), else build/nn/data/tts; everything is
written under --out (runs go to <out>/runs - NN_RUNS_DIR).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from . import ROOT, SOURCE_DIR, tts


PYTHON = sys.executable
STAGES = ('g2p', 'context', 'data', 'voice', 'package')

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
    env = {**os.environ, 'NN_RUNS_DIR': runs, 'OMP_NUM_THREADS': '2', 'PYTHONPATH': SOURCE_DIR}
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
            # The heteronym models (context.py) on the context labels, if make train built them (picovox_train.contextlabels -
            # its TRAIN_DIR's default, build/train/context)
            labels = os.path.join(ROOT, 'build', 'train', 'context', 'labels.json')
            if 'context' in stages and os.path.exists(labels):
                g2p_checkpoint, context_checkpoint = os.path.join(out, 'context-g2p.pt'), os.path.join(out, 'context.pt')
                if run('context-g2p', ['picovox_train.context', 'train-g2p', '--labels', labels, '--init',
                                       os.path.join(g2p_dir, 'focus.pt'), '--out', g2p_checkpoint, '--settings', '{"steps": 3}'],
                       [g2p_checkpoint]) and \
                        run('context-train', ['picovox_train.context', 'train-context', '--labels', labels, '--out', context_checkpoint,
                                              '--settings', '{"steps": 3, "quantSteps": 2}'], [context_checkpoint]):
                    run('context-export', ['picovox_train.context', 'export', '--g2p', g2p_checkpoint, '--context', context_checkpoint,
                                           '--out', os.path.join(out, 'g2pModel.js')], [os.path.join(out, 'g2pModel.js')])

    # The training data - a few clips of the corpus (if generated and aligned)
    if 'data' in stages and os.path.exists(os.path.join(tts.corpus_dir(tts.load_settings()), 'alignments.jsonl')):
        corpus = os.path.join(out, 'data', 'tts')
        if run('data-build', ['picovox_train.tts', 'build', '--out', corpus, '--limit', '48', '-j', '2'],
               [os.path.join(corpus, 'index.json')]):
            env['NN_DATA_DIR'] = os.path.join(out, 'data')

    common = ['--device', 'cpu', '--batch', '4', '--val-count', '4', '--keep-every', '0', '--log-every', '1']
    voice = os.path.join(runs, 'smoke-b', 'last.pt')
    if 'voice' in stages:
        if run('voice-a', ['picovox_train.trainmulti', '--name', 'smoke-a', '--recipe', 'voice.a', '--steps', '3',
                           '--val-every', '3'] + common, [os.path.join(runs, 'smoke-a', 'last.pt')]):
            run('voice-b', ['picovox_train.trainmulti', '--name', 'smoke-b', '--recipe', 'voice.b', '--init',
                            os.path.join(runs, 'smoke-a', 'last.pt'), '--steps', '3', '--val-every', '3', '--adversarial-start',
                            '1'] + common, [voice])

    if 'package' in stages and os.path.exists(voice):
        export_dir = os.path.join(out, 'export')
        if run('export', ['picovox_train.export', voice, '--voice', 'smoke', '--out', export_dir],
               [os.path.join(export_dir, 'smoke.json')]):
            run('package', ['picovox_train.package', os.path.join(export_dir, 'smoke'), '-o', os.path.join(out, 'voiceModel.js')],
                [os.path.join(out, 'voiceModel.js')])

    print('smoke: ' + (f'FAILED - {", ".join(failures)}' if failures else 'ok'), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
