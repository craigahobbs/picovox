# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
import warnings

from picovox_train import ROOT, SOURCE_DIR, smoke

from . import helpers


# Each stage's outputs (relative to the smoke run's directory)
OUTPUTS = {
    'g2p-train': 'g2p/done.json', 'g2p-export': 'g2pModel.js', 'context-g2p': 'context-g2p.pt', 'context-train': 'context.pt',
    'context-export': 'g2pModel-context.js', 'data-build': 'multi/index.json', 'teacher-a': 'runs/smoke-a/last.pt',
    'teacher-b': 'runs/smoke-b/last.pt', 'teacher-options': 'runs/smoke-options/ema-best.pt', 'distill-targets': 'targets/index.json',
    'distill-train': 'runs/distill-s32/best.pt', 'distill-polish': 'runs/distill-s32-polish/best.pt',
    'distill-options': 'runs/distill-s32-options/ema-best.pt', 'export': 'export/smoke.json', 'package': 'voiceModel.js'
}


# A smoke test of the smoke test - the entry points mocked: each stage writes its outputs, or fails (exit 1) or writes
# none (exit 0); with or without the context labels (in a temporary repository root)
class SmokeRun:

    def __init__(self, failing=(), missing=(), labels=True):
        self.failing = set(failing)
        self.missing = set(missing)
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = os.path.join(self.temp.name, 'root')
        self.out = os.path.join(self.temp.name, 'out')
        os.makedirs(os.path.join(self.root, 'data', 'nn'))
        shutil.copy(os.path.join(ROOT, 'data', 'nn', 'train.json'), os.path.join(self.root, 'data', 'nn', 'train.json'))
        if labels:
            os.makedirs(os.path.join(self.root, 'build', 'nn', 'context'))
            with open(os.path.join(self.root, 'build', 'nn', 'context', 'labels.json'), 'w', encoding='utf-8') as file:
                file.write('{}')
        self.calls = []

    def fake_run(self, command, cwd, env, stdout, stderr, check):
        name = os.path.basename(stdout.name)[:-len('.log')]
        self.calls.append((name, command))
        assert (cwd, env['PYTHONPATH'], env['NN_RUNS_DIR'], stderr, check) == \
            (self.root, SOURCE_DIR, os.path.join(self.out, 'runs'), subprocess.STDOUT, False)
        assert command[:3] == [smoke.PYTHON, '-B', '-m']
        if name in self.failing:
            return subprocess.CompletedProcess(command, 1)
        if name not in self.missing:
            path = os.path.join(self.out, OUTPUTS[name])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as file:
                file.write('x')
        return subprocess.CompletedProcess(command, 0)

    def main(self, *arguments):
        with unittest.mock.patch.object(smoke, 'ROOT', self.root), \
                unittest.mock.patch.object(smoke.subprocess, 'run', side_effect=self.fake_run), helpers.captured() as (stdout, _):
            status = smoke.main(['--out', self.out, *arguments])
        return status, stdout.getvalue()

    def names(self):
        return [name for name, _ in self.calls]

    def command(self, name):
        return next(command for stage, command in self.calls if stage == name)

    def cleanup(self):
        self.temp.cleanup()


class TestSmoke(unittest.TestCase):

    def setUp(self):
        self.runs = []

    def tearDown(self):
        for run in self.runs:
            run.cleanup()

    def smoke_run(self, **kwargs):
        run = SmokeRun(**kwargs)
        self.runs.append(run)
        return run

    def test_all(self):
        run = self.smoke_run()
        # (a previous run's output is removed)
        os.makedirs(os.path.join(run.out, 'old'))
        status, output = run.main()
        self.assertEqual(status, 0)
        self.assertEqual(run.names(), list(OUTPUTS))
        self.assertFalse(os.path.exists(os.path.join(run.out, 'old')))
        lines = output.splitlines()
        self.assertEqual(lines[-1], 'smoke: ok')
        self.assertRegex(lines[0], r'^ok g2p-train \([0-9]+ s\)$')
        self.assertEqual(run.command('g2p-train')[3:], ['picovox_train.g2p', 'train', '--out', os.path.join(run.out, 'g2p'),
                                                        '--steps', '3'])
        self.assertEqual(run.command('context-g2p')[6], os.path.join(run.root, 'build', 'nn', 'context', 'labels.json'))
        self.assertIn('--adversarial-start', run.command('teacher-b'))
        # (every training step option, in a stage-A run and a student)
        for name in ('teacher-options', 'distill-options'):
            self.assertIn(' '.join(smoke.OPTIONS), ' '.join(run.command(name)))
        self.assertEqual(run.command('teacher-options')[-2:], ['--crop-warmup', '2:1.0'])
        # The distillation's recipe - a short book, few validation items
        with open(os.path.join(run.out, 'train.json'), encoding='utf-8') as file:
            recipe = json.load(file)
        self.assertEqual(recipe['distill']['targets']['books'], os.path.join(run.out, 'books'))
        self.assertEqual(recipe['distill']['targets']['valItems'], 4)
        self.assertNotIn('texts', recipe['distill']['targets'])
        self.assertEqual(recipe['distill']['train']['batch'], 8)
        self.assertEqual(recipe['distill']['polish']['val-every'], 2)
        with open(os.path.join(run.out, 'books', 'book.txt'), encoding='utf-8') as file:
            self.assertEqual(file.read(), smoke.BOOK_TEXT)

    def test_failures(self):
        # A failed stage stops the stages that need it; the others run
        run = self.smoke_run(failing=['g2p-train', 'teacher-a'])
        status, output = run.main()
        self.assertEqual(status, 1)
        self.assertEqual(run.names(), ['g2p-train', 'data-build', 'teacher-a', 'teacher-options'])
        self.assertEqual(output.splitlines()[-1], 'smoke: FAILED - g2p-train, teacher-a')
        self.assertIn(f'FAILED g2p-train (0 s) - exit 1, missing [{os.path.join(run.out, "g2p", "done.json")!r}], see '
                      f'{os.path.join(run.out, "g2p-train.log")}', output)

    def test_missing_outputs(self):
        # A stage that exits 0 without its outputs failed
        run = self.smoke_run(missing=['context-g2p', 'distill-train', 'export'])
        status, output = run.main()
        self.assertEqual(status, 1)
        self.assertEqual(run.names(), ['g2p-train', 'g2p-export', 'context-g2p', 'data-build', 'teacher-a', 'teacher-b',
                                       'teacher-options', 'distill-targets', 'distill-train', 'distill-options'])
        self.assertIn('FAILED context-g2p (0 s) - exit 0, missing', output)

    def test_later_failures(self):
        run = self.smoke_run(failing=['context-train', 'teacher-b', 'distill-targets'])
        self.assertEqual(run.main()[0], 1)
        self.assertEqual(run.names(), ['g2p-train', 'g2p-export', 'context-g2p', 'context-train', 'data-build', 'teacher-a',
                                       'teacher-b', 'teacher-options'])
        run = self.smoke_run(failing=['distill-targets', 'export'])
        self.assertEqual(run.main()[0], 1)
        self.assertEqual(run.names()[-1], 'distill-targets')

    def test_package_failure(self):
        run = self.smoke_run(failing=['export'])
        self.assertEqual(run.main()[0], 1)
        self.assertEqual(run.names()[-3:], ['distill-polish', 'distill-options', 'export'])

    def test_no_labels(self):
        # No context labels (nn.contextlabels not run) - the context stage is skipped
        run = self.smoke_run(labels=False)
        self.assertEqual(run.main('--only', 'g2p,context')[0], 0)
        self.assertEqual(run.names(), ['g2p-train', 'g2p-export'])

    def test_only(self):
        # The stages chosen; the distillation and package stages need the teacher and the student; --keep keeps the output
        run = self.smoke_run()
        os.makedirs(os.path.join(run.out, 'old'))
        self.assertEqual(run.main('--only', 'g2p,distill,package', '--keep')[0], 0)
        self.assertEqual(run.names(), ['g2p-train', 'g2p-export'])
        self.assertTrue(os.path.exists(os.path.join(run.out, 'old')))
        self.assertEqual(run.main('--only', 'teacher,package', '--keep')[0], 0)
        self.assertEqual(run.names()[2:], ['teacher-a', 'teacher-b', 'teacher-options'])
        self.assertEqual(run.main('--only', 'distill,package', '--keep')[0], 0)
        self.assertEqual(run.names()[5:], ['distill-targets', 'distill-train', 'distill-polish', 'distill-options', 'export',
                                           'package'])

    def test_default_out(self):
        parser_default = os.path.join(ROOT, 'build', 'nn', 'smoke')
        with unittest.mock.patch.object(smoke.shutil, 'rmtree') as rmtree, \
                unittest.mock.patch.object(smoke.os, 'makedirs') as makedirs, helpers.captured():
            self.assertEqual(smoke.main(['--only', 'none']), 0)
        rmtree.assert_called_once_with(parser_default, ignore_errors=True)
        makedirs.assert_called_once_with(parser_default, exist_ok=True)

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['smoke', '--bad']), helpers.captured() as (_, stderr), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.smoke', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('unrecognized arguments: --bad', stderr.getvalue())
