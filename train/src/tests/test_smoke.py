# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

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
    'g2p-train': 'g2p/done.json', 'context-g2p': 'context-g2p.pt', 'context-train': 'context.pt', 'context-export': 'g2pModel.js',
    'data-build': 'data/tts/index.json', 'voice-a': 'runs/smoke-a/last.pt', 'voice-b': 'runs/smoke-b/last.pt',
    'export': 'export/smoke.json', 'package': 'voiceModel.js'
}


# A smoke test of the smoke test - the entry points mocked: each stage writes its outputs, or fails (exit 1) or writes
# none (exit 0); with or without the context labels (in a temporary repository root), and with or without the training
# speech's aligned corpus
class SmokeRun:

    def __init__(self, failing=(), missing=(), labels=True, corpus=True):
        self.failing = set(failing)
        self.missing = set(missing)
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = os.path.join(self.temp.name, 'root')
        self.out = os.path.join(self.temp.name, 'out')
        os.makedirs(os.path.join(self.root, 'train', 'data'))
        shutil.copy(os.path.join(ROOT, 'train', 'data', 'train.json'), os.path.join(self.root, 'train', 'data', 'train.json'))
        if labels:
            os.makedirs(os.path.join(self.root, 'build', 'nn', 'context'))
            with open(os.path.join(self.root, 'build', 'nn', 'context', 'labels.json'), 'w', encoding='utf-8') as file:
                file.write('{}')
        self.corpus = os.path.join(self.temp.name, 'tts')
        if corpus:
            os.makedirs(self.corpus)
            with open(os.path.join(self.corpus, 'alignments.jsonl'), 'w', encoding='utf-8') as file:
                file.write('{}\n')
        self.calls = []
        self.data_dirs = {}

    def fake_run(self, command, cwd, env, stdout, stderr, check):
        name = os.path.basename(stdout.name)[:-len('.log')]
        self.calls.append((name, command))
        self.data_dirs[name] = env.get('NN_DATA_DIR')
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
                unittest.mock.patch.object(smoke.tts, 'corpus_dir', return_value=self.corpus), \
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
        self.assertIn('--adversarial-start', run.command('voice-b'))
        # The data stage's clips, and the voice stages on its data
        self.assertEqual(run.command('data-build')[3:], ['picovox_train.tts', 'build', '--out', os.path.join(run.out, 'data', 'tts'),
                                                         '--limit', '48', '-j', '2'])
        self.assertEqual(run.data_dirs['voice-a'], os.path.join(run.out, 'data'))
        # (the package of stage B's last checkpoint)
        self.assertEqual(run.command('export')[4], os.path.join(run.out, 'runs', 'smoke-b', 'last.pt'))

    def test_failures(self):
        # A failed stage stops the stages that need it; the others run
        run = self.smoke_run(failing=['g2p-train', 'voice-a'])
        status, output = run.main()
        self.assertEqual(status, 1)
        self.assertEqual(run.names(), ['g2p-train', 'data-build', 'voice-a'])
        self.assertEqual(output.splitlines()[-1], 'smoke: FAILED - g2p-train, voice-a')
        self.assertIn(f'FAILED g2p-train (0 s) - exit 1, missing [{os.path.join(run.out, "g2p", "done.json")!r}], see '
                      f'{os.path.join(run.out, "g2p-train.log")}', output)

    def test_missing_outputs(self):
        # A stage that exits 0 without its outputs failed
        run = self.smoke_run(missing=['context-g2p', 'voice-b'])
        status, output = run.main()
        self.assertEqual(status, 1)
        self.assertEqual(run.names(), ['g2p-train', 'context-g2p', 'data-build', 'voice-a', 'voice-b'])
        self.assertIn('FAILED context-g2p (0 s) - exit 0, missing', output)

    def test_later_failures(self):
        run = self.smoke_run(failing=['context-train', 'voice-b'])
        self.assertEqual(run.main()[0], 1)
        self.assertEqual(run.names(), ['g2p-train', 'context-g2p', 'context-train', 'data-build', 'voice-a', 'voice-b'])

    def test_package_failure(self):
        run = self.smoke_run(failing=['export'])
        self.assertEqual(run.main()[0], 1)
        self.assertEqual(run.names()[-3:], ['voice-a', 'voice-b', 'export'])

    def test_data_failure(self):
        # A failed data build - the voice stages read build/nn/data
        run = self.smoke_run(failing=['data-build'])
        self.assertEqual(run.main('--only', 'data,voice')[0], 1)
        self.assertEqual(run.names(), ['data-build', 'voice-a', 'voice-b'])
        self.assertIsNone(run.data_dirs['voice-a'])

    def test_no_corpus(self):
        # No aligned corpus (picovox_train.tts generate and align not run) - the data stage is skipped, the voice stages
        # read build/nn/data
        run = self.smoke_run(corpus=False)
        self.assertEqual(run.main('--only', 'data,voice')[0], 0)
        self.assertEqual(run.names(), ['voice-a', 'voice-b'])
        self.assertIsNone(run.data_dirs['voice-a'])

    def test_no_labels(self):
        # No context labels (nn.contextlabels not run) - the context stage is skipped
        run = self.smoke_run(labels=False)
        self.assertEqual(run.main('--only', 'g2p,context')[0], 0)
        self.assertEqual(run.names(), ['g2p-train'])

    def test_only(self):
        # The stages chosen; the package stage needs the voice's stage B; --keep keeps the output
        run = self.smoke_run()
        os.makedirs(os.path.join(run.out, 'old'))
        self.assertEqual(run.main('--only', 'g2p,package', '--keep')[0], 0)
        self.assertEqual(run.names(), ['g2p-train'])
        self.assertTrue(os.path.exists(os.path.join(run.out, 'old')))
        self.assertEqual(run.main('--only', 'voice', '--keep')[0], 0)
        self.assertEqual(run.names()[1:], ['voice-a', 'voice-b'])
        self.assertEqual(run.main('--only', 'package', '--keep')[0], 0)
        self.assertEqual(run.names()[3:], ['export', 'package'])

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
