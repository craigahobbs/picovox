# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import tempfile
import unittest
import unittest.mock
import warnings

from picovox_train import stepcheck

from . import helpers
from .test_trainmulti import TrainData


DATA = None


def setUpModule():  # pylint: disable=invalid-name
    global DATA  # pylint: disable=global-statement
    DATA = TrainData()
    DATA.train('init', '--steps', '1')


def tearDownModule():  # pylint: disable=invalid-name
    DATA.cleanup()


# Run stepcheck.main in a working directory (the tree it checks), with the tests' data - (its output, stderr)
def run_stepcheck(directory, *arguments):
    out = os.path.join(directory, 'out.json')
    previous = os.getcwd()
    os.chdir(directory)
    try:
        with DATA.patched(), helpers.captured() as (_, stderr):
            status = stepcheck.main(['--init', DATA.run_path('init', 'last.pt'), '--out', out, *arguments])
    finally:
        os.chdir(previous)
    assert status == 0
    with open(out, encoding='utf-8') as file:
        return json.load(file), stderr.getvalue()


class TestStepcheck(unittest.TestCase):

    def test_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            result, stderr = run_stepcheck(directory, '--batches', '1', '--batch', '2')
            again, _ = run_stepcheck(directory, '--batches', '1', '--batch', '2')
        self.assertEqual(result, again)
        self.assertEqual(result['device'], 'cpu')
        self.assertEqual(result['config'], {})
        self.assertEqual(len(result['batches']), 1)
        row = result['batches'][0]
        self.assertEqual(row['corpus'], 'multi')
        self.assertEqual(sorted(row['items']), ['arctic-bdl/0002', 'arctic-slt/0002'])
        for mode in ('eval', 'train'):
            self.assertEqual(set(row[mode]) >= {'spectral', 'total', 'durPhone', 'b1'}, True)
            float.fromhex(row[mode]['total'])
        self.assertNotEqual(row['eval']['total'], row['train']['total'])
        self.assertGreater(float.fromhex(row['gradNorm']), 0)
        self.assertEqual(json.loads(stderr.splitlines()[0])['corpus'], 'multi')

    def test_emotional_corpus(self):
        # The emotional corpus too, when the tree has built it; a model config override
        with tempfile.TemporaryDirectory() as directory:
            cremad = os.path.join(directory, 'build', 'nn', 'data', 'cremad')
            os.makedirs(cremad)
            with open(os.path.join(cremad, 'index.json'), 'w', encoding='utf-8') as file:
                file.write('{}')
            result, _ = run_stepcheck(directory, '--batches', '1', '--batch', '1', '--config', '{"jitter": 0.01}')
        self.assertEqual([row['corpus'] for row in result['batches']], ['multi', 'cremad'])
        self.assertEqual(result['config'], {'jitter': 0.01})
        self.assertEqual(len(result['batches'][1]['items']), 1)

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['stepcheck.py']), helpers.captured() as (_, stderr), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.stepcheck', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('--init', stderr.getvalue())
