# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import fcntl
import os
import runpy
import sys
import tempfile
import threading
import unittest
import unittest.mock
import warnings

from picovox_train import gpulock

from . import helpers


class TestGpulock(unittest.TestCase):

    def test_usage(self):
        for argv in ([], ['lock'], ['lock', 'true'], ['lock', 'x', 'true']):
            with helpers.captured() as (_, stderr):
                self.assertEqual(gpulock.main(argv), 2)
            self.assertEqual(stderr.getvalue(), 'usage: python -m picovox_train.gpulock LOCKFILE -- COMMAND [ARGUMENT ...]\n')

    def test_run(self):
        # The command runs holding the lock (its directory made) - its exit status
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sub', 'gpu.lock')
            with helpers.captured() as (_, stderr):
                self.assertEqual(gpulock.main([path, '--', sys.executable, '-c', 'raise SystemExit(3)']), 3)
            self.assertTrue(os.path.exists(path))
        self.assertEqual(stderr.getvalue(), '')

    def test_wait(self):
        # Another holder has the lock - the command waits for it, then runs
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'gpu.lock')
            with open(path, 'a', encoding='utf-8') as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                timer = threading.Timer(0.3, fcntl.flock, (holder, fcntl.LOCK_UN))
                timer.start()
                with helpers.captured() as (_, stderr):
                    self.assertEqual(gpulock.main([path, '--', 'true']), 0)
                timer.join()
        lines = stderr.getvalue().splitlines()
        self.assertEqual(lines[0], f'gpulock: waiting for {path} (true)')
        self.assertRegex(lines[1], f'^gpulock: acquired {path} after [0-9]+ s$')

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['gpulock']), helpers.captured() as (_, stderr), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.gpulock', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('usage:', stderr.getvalue())
