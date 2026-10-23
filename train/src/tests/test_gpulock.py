# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import fcntl
import os
import runpy
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
import warnings

from picovox_train import gpulock

from . import helpers


# A command that exits 3 if it holds the lock (a lock of its own fails), else 4
HOLDS_LOCK = '''
import fcntl, sys
with open(sys.argv[1], 'a', encoding='utf-8') as other:
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(3)
sys.exit(4)
'''


class TestGpulock(unittest.TestCase):

    def test_usage(self):
        for argv in ([], ['lock'], ['lock', 'true'], ['lock', 'x', 'true']):
            with helpers.captured() as (_, stderr):
                self.assertEqual(gpulock.main(argv), 2)
            self.assertEqual(stderr.getvalue(), 'usage: python -m picovox_train.gpulock LOCKFILE -- COMMAND [ARGUMENT ...]\n')

    def test_run(self):
        # The command runs in this process (exec) holding the lock (its directory made) - its exit status; a command that
        # isn't found exits 127
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sub', 'gpu.lock')
            env = {**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(gpulock.__file__))}
            process = subprocess.run([sys.executable, '-m', 'picovox_train.gpulock', path, '--', sys.executable, '-c', HOLDS_LOCK, path],
                                     env=env, capture_output=True, text=True, check=False)
            self.assertEqual((process.returncode, process.stderr), (3, ''))
            process = subprocess.run([sys.executable, '-m', 'picovox_train.gpulock', path, '--', os.path.join(directory, 'missing')],
                                     env=env, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 127)
            self.assertIn('gpulock: [Errno 2]', process.stderr)

        # (in this process, the exec - the lock's descriptor inherited)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'gpu.lock')
            with unittest.mock.patch('os.execvp') as execvp, unittest.mock.patch('os.set_inheritable') as set_inheritable:
                self.assertEqual(gpulock.main([path, '--', 'train', '--fast']), 127)
        execvp.assert_called_once_with('train', ['train', '--fast'])
        self.assertTrue(set_inheritable.call_args.args[1])
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (_, stderr), \
                unittest.mock.patch('os.execvp', side_effect=FileNotFoundError(2, 'No such file or directory')):
            self.assertEqual(gpulock.main([os.path.join(directory, 'gpu.lock'), '--', 'missing']), 127)
        self.assertEqual(stderr.getvalue(), 'gpulock: [Errno 2] No such file or directory\n')

    def test_wait(self):
        # Another holder has the lock - the command waits for it, then runs
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'gpu.lock')
            with open(path, 'a', encoding='utf-8') as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                timer = threading.Timer(0.3, fcntl.flock, (holder, fcntl.LOCK_UN))
                timer.start()
                with helpers.captured() as (_, stderr), unittest.mock.patch('os.execvp') as execvp:
                    self.assertEqual(gpulock.main([path, '--', 'true']), 127)
                timer.join()
        execvp.assert_called_once_with('true', ['true'])
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
