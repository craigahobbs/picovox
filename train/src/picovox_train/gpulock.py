# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Run a command holding the GPU lock - make train's GPU stages (teacher A and B, the distillation targets, the student,
its polish) one at a time, while its CPU stages run beside them (two GPU jobs at once each run at about half speed):

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.gpulock build/gpu.lock -- python -m picovox_train.trainmulti ...

The lock is an exclusive flock on the file; waiting for it is logged to stderr. The command then runs in this process
(exec), holding the lock until it exits - a signal to this process (make train's stage stopped) is the command's, so the
lock is never released while the command runs.
"""

import fcntl
import os
import sys
import time


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or argv[1] != '--':
        print('usage: python -m picovox_train.gpulock LOCKFILE -- COMMAND [ARGUMENT ...]', file=sys.stderr)
        return 2
    path, command = argv[0], argv[2:]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            start = time.time()
            print(f'gpulock: waiting for {path} ({" ".join(command)})', file=sys.stderr, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(f'gpulock: acquired {path} after {time.time() - start:.0f} s', file=sys.stderr, flush=True)
        os.set_inheritable(lock.fileno(), True)
        try:
            os.execvp(command[0], command)
        except OSError as error:
            print(f'gpulock: {error}', file=sys.stderr)
    return 127


if __name__ == '__main__':
    sys.exit(main())
