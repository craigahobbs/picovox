# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Run a command holding the GPU lock - make train's GPU stages (teacher A and B, the distillation targets, the student,
its polish) one at a time, while its CPU stages run beside them (two GPU jobs at once each run at about half speed):

    build/venv-nn/bin/python -B -m nn.gpulock build/train/gpu.lock -- python -m nn.trainmulti ...

The lock is an exclusive flock on the file (released when the command exits, or if this process dies); waiting for it
is logged to stderr.
"""

import fcntl
import os
import subprocess
import sys
import time


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or argv[1] != '--':
        print('usage: python -m nn.gpulock LOCKFILE -- COMMAND [ARGUMENT ...]', file=sys.stderr)
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
        return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
