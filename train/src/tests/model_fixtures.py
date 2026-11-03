# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The export, packaging, and panel tests' fixtures - a tiny checkpoint (trainmulti.py's format) and
its training data (the training speech, helpers.write_corpus), with the modules' directories patched
"""

import contextlib
import os
import unittest.mock

import torch

from picovox_train import export, panel, trainmulti
from picovox_train.model import MultiModel
from tests import helpers


# The tiny checkpoints' speaker table - the training speech's voices (rows 0 and 1)
SPEAKERS = ['male', 'female']


# Write the training data - DATA_DIR/tts
def write_data(data_dir):
    helpers.write_corpus(os.path.join(data_dir, 'tts'))


# A tiny model's checkpoint state (trainmulti.py's format, the critic aside)
def checkpoint_state(seed=1, args=None, **config):
    torch.manual_seed(seed)
    model = MultiModel(helpers.tiny_config(speakers=len(SPEAKERS), **config), priors=trainmulti.MULTI_PRIORS)
    return {'model': model.state_dict(), 'config': model.config, 'priors': trainmulti.MULTI_PRIORS, 'speakers': list(SPEAKERS),
            'step': 3, 'val': {'score': 1.5}, 'args': dict(args or {})}


# Write a tiny model's checkpoint - its path
def write_checkpoint(path, seed=1, args=None, **config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint_state(seed, args, **config), path)
    return path


# The modules' directories patched under a temporary root (data, runs, the panel's evaluations)
@contextlib.contextmanager
def patched_dirs(root):
    data_dir = os.path.join(root, 'data')
    runs_dir = os.path.join(root, 'runs')
    with contextlib.ExitStack() as stack:
        for module in (export, trainmulti):
            stack.enter_context(unittest.mock.patch.object(module, 'DATA_DIR', data_dir))
        stack.enter_context(unittest.mock.patch.object(trainmulti, 'RUNS_DIR', runs_dir))
        stack.enter_context(unittest.mock.patch.object(panel, 'EVAL_DIR', os.path.join(root, 'eval')))
        yield data_dir
