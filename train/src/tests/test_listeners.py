# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import contextlib
import importlib
import io
import json
import logging
import os
import runpy
import sys
import tempfile
import types
import unittest
import unittest.mock
import warnings

import numpy
import torch

from tests import data_fixtures, helpers
from tests.helpers import write_wav as write_pcm


# (listeners.py sets the model caches' environment variables when imported - kept out of the tests' environment)
with unittest.mock.patch.dict(os.environ):
    listeners = importlib.import_module('picovox_train.listeners')


# A fake UTMOS model - the batch's mean absolute sample, plus 3
class FakeUtmos(torch.nn.Module):

    def forward(self, batch, rate):
        assert rate == listeners.SAMPLE_RATE
        return batch.abs().mean() + 3


# A fake ECAPA encoder - a 3-dimension embedding of the batch's length
class FakeEcapa:

    @staticmethod
    def encode_batch(batch):
        return torch.tensor([[[float(batch.shape[1]), 1.0, 0.0]]])


# Write a WAV file (a buzz) - its path
def write_wav(directory, name, seconds, rate, channels=1):
    samples = data_fixtures.buzz(seconds, rate)
    path = os.path.join(directory, name)
    write_pcm(path, numpy.stack([samples] * channels, axis=1) if channels > 1 else samples, rate)
    return path


# The listener model libraries' fake modules - speechbrain, huggingface_hub (local_files_only fails when
# offline is False)
@contextlib.contextmanager
def fake_libraries(cached=True):
    def snapshot_download(repo, local_files_only=False):
        if not local_files_only or cached:
            return f'/snapshots/{repo}'
        raise OSError('not cached')
    hub = types.ModuleType('huggingface_hub')
    hub.snapshot_download = unittest.mock.Mock(side_effect=snapshot_download)
    speaker = types.ModuleType('speechbrain.inference.speaker')
    speaker.EncoderClassifier = types.SimpleNamespace(from_hparams=unittest.mock.Mock(return_value=FakeEcapa()))
    with data_fixtures.fake_modules(**{'huggingface_hub': hub, 'speechbrain': types.ModuleType('speechbrain'),
                                       'speechbrain.inference': types.ModuleType('speechbrain.inference'),
                                       'speechbrain.inference.speaker': speaker}):
        yield hub, speaker


class TestArguments(unittest.TestCase):

    def test_parse_args(self):
        with tempfile.TemporaryDirectory() as directory:
            list_path = os.path.join(directory, 'list.txt')
            with open(list_path, 'w', encoding='utf-8') as file:
                file.write('b.wav\n\n  c.wav \n')
            args = listeners.parse_args(['a.wav', '--files-from', list_path, '--no-mos', '--device', 'cpu'])
        self.assertEqual(args.files, ['a.wav', 'b.wav', 'c.wav'])
        self.assertEqual((args.no_mos, args.no_embedding, args.device, args.chunk_seconds), (True, False, 'cpu', 20))
        with unittest.mock.patch.object(sys, 'stdin', io.StringIO('d.wav\n')):
            self.assertEqual(listeners.parse_args(['--files-from', '-']).files, ['d.wav'])
        with helpers.captured() as (_, stderr), self.assertRaises(SystemExit):
            listeners.parse_args([])
        self.assertIn('no WAV files', stderr.getvalue())

    def test_quiet_libraries(self):
        with warnings.catch_warnings(), unittest.mock.patch.dict(os.environ):
            os.environ.pop('TQDM_DISABLE', None)
            try:
                listeners.quiet_libraries()
                warnings.warn('hidden')
                self.assertEqual(os.environ['TQDM_DISABLE'], '1')
                self.assertTrue(logging.getLogger('x').manager.disable >= logging.WARNING)
            finally:
                logging.disable(logging.NOTSET)

    def test_to_stderr(self):
        with helpers.captured() as (stdout, stderr):
            self.assertEqual(listeners.to_stderr(lambda value, add=0: print(value) or value + add, 2, add=1), 3)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ('', '2\n'))


class TestModels(unittest.TestCase):

    def test_hf_snapshot(self):
        with fake_libraries() as (hub, _):
            self.assertEqual(listeners.hf_snapshot('a/b'), '/snapshots/a/b')
        self.assertEqual(hub.snapshot_download.call_args_list, [unittest.mock.call('a/b', local_files_only=True)])
        with fake_libraries(cached=False) as (hub, _):
            self.assertEqual(listeners.hf_snapshot('a/b'), '/snapshots/a/b')
        self.assertEqual(hub.snapshot_download.call_args_list[1], unittest.mock.call('a/b'))

    def test_load_utmos(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(torch.hub, 'get_dir', return_value=directory), \
                unittest.mock.patch.object(torch.hub, 'load', return_value=FakeUtmos()) as load:
            model = listeners.load_utmos('cpu')
            self.assertFalse(model.training)
            self.assertEqual(load.call_args, unittest.mock.call(listeners.UTMOS_REPO, 'utmos22_strong', trust_repo=True, progress=False))
            repo_dir = os.path.join(directory, listeners.UTMOS_REPO_DIR)
            os.makedirs(repo_dir)
            with open(os.path.join(repo_dir, 'hubconf.py'), 'w', encoding='utf-8') as file:
                file.write('')
            listeners.load_utmos('cpu')
            self.assertEqual(load.call_args, unittest.mock.call(repo_dir, 'utmos22_strong', source='local', progress=False))

    def test_load_ecapa(self):
        with fake_libraries() as (_, speaker):
            self.assertIsInstance(listeners.load_ecapa('cpu'), FakeEcapa)
        self.assertEqual(speaker.EncoderClassifier.from_hparams.call_args, unittest.mock.call(
            source=f'/snapshots/{listeners.ECAPA_REPO}', savedir=os.path.join(listeners.CACHE_DIR, 'spkrec-ecapa-voxceleb'),
            run_opts={'device': 'cpu'}))


class TestScoring(unittest.TestCase):

    def test_read_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            samples, seconds = listeners.read_audio(write_wav(directory, 'a.wav', 0.5, 16000))
            self.assertEqual((len(samples), seconds), (8000, 0.5))
            samples, seconds = listeners.read_audio(write_wav(directory, 'b.wav', 0.5, 24000, channels=2))
        self.assertEqual((len(samples), seconds, samples.dtype), (8000, 0.5, numpy.float32))

    def test_chunks(self):
        samples = numpy.arange(100, dtype=numpy.float32)
        self.assertEqual([len(part) for part in listeners.chunks(samples, 1)], [100])
        with unittest.mock.patch.object(listeners, 'SAMPLE_RATE', 40):
            self.assertEqual([len(part) for part in listeners.chunks(samples, 1)], [34, 34, 32])

    def test_score_file(self):
        models = {'mos': FakeUtmos(), 'embedding': FakeEcapa()}
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (stdout, _):
            path = write_wav(directory, 'a.wav', 1.0, 16000)
            args = argparse.Namespace(chunk_seconds=0.4, device='cpu')
            result = listeners.score_file(path, models, args)
            alone = listeners.score_file(path, {}, args)
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(set(result), {'file', 'seconds', 'mos', 'embedding'})
        self.assertEqual((result['file'], result['seconds']), (path, 1.0))
        self.assertGreater(result['mos'], 3)
        lengths = [5334, 5334, 5332]
        self.assertEqual(result['embedding'], [round(sum(length * length for length in lengths) / 16000, 5), 1.0, 0.0])
        self.assertEqual(alone, {'file': path, 'seconds': 1.0})


# Standard output with a bytes buffer (main writes its JSON lines to sys.stdout.buffer), kept readable after main closes
# its wrapper
class FakeStdout(io.StringIO):

    def __init__(self):
        super().__init__()
        self.buffer = io.BytesIO()
        self.buffer.close = lambda: None


class TestMain(unittest.TestCase):

    def test_main(self):
        with tempfile.TemporaryDirectory() as directory, fake_libraries(), helpers.captured(), \
                unittest.mock.patch.object(torch, 'set_num_threads') as set_num_threads, \
                unittest.mock.patch.object(torch.backends.mps, 'is_available', return_value=False), \
                unittest.mock.patch.object(listeners, 'load_utmos', return_value=FakeUtmos()) as load_utmos, \
                unittest.mock.patch.object(listeners, 'quiet_libraries') as quiet:
            paths = [write_wav(directory, 'a.wav', 0.5, 16000), write_wav(directory, 'b.wav', 0.25, 22050)]
            stdout = FakeStdout()
            with unittest.mock.patch.object(sys, 'stdout', stdout):
                listeners.main(paths)
            lines = [json.loads(line) for line in stdout.buffer.getvalue().decode('utf-8').splitlines()]
            stdout = FakeStdout()
            argv = ['listeners.py', paths[0], '--no-mos', '--no-embedding', '--device', 'cpu']
            with unittest.mock.patch.object(sys, 'stdout', stdout), unittest.mock.patch.object(sys, 'argv', argv):
                listeners.main()
            bare = [json.loads(line) for line in stdout.buffer.getvalue().decode('utf-8').splitlines()]
        quiet.assert_called()
        set_num_threads.assert_called_with(max(1, min(8, os.cpu_count() or 1)))
        self.assertEqual(load_utmos.call_args, unittest.mock.call('cpu'))
        self.assertEqual([line['file'] for line in lines], paths)
        self.assertEqual(set(lines[0]), {'file', 'seconds', 'mos', 'embedding'})
        self.assertEqual(bare, [{'file': paths[0], 'seconds': 0.5}])

    def test_module_main(self):
        with unittest.mock.patch.object(sys, 'argv', ['listeners.py']), self.assertRaises(SystemExit), helpers.captured(), \
                unittest.mock.patch.dict(os.environ), warnings.catch_warnings():
            # (runpy warns that the module is already imported - it runs a fresh copy as __main__)
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.listeners', run_name='__main__', alter_sys=False)
