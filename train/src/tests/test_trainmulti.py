# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import functools
import json
import math
import os
import runpy
import tempfile
import threading
import time
import unittest
import unittest.mock
import warnings

import numpy
import torch

from picovox_train import dsynth, params, trainmulti
from picovox_train.model import MultiModel

from . import helpers


# The tests' training data - a temporary directory with the readers' and the emotional corpora (data/multi,
# data/cremad), a tiny recipe (train.json), and the runs' directory (runs/)
class TrainData:

    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.root = self.temp.name
        self.data_dir = os.path.join(self.root, 'data')
        self.runs_dir = os.path.join(self.root, 'runs')
        helpers.write_corpus(os.path.join(self.data_dir, 'multi'))
        helpers.write_corpus(os.path.join(self.data_dir, 'cremad'), speakers=helpers.CREMAD_SPEAKERS,
                             emotions=helpers.CREMAD_EMOTIONS)
        with open(trainmulti.RECIPE_PATH, encoding='utf-8') as file:
            recipe = json.load(file)
        recipe['teacher']['defaults'].update({'steps': 3, 'batch': 2, 'max-frames': 400, 'crop': 0.1, 'val-count': 4,
                                              'warmup': 1})
        recipe['teacher']['b'] = {'description': 'stage B', 'cremad': 0.5, 'adversarial': 0.1, 'adversarial-start': 1}
        self.recipe = os.path.join(self.root, 'train.json')
        with open(self.recipe, 'w', encoding='utf-8') as file:
            json.dump(recipe, file)

    # Patch the corpora's and runs' directories (a context manager)
    def patched(self):
        patch_data = unittest.mock.patch.object(trainmulti, 'DATA_DIR', self.data_dir)
        patch_runs = unittest.mock.patch.object(trainmulti, 'RUNS_DIR', self.runs_dir)

        class Both:
            def __enter__(self):
                patch_data.start()
                patch_runs.start()

            def __exit__(self, *args):
                patch_runs.stop()
                patch_data.stop()
        return Both()

    # Run trainmulti.main with the tiny recipe and model (output captured) - its exit status
    def train(self, name, *arguments):
        argv = ['--name', name, '--recipe-file', self.recipe, '--device', 'cpu', '--config', json.dumps(helpers.tiny_config()),
                *arguments]
        with self.patched(), helpers.captured() as (_, stderr):
            status = trainmulti.main(argv)
        self.last_log = stderr.getvalue()  # pylint: disable=attribute-defined-outside-init
        return status

    def run_path(self, name, *parts):
        return os.path.join(self.runs_dir, name, *parts)

    def cleanup(self):
        self.temp.cleanup()


DATA = None


# A corpus of no utterances
class EmptyCorpus:
    items = []

    def __len__(self):
        return 0


def setUpModule():  # pylint: disable=invalid-name
    global DATA  # pylint: disable=global-statement
    DATA = TrainData()


def tearDownModule():  # pylint: disable=invalid-name
    DATA.cleanup()


# A tiny trained stage-A checkpoint (trained once) - its path; for the other tests (stepcheck, synthesize)
@functools.cache
def tiny_checkpoint():
    DATA.train('shared', '--steps', '2')
    return DATA.run_path('shared', 'last.pt')


class TestHelpers(unittest.TestCase):

    def test_sonorant_mask(self):
        vocab = ['<pad>', '<unk>', '<bos>', '_', '<comma>', '<period>', '<question>', '<wh>', '<exclamation>', 'AA1', 'B',
                 'L', 'S', 'ER0']
        self.assertEqual(trainmulti.sonorant_mask(vocab).tolist(),
                         [False] * 9 + [True, False, True, False, True])

    def test_multi_priors(self):
        self.assertAlmostEqual(trainmulti.MULTI_PRIORS['f0'], math.sqrt(185 * 122))
        self.assertEqual(trainmulti.MULTI_PRIORS['av'], -30)

    def test_corpus(self):
        with DATA.patched():
            corpus = trainmulti.Corpus('cremad', 'train', speaker_offset=10)
            limited = trainmulti.Corpus('multi', 'train', max_items=1)
        self.assertEqual(len(corpus), 4)
        self.assertEqual(len(limited), 1)
        angry = corpus.items[0]
        self.assertEqual(angry['speaker'], 10)
        self.assertEqual(angry['emotion'], 0)
        self.assertEqual(angry['intensity'], 1.0)
        neutral = corpus.items[1]
        self.assertEqual(neutral['emotion'], -1)
        self.assertEqual(neutral['intensity'], 0.0)
        # (the angry utterances' pitch range - the actor's emotion range, 80-300 Hz)
        self.assertTrue(numpy.all(angry['f0'] >= 80) and numpy.all(angry['f0'] <= 300))
        self.assertEqual(len(angry['f0']), 2 * angry['frameCount'] + 1)
        self.assertEqual(len(angry['voiced']), 2 * angry['frameCount'])
        self.assertEqual(angry['formants'].shape, (angry['frameCount'], 3))
        # The audio, zero-padded past the end
        count = angry['sampleCount']
        audio = corpus.audio(angry, count - 10, count + 10)
        self.assertEqual(len(audio), 20)
        self.assertTrue(numpy.all(audio[10:] == 0))
        self.assertTrue(numpy.any(audio[:10] != 0))

    def test_make_batches(self):
        with DATA.patched():
            corpus = trainmulti.Corpus('multi', 'train')
        rng = numpy.random.default_rng(1)
        batches = trainmulti.make_batches(corpus, 2, 10000, rng)
        self.assertEqual(sorted(len(batch) for batch in batches), [2, 2])
        self.assertEqual(sorted(index for batch in batches for index in batch), [0, 1, 2, 3])
        # (the frame limit - one utterance a batch)
        self.assertEqual(len(trainmulti.make_batches(corpus, 4, 60, rng)), 4)
        self.assertEqual(trainmulti.make_batches(EmptyCorpus(), 2, 100, rng), [])
        stream = trainmulti.batch_stream(corpus, 4, 10000, rng)
        self.assertEqual(sorted(next(stream)), [0, 1, 2, 3])
        self.assertEqual(sorted(next(stream)), [0, 1, 2, 3])

    def test_bucket_length(self):
        self.assertEqual(trainmulti.bucket_length(37, 0), 37)
        self.assertEqual(trainmulti.bucket_length(5, 1.125), 8)
        self.assertEqual(trainmulti.bucket_length(37, 1.125), 40)
        # (8, 16, 24, 40, 64, 96, 144)
        self.assertEqual(trainmulti.bucket_length(100, 1.5), 144)

    def test_collate(self):
        with DATA.patched():
            corpus = trainmulti.Corpus('multi', 'train')
        batch = trainmulti.collate(corpus, [0, 1], 'cpu')
        items = batch['items']
        frames = max(item['frameCount'] for item in items)
        self.assertEqual(batch['frameCount'], frames)
        self.assertEqual(tuple(batch['tokens'].shape), (2, max(len(item['tokens']) for item in items)))
        self.assertEqual(tuple(batch['f0'].shape), (2, 2 * frames + 1))
        self.assertEqual(batch['renderCounts'].tolist(), [2 * item['frameCount'] for item in items])
        bucketed = trainmulti.collate_host(corpus, [0, 1], bucket_ratio=1.125)
        self.assertEqual(bucketed['frameCount'], trainmulti.bucket_length(frames, 1.125))
        self.assertEqual(bucketed['host']['tokens'].shape[1] % 8, 0)
        self.assertEqual(bucketed['host']['f0'].shape[1], 2 * bucketed['frameCount'] + 1)

    def test_crops(self):
        with DATA.patched():
            corpus = trainmulti.Corpus('multi', 'train')
        batch = trainmulti.collate(corpus, [0, 1], 'cpu')
        rng = numpy.random.default_rng(3)
        starts, length = trainmulti.crop_starts(batch['items'], 41, 20, rng)
        self.assertEqual(length, 20)
        self.assertTrue(all(start % 2 == 0 and start + length <= 40 for start in starts))
        self.assertEqual(trainmulti.crop_starts(batch['items'][:1], 11, 20, rng), ([0], 10))
        audio = trainmulti.crop_audio(corpus, batch['items'], [0, 2], 20, 24000)
        self.assertEqual(audio.shape, (2, 2400))
        resampled = trainmulti.crop_audio(corpus, batch['items'], [0, 2], 20, 16000)
        self.assertEqual(resampled.shape, (2, 1600))
        values = torch.zeros(2, 2 * batch['frameCount'] + 1, params.COUNT)
        crop_values, target = trainmulti.crop(values, batch, 20, 24000, rng)
        self.assertEqual(tuple(crop_values.shape), (2, 21, params.COUNT))
        self.assertEqual(tuple(target.shape), (2, 2400))

    def test_band_limit(self):
        audio = torch.randn(2, 480, dtype=torch.float64)
        self.assertIs(trainmulti.band_limit(audio, [12000, 13000], 24000), audio)
        limited = trainmulti.band_limit(audio, [6000, 12000], 24000)
        spectrum = torch.fft.rfft(limited).abs()
        self.assertTrue(torch.all(spectrum[0, 121:] < 1e-9))
        self.assertTrue(torch.allclose(limited[1], audio[1]))
        same = trainmulti.band_limit(audio, [6000, 12000], 24000, cutoff_tensor=torch.tensor([6000.0, 12000.0]))
        self.assertTrue(torch.equal(same, limited))

    def test_quality_stats(self):
        values = torch.tensor([params.DEFAULTS[name] for name in params.NAMES]).repeat(1, 4, 1)
        values[0, :, params.INDEX['av']] = torch.tensor([0.0, -5.0, -40.0, 0.0])
        values[0, :, params.INDEX['b1']] = torch.tensor([100.0, 200.0, 900.0, 300.0])
        values[0, :, params.INDEX['ah']] = -20.0
        stats = trainmulti.quality_stats(values, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
        self.assertEqual(float(stats['b1']), 100.0)
        self.assertEqual(float(stats['ahRel']), -20.0)
        empty = trainmulti.quality_stats(values, torch.zeros(1, 4))
        self.assertTrue(math.isnan(float(empty['b1'])))

    def test_frame_tokens(self):
        index, position = trainmulti.frame_tokens(torch.tensor([[2, 0, 3]]), 6)
        self.assertEqual(index.tolist(), [[0, 0, 2, 2, 2, 2]])
        self.assertTrue(numpy.allclose(position.numpy(), [[0.25, 0.75, 0.5 / 3, 1.5 / 3, 2.5 / 3, 3.5 / 3]]))

    def test_recipe_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'recipe.json')
            with open(path, 'w', encoding='utf-8') as file:
                json.dump({'teacher': {'a': {'description': 'x', 'max-frames': 10, 'lr': 0.5}}}, file)
            self.assertEqual(trainmulti.recipe_defaults('teacher.a', path), {'max_frames': 10, 'lr': 0.5})

    def test_default_settings(self):
        settings = trainmulti.default_settings(crop=0.5, batch=3)
        self.assertEqual(settings.name, 'defaults')
        self.assertEqual(settings.batch, 3)
        self.assertEqual(settings.crop_frames, 100)
        self.assertEqual(settings.steps, 30000)

    def test_batch_losses(self):
        with DATA.patched():
            corpus = trainmulti.Corpus('multi', 'train')
        settings = trainmulti.default_settings(crop=0.1)
        torch.manual_seed(1)
        model = MultiModel(helpers.tiny_config(), priors=trainmulti.MULTI_PRIORS)
        synth = dsynth.DSynth(24000)
        sonorants = torch.tensor(trainmulti.sonorant_mask(corpus.data.index['vocab']))
        batch = trainmulti.collate(corpus, [0, 1], 'cpu')
        rng = numpy.random.default_rng(1)
        losses, audio, target, stats = trainmulti.batch_losses(model, synth, batch, settings, rng, 5, 1.0, sonorants)
        self.assertEqual(audio.shape, target.shape)
        self.assertEqual(set(stats), {'b1', 'b2', 'b3', 'ahRel', 'afRel'})
        expected = (losses['spectral'] + losses['durPhone'] + losses['durPause'] + losses['durPaused'] + 2 * losses['f0'] +
                    losses['formant'] + losses['order'] + 0.05 * losses['smooth'] + losses['band'] + losses['sonorantAf'] +
                    losses['eqSmooth'])
        self.assertAlmostEqual(float(losses['total'].detach()), float(expected.detach()), places=4)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        # (the same again - dropout's draws reseeded)
        torch.manual_seed(2)
        losses = trainmulti.batch_losses(model, synth, batch, settings, numpy.random.default_rng(1), 5, 1.0, sonorants)[0]
        torch.manual_seed(2)
        result = trainmulti.batch_losses(model, synth, batch, settings, numpy.random.default_rng(1), 5, 1.0, sonorants,
                                         with_values=True)
        self.assertEqual(float(result[0]['total'].detach()), float(losses['total'].detach()))
        self.assertEqual(result[4].shape[-1], params.COUNT)


class TestPrefetcher(unittest.TestCase):

    def setUp(self):
        with DATA.patched():
            self.corpora = {'multi': trainmulti.Corpus('multi', 'train'), 'cremad': trainmulti.Corpus('cremad', 'train')}
        self.rng = numpy.random.default_rng(2)
        self.streams = {name: trainmulti.batch_stream(corpus, 2, 10000, self.rng) for name, corpus in self.corpora.items()}
        self.synth = dsynth.DSynth(24000)

    def test_steps(self):
        args = argparse.Namespace(cremad=0.5, crop_frames=20, rate=24000, bucket_ratio=1.125)
        prefetcher = trainmulti.Prefetcher(self.corpora, self.streams, args, self.rng, self.synth, 0.005, depth=1)
        # (the queue fills - the thread waits for room)
        time.sleep(0.3)
        names = set()
        for _ in range(4):
            prepared = prefetcher.next()
            names.add(prepared['corpusName'])
            self.assertEqual(prepared['length'], 20)
            self.assertEqual(tuple(prepared['batch']['host']['target'].shape), (2, 2400))
            self.assertEqual(prepared['cutoffs'], [7600.0, 7600.0])
            self.assertEqual(prepared['plan']['count'], 2400)
            self.assertIn('host', prepared['plan'])
        prefetcher.close()
        self.assertFalse(prefetcher.thread.is_alive())
        self.assertEqual(names, {'multi', 'cremad'})

    def test_no_plan(self):
        # (no samples to render - a zero-length crop)
        args = argparse.Namespace(cremad=0.0, crop_frames=0, rate=24000)
        prefetcher = trainmulti.Prefetcher({'multi': self.corpora['multi']}, self.streams, args, self.rng, self.synth, 0.005)
        prepared = prefetcher.next()
        prefetcher.close()
        self.assertEqual(prepared['corpusName'], 'multi')
        self.assertIsNone(prepared['plan'])

    def test_error(self):
        args = argparse.Namespace(cremad=0.0, crop_frames=20, rate=24000)
        prefetcher = trainmulti.Prefetcher({'multi': self.corpora['multi']}, {'multi': iter(())}, args, self.rng, self.synth,
                                           0.005)
        with self.assertRaises(StopIteration):
            prefetcher.next()
        prefetcher.close()

    def test_close_while_preparing(self):
        # (close with the queue empty while a step is being made - the thread finishes it, then stops)
        started = threading.Event()

        def slow_prepare():
            started.set()
            time.sleep(0.3)
            return {}
        with unittest.mock.patch.object(trainmulti.Prefetcher, '_prepare', side_effect=slow_prepare):
            prefetcher = trainmulti.Prefetcher({}, {}, None, None, None, 0.005)
            started.wait()
            prefetcher.close()
        self.assertFalse(prefetcher.thread.is_alive())


class TestValidate(unittest.TestCase):

    def test_validate(self):
        with DATA.patched():
            corpus = trainmulti.Corpus('multi', 'val')
        settings = trainmulti.default_settings(crop=0.1, batch=1)
        model = MultiModel(helpers.tiny_config())
        sonorants = torch.tensor(trainmulti.sonorant_mask(corpus.data.index['vocab']))
        result = trainmulti.validate(model, dsynth.DSynth(24000), corpus, settings, 'cpu', sonorants)
        self.assertTrue(model.training)
        self.assertEqual(result['score'], result['spectral'] + result['durPhone'] + result['f0'])
        self.assertGreater(result['f0Cents'], 0)
        self.assertIn('b1', result)
        # Deterministic (its own random state)
        again = trainmulti.validate(model, dsynth.DSynth(24000), corpus, settings, 'cpu', sonorants)
        self.assertEqual(result, again)


class TestMain(unittest.TestCase):

    def test_stage_a(self):
        self.assertEqual(DATA.train('a', '--log-every', '2', '--val-every', '2', '--keep-every', '2', '--speaker-noise', '0',
                                    '--save-every', '0'), 0)
        files = sorted(os.listdir(DATA.run_path('a')))
        self.assertEqual(files, ['best.pt', 'last.pt', 'log.jsonl', 'step-2.pt'])
        state = torch.load(DATA.run_path('a', 'last.pt'), weights_only=False)
        self.assertEqual(state['step'], 3)
        self.assertEqual(state['speakers'], ['arctic-slt', 'arctic-bdl', 'cremad-1001', 'cremad-1002'])
        self.assertEqual(state['readerCount'], 2)
        self.assertEqual(state['args']['steps'], 3)
        self.assertNotIn('critic', state)
        self.assertEqual(state['val']['score'], state['best'])
        with open(DATA.run_path('a', 'log.jsonl'), encoding='utf-8') as file:
            rows = [json.loads(line) for line in file]
        self.assertEqual([next(iter(row)) for row in rows], ['train', 'val', 'val'])
        self.assertEqual(rows[0]['train']['step'], 2)
        self.assertEqual(rows[2]['val']['step'], 3)
        self.assertIn('data: 4 reader utterances, 0 emotional, val multi 2', DATA.last_log)

    def test_recipe_stage_b(self):
        # Stage B (the recipe's section): the emotional speech and the critic, from stage A's weights
        init = tiny_checkpoint()
        self.assertEqual(DATA.train('b', '--recipe', 'teacher.b', '--init', init, '--log-every', '1'), 0)
        self.assertIn(f'init from {init}: missing [], unexpected []', DATA.last_log)
        state = torch.load(DATA.run_path('b', 'last.pt'), weights_only=False)
        self.assertIn('critic', state)
        self.assertEqual(state['args']['cremad'], 0.5)
        self.assertIn('cremad.score', state['val'])
        self.assertIn('critic', DATA.last_log)
        self.assertIn('cremad.durPhone', DATA.last_log)

    def test_critic(self):
        # The critic alone, then with the generator's adversarial and feature-matching losses; the critic's initial
        # weights from another run's; float32 convolutions
        self.assertEqual(DATA.train('critic', '--adversarial', '0.1', '--adversarial-start', '1', '--log-every', '1'), 0)
        self.assertIn('adversarial', DATA.last_log)
        self.assertIn('matching', DATA.last_log)
        self.assertEqual(DATA.train('critic-init', '--adversarial', '0.1', '--critic-precision', 'float32', '--init-critic',
                                    DATA.run_path('critic', 'last.pt'), '--steps', '1'), 0)
        self.assertIn('critic init from', DATA.last_log)

    def test_resume(self):
        # Stopped at step 2, then resumed to the end - the critic's state restored too
        self.assertEqual(DATA.train('resume', '--adversarial', '0.1', '--stop', '2', '--resume'), 0)
        self.assertEqual(torch.load(DATA.run_path('resume', 'last.pt'), weights_only=False)['step'], 2)
        self.assertEqual(DATA.train('resume', '--adversarial', '0.1', '--resume'), 0)
        self.assertIn('resume: 3 steps from 2', DATA.last_log)
        self.assertEqual(torch.load(DATA.run_path('resume', 'last.pt'), weights_only=False)['step'], 3)

    def test_resume_without_critic_state(self):
        # A stage A checkpoint resumed with the critic - the critic starts fresh
        self.assertEqual(DATA.train('resume-critic', '--stop', '1'), 0)
        self.assertEqual(DATA.train('resume-critic', '--adversarial', '0.1', '--resume', '--steps', '2'), 0)
        self.assertIn('critic', torch.load(DATA.run_path('resume-critic', 'last.pt'), weights_only=False))

    def test_validation_schedule(self):
        # Validated only from --val-from (and at the end); the best is kept only when the score improves
        scores = iter([2.0, 3.0])

        def validate(*_args):
            return {'score': next(scores)}
        with unittest.mock.patch.object(trainmulti, 'validate', side_effect=validate):
            self.assertEqual(DATA.train('schedule', '--steps', '4', '--val-every', '1', '--val-from', '3'), 0)
        state = torch.load(DATA.run_path('schedule', 'best.pt'), weights_only=False)
        self.assertEqual((state['step'], state['val']), (3, {'score': 2.0}))
        last = torch.load(DATA.run_path('schedule', 'last.pt'), weights_only=False)
        self.assertEqual((last['step'], last['val'], last['best']), (4, {'score': 3.0}, 2.0))

    def test_nonfinite_gradient(self):
        # A step with a non-finite gradient norm is skipped (the unfused optimizer)
        with unittest.mock.patch.object(torch.nn.utils, 'clip_grad_norm_', return_value=torch.tensor(math.inf)), \
                warnings.catch_warnings():
            # (the schedule steps though the optimizer didn't - torch warns)
            warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step', category=UserWarning)
            self.assertEqual(DATA.train('nonfinite', '--steps', '1'), 0)
        initial = MultiModel({**helpers.tiny_config(), 'speakers': 4})
        state = torch.load(DATA.run_path('nonfinite', 'last.pt'), weights_only=False)
        self.assertEqual(state['optimizer']['state'], {})
        self.assertEqual(state['step'], 1)
        self.assertEqual(set(state['model']), set(initial.state_dict()))

    def test_nan_statistics(self):
        # The statistics are left out of the log when NaN (no strongly voiced frames)
        def nan_stats(values, _frame_mask):
            return {'b1': torch.tensor(math.nan, device=values.device)}
        with unittest.mock.patch.object(trainmulti, 'quality_stats', side_effect=nan_stats):
            self.assertEqual(DATA.train('nan', '--steps', '1', '--log-every', '1'), 0)
        with open(DATA.run_path('nan', 'log.jsonl'), encoding='utf-8') as file:
            train_row = json.loads(file.readline())['train']
        self.assertNotIn('b1', train_row)
        self.assertIn('spectral', train_row)

    def test_module_main(self):
        # python -m picovox_train.trainmulti (the usage error - no --name)
        with unittest.mock.patch('sys.argv', ['trainmulti']), helpers.captured() as (_, stderr), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=".*found in sys.modules", category=RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.trainmulti', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('the following arguments are required: --name', stderr.getvalue())


class TestStepOptions(unittest.TestCase):

    def test_options(self):
        # Stage B (the critic, the emotional speech's validation) with every training step option, stopped and resumed
        init = tiny_checkpoint()
        options = ['--recipe', 'teacher.b', '--init', init, '--lean-clip', '--schedule', 'wsd', '--cooldown', '0.5', '--val-every', '1']
        self.assertEqual(DATA.train('options', *options, '--stop', '2', '--resume'), 0)
        self.assertEqual(DATA.train('options', *options, '--resume'), 0)
        self.assertIn('options: 3 steps from 2', DATA.last_log)
        last = torch.load(DATA.run_path('options', 'last.pt'), weights_only=False)
        self.assertEqual(last['step'], 3)
        self.assertIn('cremad.score', last['val'])
        # (the lean optimizer's state - AdamW's shared step count)
        self.assertTrue(all(state['step'].item() == 3 for state in last['optimizer']['state'].values()))

    def test_lean_clip(self):
        # The lean step alone (AdamW's groups only); a non-finite step skipped on the device
        def infinite_norm(_optimizer):
            return torch.tensor(math.inf)
        with unittest.mock.patch.object(trainmulti.optim.LeanOptimizer, 'grad_norm', infinite_norm), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step', category=UserWarning)
            self.assertEqual(DATA.train('lean', '--lean-clip', '--steps', '1'), 0)
        state = torch.load(DATA.run_path('lean', 'last.pt'), weights_only=False)
        self.assertTrue(all(value['step'].item() == 0 for value in state['optimizer']['state'].values()))
