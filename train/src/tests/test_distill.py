# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import hashlib
import io
import json
import os
import runpy
import tempfile
import time
import unittest
import unittest.mock
import warnings

import numpy
import torch

from picovox_train import ROOT, distill, export, package, params
from picovox_train.books import PinError
from tests import helpers, model_fixtures


# A short public-domain text, as a Gutenberg book (its license header and footer)
BOOK = '''The Project Gutenberg eBook of a test

*** START OF THE PROJECT GUTENBERG EBOOK A TEST ***

Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in
my purse, I thought I would sail about a little and see the watery part of the world.

It is a way I have of driving off the spleen. Whenever I find myself growing grim about the mouth, I account it high
time to get to sea as soon as I can. _There_ is nothing surprising in this.

1 2 3

Is it cold? Wait here! The cat sat on the mat.

*** END OF THE PROJECT GUTENBERG EBOOK A TEST ***

The license.
'''


# A tiny recipe (data/nn/train.json with tiny distillation settings) - its path
def write_config(root, teacher):
    with open(os.path.join(ROOT, 'data', 'nn', 'train.json'), encoding='utf-8') as file:
        recipe = json.load(file)
    books = os.path.join(root, 'books')
    os.makedirs(books, exist_ok=True)
    with open(os.path.join(books, 'book.txt'), 'w', encoding='utf-8') as file:
        file.write(BOOK)
    with open(os.path.join(books, 'plain.txt'), 'w', encoding='utf-8') as file:
        file.write('We went home. A dark night fell on the town. Oh. Call me Ishmael.\n')
    recipe['distill'] = {
        'teacher': teacher,
        'targets': {'books': books, 'bookCopies': 2, 'actorShare': 0.3, 'blendShare': 0.5, 'emotionShare': 0.5,
                    'intensity': [0.25, 1.0], 'minTokens': 4, 'maxTokens': 256, 'valItems': 3, 'seed': 1},
        'train': {'steps': 3, 'batch': 4, 'maxFrames': 4000, 'lr': 0.001, 'warmup': 1, 'durationWeight': 0.25,
                  'dropout': 0.05, 'weightDecay': 0.0001, 'valEvery': 2},
        'student': 'tiny',
        'students': [{'name': 'tiny', 'config': {'dim': 8, 'decoderDim': 8, 'encoderLayers': 1, 'decoderLayers': 1}},
                     {'name': 'small', 'config': {'dim': 12, 'decoderDim': 12, 'encoderLayers': 1, 'decoderLayers': 1}}],
        'polish': {'steps': 2, 'batch': 2, 'max-frames': 2000, 'crop': 0.2, 'lr': 0.0002, 'warmup': 1, 'cremad': 0.5,
                   'adversarial': 0.0, 'val-every': 2, 'val-count': 2, 'log-every': 1, 'keep-every': 0},
        'eval': {'speakers': [['arctic-slt', 'female'], ['arctic-bdl', 'male']], 'rate': 16000},
        'tolerance': {'mos': 0.05, 'wer': 1.0, 'similarity': 0.85},
        'samples': {'speakers': ['arctic-bdl'], 'texts': ['melville-moby'], 'rate': 16000}
    }
    path = os.path.join(root, 'train.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(recipe, file)
    return path


# A distill test's temporary root, directories, data, teacher, and recipe
class DistillCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        cls.data_dir = cls.patches.__enter__()
        model_fixtures.write_data(cls.data_dir)
        cls.teacher = model_fixtures.write_checkpoint(os.path.join(cls.root, 'teacher.pt'), seed=2)
        cls.config = write_config(cls.root, cls.teacher)

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    # Run a distill subcommand quietly (one CPU core for the book workers) - its exit status
    def run_main(self, argv):
        with helpers.captured(), unittest.mock.patch('os.cpu_count', return_value=1):
            return distill.main(argv)


class TestDistillSettings(DistillCase):

    def test_load_config(self):
        self.assertEqual(distill.load_config(self.config)['student'], 'tiny')
        self.assertEqual(distill.load_config()['student'], 's80')

    def test_root_path(self):
        self.assertEqual(distill.root_path('/a/b'), '/a/b')
        self.assertEqual(distill.root_path('data/nn'), os.path.join(ROOT, 'data/nn'))

    def test_log_line(self):
        with helpers.captured() as (_, stderr):
            distill.log_line('test', 'hello')
        self.assertRegex(stderr.getvalue(), r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d hello\n$')
        with open(os.path.join(distill.DISTILL_DIR, 'test.log'), encoding='utf-8') as file:
            self.assertTrue(file.read().endswith(' hello\n'))

    def test_default_device(self):
        with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(distill.default_device(), 'cpu')
        with unittest.mock.patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(distill.default_device(), 'mps')


class TestBooks(DistillCase):

    def test_book_paths(self):
        settings = distill.load_config(self.config)['targets']
        self.assertEqual([os.path.basename(path) for path in distill.book_paths(settings)], ['book.txt', 'plain.txt'])
        # (texts - pinned texts, fetched when missing and verified - books.py)
        books_dir = os.path.join(self.root, 'gutenberg')
        os.makedirs(books_dir)
        with open(os.path.join(books_dir, 'pg2.txt'), 'w', encoding='utf-8') as file:
            file.write('present')
        pins = {name: {'id': book_id, 'url': f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
                       'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'pinned': '2026-10-15', 'archive': None}
                for name, book_id, data in (('pg11-2026-10-16', 11, b'downloaded'), ('pg2', 2, b'present'))}
        pins_path = os.path.join(self.root, 'books.json')
        with open(pins_path, 'w', encoding='utf-8') as file:
            json.dump({'description': 'pins', 'texts': pins}, file)
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'downloaded'
        with unittest.mock.patch('urllib.request.urlopen', return_value=response) as urlopen:
            paths = distill.book_paths({'books': books_dir, 'pins': pins_path, 'texts': ['pg2', 'pg11-2026-10-16']})
        self.assertEqual([os.path.basename(path) for path in paths], ['pg11-2026-10-16.txt', 'pg2.txt'])
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(urlopen.call_args[0][0].full_url, 'https://www.gutenberg.org/cache/epub/11/pg11.txt')
        with open(os.path.join(books_dir, 'pg11-2026-10-16.txt'), encoding='utf-8') as file:
            self.assertEqual(file.read(), 'downloaded')

        # A text that isn't the pinned one is never used
        with open(os.path.join(books_dir, 'pg2.txt'), 'w', encoding='utf-8') as file:
            file.write('revised')
        with self.assertRaises(PinError):
            distill.book_paths({'books': books_dir, 'pins': pins_path, 'texts': ['pg2']})

    def test_book_paragraphs(self):
        paragraphs = distill.book_paragraphs(os.path.join(self.root, 'books', 'book.txt'))
        self.assertEqual(len(paragraphs), 3)
        self.assertTrue(paragraphs[0].startswith('Call me Ishmael. Some years ago, never mind how long precisely, having little or '
                                                 'no money in my'))
        self.assertIn('There is nothing', paragraphs[1])
        self.assertEqual(paragraphs[2], 'Is it cold? Wait here! The cat sat on the mat.')
        self.assertEqual(distill.book_paragraphs(os.path.join(self.root, 'books', 'plain.txt')),
                         ['We went home. A dark night fell on the town. Oh. Call me Ishmael.'])

    def test_book_sentences(self):
        path, sentences = distill.book_sentences(os.path.join(self.root, 'books', 'plain.txt'))
        self.assertTrue(path.endswith('plain.txt'))
        self.assertEqual(len(sentences), 4)
        self.assertTrue(all(sentence[0] == 2 for sentence in sentences))


class TestTargets(DistillCase):

    def test_blend_vectors(self):
        table = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        vectors = distill.blend_vectors(table, torch.tensor([0, 1]), torch.tensor([1, 1]), torch.tensor([0.25, 0.0]))
        self.assertTrue(torch.allclose(vectors, torch.tensor([[0.75, 0.25], [0.0, 1.0]])))

    def test_teacher_outputs(self):
        model, _ = export.load_model(self.teacher)
        tokens = torch.tensor([[2, 20, 30, 3, 40, 5]])
        with torch.no_grad():
            duration_outputs, durations, unit = distill.teacher_outputs(model, tokens, model.speaker.weight[:1], torch.tensor([-1]),
                                                                        torch.tensor([0.0]))
            inferred = model.infer(tokens, speaker=torch.tensor([0]))
        self.assertEqual(duration_outputs.shape, (1, 6, 3))
        self.assertTrue(torch.equal(durations, inferred['durations']))
        self.assertEqual(unit.shape, (1, 2 * int(durations.sum()), params.COUNT))

    def test_target_inputs(self):
        state = torch.load(self.teacher, weights_only=False)
        settings = distill.load_config(self.config)['targets']
        lines = []
        with unittest.mock.patch('os.cpu_count', return_value=None):
            token_lists, records, report = distill.target_inputs(state, settings, {9: 10}, lines.append)
        self.assertEqual(len(token_lists), len(records))
        sources = [record[9] for record in records]
        self.assertEqual(sources.count(0), 4)
        self.assertEqual(sources.count(1), 4)
        self.assertEqual(report['book.txt']['sentences'], 8)
        # (a sentence too short, and one already seen, left out)
        self.assertEqual(report['plain.txt'], {'sentences': 4, 'kept': 2})
        books = [record for record in records if record[9] == 2]
        # (each kept sentence twice, but the validation ones once)
        kept = report['book.txt']['kept'] + 2
        self.assertEqual(len(books), 2 * kept - 3)
        self.assertEqual(sum(1 for record in books if record[10] == 1), 3)
        self.assertTrue(any(record[6] > 0 for record in books))
        self.assertTrue(any(record[7] >= 0 for record in books))
        self.assertTrue(any(record[4] >= 2 for record in books))
        self.assertTrue(all(9 not in tokens for tokens in token_lists))
        # (the actors' records have their emotion and intensity)
        actors = [record for record in records if record[9] == 1]
        self.assertEqual(sorted(record[7] for record in actors), [-1, -1, 0, 0])
        self.assertEqual(sorted(record[8] for record in actors), [0.0, 0.0, 1.0, 1.0])
        self.assertIn('training utterances: 8 (4 readers, 4 actors)', lines)

    def test_make_targets(self):
        out = os.path.join(self.root, 'targets-all')
        self.assertEqual(self.run_main(['targets', '--config', self.config, '--device', 'cpu', '--out', out, '--batch', '3',
                                        '--batch-tokens', '60']), 0)
        targets = distill.Targets(out)
        index = targets.index
        self.assertEqual(index['teacher'], self.teacher)
        self.assertEqual(index['count'], len(targets.items))
        self.assertEqual(index['sources']['readers'], 4)
        self.assertEqual(index['validation'], 3)
        self.assertEqual(len(targets.split(1)), 3)
        self.assertEqual(index['tokens'], int(targets.items['tokenCount'].sum()))
        self.assertEqual(index['frames'], 2 * int(targets.items['frameCount'].sum()))
        self.assertEqual(targets.arrays['frames'].shape, (index['frames'], params.COUNT))
        # Each item's frames and durations are the teacher's inference
        model, _ = export.load_model(self.teacher)
        item = targets.items[0]
        token_slice = slice(item['tokenOffset'], item['tokenOffset'] + item['tokenCount'])
        tokens = torch.tensor(targets.arrays['tokens'][token_slice][None].astype(numpy.int64))
        speaker = distill.blend_vectors(model.speaker.weight.detach(), torch.tensor([int(item['speakerA'])]),
                                        torch.tensor([int(item['speakerB'])]), torch.tensor([float(item['blend'])]))
        with torch.no_grad():
            _, durations, unit = distill.teacher_outputs(model, tokens, speaker, torch.tensor([int(item['emotion'])]),
                                                         torch.tensor([float(item['intensity'])]))
        self.assertEqual(targets.arrays['durations'][token_slice].tolist(), durations[0].tolist())
        frames = targets.arrays['frames'][item['frameOffset']:item['frameOffset'] + 2 * item['frameCount']]
        self.assertTrue(numpy.allclose(frames, unit[0].numpy(), atol=1e-3))

        # A limited set - a quarter of it validation; and the book copies
        out = os.path.join(self.root, 'targets-limit')
        self.assertEqual(self.run_main(['targets', '--config', self.config, '--device', 'cpu', '--out', out, '--limit', '8',
                                        '--book-copies', '1', '--teacher', self.teacher]), 0)
        limited = distill.Targets(out)
        self.assertEqual(len(limited.items), 8)
        self.assertEqual(len(limited.split(1)), 2)
        self.assertEqual(limited.index['settings']['bookCopies'], 1)

    def test_make_targets_empty(self):
        out = os.path.join(self.root, 'targets-empty')
        with unittest.mock.patch.object(distill, 'target_inputs', return_value=([], [], {})):
            self.assertEqual(self.run_main(['targets', '--config', self.config, '--device', 'cpu', '--out', out]), 0)
        with open(os.path.join(out, 'index.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file)['count'], 0)

    def test_targets_default_out(self):
        with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(self.run_main(['targets', '--config', self.config, '--limit', '4']), 0)
        self.assertTrue(os.path.exists(os.path.join(distill.DISTILL_DIR, 'targets', 'index.json')))


class TestTraining(DistillCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.targets = os.path.join(cls.root, 'distill', 'targets')
        with helpers.captured(), unittest.mock.patch('os.cpu_count', return_value=1):
            distill.main(['targets', '--config', cls.config, '--device', 'cpu', '--out', cls.targets])

    def test_round_up(self):
        self.assertEqual(distill.round_up(1, 8), 8)
        self.assertEqual(distill.round_up(8, 8), 8)
        self.assertEqual(distill.round_up(9, 8), 16)

    def test_collate_targets(self):
        targets = distill.Targets(self.targets)
        indices = numpy.array([0, 1])
        batch = distill.collate_targets(targets, indices)
        items = targets.items[indices]
        self.assertEqual(batch['tokens'].shape[1] % 8, 0)
        self.assertEqual(batch['frameCount'] % 16, 0)
        self.assertEqual(batch['unit'].shape, (2, 2 * batch['frameCount'], params.COUNT))
        count = int(items['tokenCount'][1])
        self.assertEqual(batch['tokens'][1, :count].tolist(),
                         targets.arrays['tokens'][items['tokenOffset'][1]:items['tokenOffset'][1] + count].tolist())
        self.assertTrue(torch.all(batch['tokens'][1, count:] == 0))
        self.assertEqual(batch['frameCounts'].tolist(), items['frameCount'].tolist())

    def test_target_batches(self):
        targets = distill.Targets(self.targets)
        indices = targets.split(0)
        batches = distill.target_batches(targets, indices, 3, 10 ** 6, numpy.random.default_rng(0))
        self.assertEqual(sorted(index for batch in batches for index in batch), sorted(indices))
        self.assertTrue(all(len(batch) <= 3 for batch in batches))
        self.assertEqual(distill.target_batches(targets, indices[:0], 3, 100, numpy.random.default_rng(0)), [])
        frames = distill.target_batches(targets, indices, 100, 60, numpy.random.default_rng(0))
        for batch in frames:
            counts = targets.items['frameCount'][batch]
            self.assertTrue(len(batch) == 1 or counts.max() * len(batch) <= 60)

    def test_batch_prefetcher(self):
        targets = distill.Targets(self.targets)
        prefetcher = distill.BatchPrefetcher(targets, targets.split(0), 4, 10 ** 6, numpy.random.default_rng(0), depth=1)
        batch = prefetcher.next()
        self.assertIn('unit', batch)
        # (the queue full - the thread waits on it)
        deadline = time.time() + 5
        while not prefetcher.queue.full() and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.3)
        prefetcher.close()
        self.assertFalse(prefetcher.thread.is_alive())

        # An empty epoch - the thread waits for close
        with unittest.mock.patch.object(distill, 'target_batches', return_value=[]):
            prefetcher = distill.BatchPrefetcher(targets, targets.split(0), 4, 10 ** 6, numpy.random.default_rng(0))
            time.sleep(0.05)
            prefetcher.close()

        # An error on the thread is raised by next
        with unittest.mock.patch.object(distill, 'target_batches', side_effect=ValueError('bad')):
            prefetcher = distill.BatchPrefetcher(targets, targets.split(0), 4, 10 ** 6, numpy.random.default_rng(0))
            with self.assertRaisesRegex(ValueError, 'bad'):
                prefetcher.next()
            prefetcher.close()

    def test_batch_prefetcher_stopping(self):
        # (stopped while waiting to put - the thread returns)
        targets = distill.Targets(self.targets)
        prefetcher = distill.BatchPrefetcher(targets, targets.split(0), 4, 10 ** 6, numpy.random.default_rng(0), depth=1)
        # (wait until the thread has tried to put into the full queue - deterministic, however loaded the machine)
        deadline = time.time() + 120
        while not prefetcher.queue.full() and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(prefetcher.queue.full())
        with unittest.mock.patch.object(prefetcher.queue, 'put', wraps=prefetcher.queue.put) as put:
            while put.call_count < 2 and time.time() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(put.call_count, 2)
            prefetcher.stopping.set()
            prefetcher.thread.join()
        prefetcher.thread.join()
        prefetcher.close()

    def test_student_config(self):
        config = distill.load_config(self.config)
        self.assertEqual(distill.student_config(config, 'small')['dim'], 12)
        with self.assertRaisesRegex(ValueError, 'unknown student big \\(the ladder: tiny, small\\)'):
            distill.student_config(config, 'big')

    def test_build_student(self):
        teacher, state = export.load_model(self.teacher)
        model, copied = distill.build_student({'dim': 8, 'decoderDim': 8}, teacher, state, 0.05)
        self.assertEqual(sorted(copied), ['eq_free', 'speaker.weight', 'style'])
        self.assertEqual(model.config['dim'], 8)
        self.assertFalse(model.speaker.weight.requires_grad)
        self.assertTrue(torch.equal(model.style, teacher.style))
        model, copied = distill.build_student({'decoderDim': 8}, teacher, state, 0.05, init_teacher=True)
        self.assertIn('embedding.weight', copied)
        self.assertNotIn('head.weight', copied)
        self.assertTrue(torch.equal(model.embedding.weight, teacher.embedding.weight))

    def test_losses(self):
        targets = distill.Targets(self.targets)
        batch = distill.collate_targets(targets, numpy.array([0, 1, 2]))
        teacher, state = export.load_model(self.teacher)
        student, _ = distill.build_student({'dim': 8, 'decoderDim': 8}, teacher, state, 0.05)
        column_mask = torch.ones(params.COUNT)
        outputs = distill.run_model(student, batch)
        self.assertEqual(outputs['unit'].shape, batch['unit'].shape)
        self.assertEqual(outputs['decoded'].shape[-1], 8)
        losses = distill.distill_losses(outputs, batch, column_mask)
        self.assertEqual(set(losses), {'frames', 'durPhone', 'durPause', 'durPaused'})
        self.assertTrue(all(float(value.detach()) >= 0 for value in losses.values()))
        # The teacher matches its own targets but for the float16 rounding; its pause KL is zero
        with torch.no_grad():
            teacher_outputs = distill.run_model(teacher, batch)
            teacher_losses = distill.distill_losses(teacher_outputs, batch, column_mask)
        self.assertLess(float(teacher_losses['frames']), 1e-3)
        self.assertLess(abs(float(teacher_losses['durPause'])), 1e-3)
        # (the predicted durations - the duration outputs' - given)
        self.assertEqual(distill.run_model(student, batch, batch['durations'] + 1)['unit'].shape[1], batch['unit'].shape[1])
        # The feature-matching loss
        projections = distill.FeatureProjections(student, teacher)
        loss = distill.feature_loss(projections, outputs, teacher_outputs, batch)
        self.assertGreater(float(loss.detach()), 0)

    def test_decode_features(self):
        teacher, _ = export.load_model(self.teacher)
        regulated = torch.randn(1, 5, 16)
        positions = torch.rand(1, 5, 5)
        mask = torch.ones(1, 5, 1)
        condition = teacher.condition(torch.tensor([0]))
        with torch.no_grad():
            features, unit = distill.decode_features(teacher, regulated, positions, mask, condition)
            expected = teacher.decode(regulated, positions, mask, condition)
        self.assertEqual(features.shape, (1, 5, 16))
        self.assertTrue(torch.allclose(unit, expected))

    def test_validate_student(self):
        targets = distill.Targets(self.targets)
        teacher, _ = export.load_model(self.teacher)
        teacher.train()
        result = distill.validate_student(teacher, targets, targets.split(1), 'cpu', torch.ones(params.COUNT), batch_size=2)
        self.assertTrue(teacher.training)
        self.assertEqual(set(result), {'frames', 'durPhone', 'durPause', 'durPaused', 'durationErrorFrames'})
        self.assertLess(result['frames'], 1e-3)
        self.assertEqual(result['durationErrorFrames'], 0.0)

    def test_checkpoint_state(self):
        teacher, state = export.load_model(self.teacher)
        checkpoint = distill.checkpoint_state(teacher, state, 7, {'name': 'x'}, {'total': 1.0}, {'distill': {'student': {}}})
        self.assertEqual(checkpoint['step'], 7)
        self.assertEqual(checkpoint['args'], {'name': 'x', 'cremad': 0.2})
        self.assertEqual(checkpoint['speakers'], model_fixtures.SPEAKERS)
        self.assertIn('distill', checkpoint)
        state['args'] = {}
        self.assertEqual(distill.checkpoint_state(teacher, state, 1, {})['args'], {'cremad': 0})

    def test_train_student(self):
        validate = distill.validate_student
        calls = []

        # (each validation worse than the last - the best checkpoint stays the first's)
        def worse(*args, **kwargs):
            result = validate(*args, **kwargs)
            calls.append(result)
            result['frames'] += len(calls)
            return result
        with unittest.mock.patch.object(distill, 'validate_student', side_effect=worse):
            self.assertEqual(self.run_main(['train', '--config', self.config, '--device', 'cpu', '--log-every', '2',
                                            '--keep-every', '2', '--targets', self.targets]), 0)
        directory = os.path.join(distill.RUNS_DIR, 'distill-tiny')
        best = torch.load(os.path.join(directory, 'best.pt'), weights_only=False)
        last = torch.load(os.path.join(directory, 'last.pt'), weights_only=False)
        self.assertEqual((best['step'], last['step']), (2, 3))
        self.assertTrue(os.path.exists(os.path.join(directory, 'ckpt-2.pt')))
        self.assertFalse(os.path.exists(os.path.join(directory, 'ckpt-3.pt')))
        self.assertEqual(best['distill']['targets'], self.targets)
        self.assertEqual(best['config']['dim'], 8)
        with open(os.path.join(directory, 'log.jsonl'), encoding='utf-8') as file:
            rows = [json.loads(line) for line in file]
        self.assertEqual([list(row)[0] for row in rows], ['train', 'val', 'val'])
        self.assertEqual(rows[0]['train']['step'], 2)

    def test_train_student_options(self):
        # A JSON config, the teacher's tensors, feature matching, the settings overridden, a skipped non-finite step
        clip = torch.nn.utils.clip_grad_norm_
        norms = []

        def clip_once_infinite(*args, **kwargs):
            norm = clip(*args, **kwargs)
            norms.append(norm)
            return torch.tensor(float('inf')) if len(norms) == 1 else norm
        with unittest.mock.patch('torch.nn.utils.clip_grad_norm_', side_effect=clip_once_infinite), \
                unittest.mock.patch('torch.backends.mps.is_available', return_value=False), warnings.catch_warnings():
            # (the skipped first step - torch's warning that the schedule stepped before the optimizer)
            warnings.filterwarnings('ignore', 'Detected call of `lr_scheduler.step\\(\\)` before', UserWarning)
            self.assertEqual(self.run_main(['train', '--config', self.config, '--student-config', '{"decoderDim": 8}',
                                            '--name', 'options', '--teacher', self.teacher, '--steps', '2', '--batch', '2',
                                            '--lr', '0.01', '--warmup', '0', '--init-teacher', '--feature-weight', '0.5',
                                            '--log-every', '1']), 0)
        state = torch.load(os.path.join(distill.RUNS_DIR, 'distill-options', 'best.pt'), weights_only=False)
        self.assertEqual(state['args']['settings']['steps'], 2)
        self.assertEqual(state['args']['settings']['batch'], 2)
        self.assertEqual(state['config']['dim'], 16)
        with open(os.path.join(distill.RUNS_DIR, 'distill-options', 'log.jsonl'), encoding='utf-8') as file:
            rows = [json.loads(line) for line in file]
        self.assertIn('features', rows[0]['train'])

    def test_train_student_step_options(self):
        # Every training step option (Muon and the lean step, the EMA, the wsd schedule, zero-initialized projections
        # past the teacher's tensors); each validation worse than the last - the EMA's best stays the first's
        validate = distill.validate_student
        calls = []

        def worse(*args, **kwargs):
            result = validate(*args, **kwargs)
            calls.append(result)
            result['frames'] += len(calls)
            return result
        with unittest.mock.patch.object(distill, 'validate_student', side_effect=worse), \
                unittest.mock.patch.object(distill.optim, 'zero_init_projections', wraps=distill.optim.zero_init_projections) as zero:
            self.assertEqual(self.run_main(['train', '--config', self.config, '--device', 'cpu', '--targets', self.targets,
                                            '--name', 'step-options', '--optimizer', 'muon', '--ema', '0.9', '--schedule', 'wsd',
                                            '--cooldown', '0.5', '--cooldown-shape', '1-sqrt', '--zero-init-proj', '--init-teacher',
                                            '--max-frames', '3000']), 0)
        zero.assert_called_once()
        directory = os.path.join(distill.RUNS_DIR, 'distill-step-options')
        ema_best = torch.load(os.path.join(directory, 'ema-best.pt'), weights_only=False)
        ema_last = torch.load(os.path.join(directory, 'ema-last.pt'), weights_only=False)
        self.assertEqual((ema_best['step'], ema_last['step']), (2, 3))
        settings = ema_best['args']['settings']
        self.assertEqual((settings['optimizer'], settings['ema'], settings['schedule'], settings['cooldown'], settings['cooldownShape'],
                          settings['zeroInitProj'], settings['maxFrames'], settings['leanClip']),
                         ('muon', 0.9, 'wsd', 0.5, '1-sqrt', True, 3000, False))
        with open(os.path.join(directory, 'log.jsonl'), encoding='utf-8') as file:
            rows = [json.loads(line) for line in file]
        self.assertEqual([list(row)[0] for row in rows], ['emaVal', 'val', 'emaVal', 'val'])

        # The lean step alone, with feature matching's projections
        self.assertEqual(self.run_main(['train', '--config', self.config, '--device', 'cpu', '--targets', self.targets,
                                        '--name', 'lean', '--lean-clip', '--feature-weight', '0.5', '--steps', '1']), 0)
        state = torch.load(os.path.join(distill.RUNS_DIR, 'distill-lean', 'best.pt'), weights_only=False)
        self.assertTrue(state['args']['settings']['leanClip'])

    def test_tee(self):
        first, second = io.StringIO(), io.StringIO()
        tee = distill.Tee(first, second)
        self.assertEqual(tee.write('hello'), 5)
        tee.flush()
        self.assertEqual((first.getvalue(), second.getvalue()), ('hello', 'hello'))

    def test_polish(self):
        with helpers.captured():
            distill.main(['train', '--config', self.config, '--device', 'cpu', '--targets', self.targets, '--name', 'p',
                          '--steps', '1'])
        init = os.path.join(distill.RUNS_DIR, 'distill-p', 'best.pt')
        self.assertEqual(self.run_main(['polish', '--config', self.config, '--init', init, '--device', 'cpu', '--name', 'shine']), 0)
        directory = os.path.join(distill.RUNS_DIR, 'distill-shine')
        for name in ('best.pt', 'last.pt'):
            state = torch.load(os.path.join(directory, name), weights_only=False)
            self.assertEqual(state['distill']['polishedFrom'], init)
            self.assertEqual(state['distill']['targets'], self.targets)
            self.assertEqual(state['step'], 2)
            self.assertEqual(state['args']['init_critic'], self.teacher)
            self.assertEqual(state['config']['dim'], 8)
        with open(os.path.join(directory, 'train.log'), encoding='utf-8') as file:
            self.assertIn('VAL', file.read())

        # The defaults - the student's run's best.pt, the teacher given, the recipe's steps overridden
        os.makedirs(os.path.join(distill.RUNS_DIR, 'distill-tiny'), exist_ok=True)
        torch.save(torch.load(init, weights_only=False), os.path.join(distill.RUNS_DIR, 'distill-tiny', 'best.pt'))
        with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(self.run_main(['polish', '--config', self.config, '--teacher', self.teacher, '--steps', '1']), 0)
        state = torch.load(os.path.join(distill.RUNS_DIR, 'distill-tiny-polish', 'last.pt'), weights_only=False)
        self.assertEqual(state['step'], 1)

        # trainmulti.py's flags added (--train-args)
        self.assertEqual(self.run_main(['polish', '--config', self.config, '--init', init, '--device', 'cpu', '--name', 'shine-ema',
                                        '--train-args', '--ema 0.9 --lean-clip']), 0)
        state = torch.load(os.path.join(distill.RUNS_DIR, 'distill-shine-ema', 'last.pt'), weights_only=False)
        self.assertEqual((state['args']['ema'], state['args']['lean_clip']), (0.9, True))
        self.assertTrue(os.path.exists(os.path.join(distill.RUNS_DIR, 'distill-shine-ema', 'ema-best.pt')))


class TestEvaluation(DistillCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.student = model_fixtures.write_checkpoint(os.path.join(distill.RUNS_DIR, 'distill-tiny', 'best.pt'), seed=4)

    def tearDown(self):
        for name in ('eval', 'targets'):
            directory = os.path.join(distill.DISTILL_DIR, name)
            if os.path.isdir(directory):
                for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
                    for filename in filenames:
                        os.remove(os.path.join(dirpath, filename))
                    for dirname in dirnames:
                        os.rmdir(os.path.join(dirpath, dirname))

    def test_model_checkpoint(self):
        config = distill.load_config(self.config)
        self.assertEqual(distill.model_checkpoint('teacher', config), self.teacher)
        os.makedirs(os.path.join(distill.DISTILL_DIR, 'targets'), exist_ok=True)
        with open(os.path.join(distill.DISTILL_DIR, 'targets', 'index.json'), 'w', encoding='utf-8') as file:
            json.dump({'teacher': '/other/teacher.pt'}, file)
        self.assertEqual(distill.model_checkpoint('teacher', config), '/other/teacher.pt')
        self.assertEqual(distill.model_checkpoint('s64', config), os.path.join(distill.RUNS_DIR, 'distill-s64', 'best.pt'))

    def score_model(self, model_path, out_dir, settings, reference, whisper_workers):
        self.assertTrue(os.path.exists(model_path + '.json'))
        self.assertEqual(settings['rate'], 16000)
        self.assertEqual(whisper_workers, 2)
        mos = 3.0 if reference is None else 3.5
        return {'panel': {'mos': mos, 'wer': 2.0, **({} if reference is None else {'similarity': 0.9})}, 'outDir': out_dir}

    def test_evaluate_model(self):
        config = distill.load_config(self.config)
        os.makedirs(os.path.join(distill.DISTILL_DIR, 'eval', 'tiny'))
        with unittest.mock.patch.object(distill.panel, 'score_model', side_effect=self.score_model):
            result = distill.evaluate_model('tiny', self.student, config, whisper_workers=2)
        self.assertEqual(result['name'], 'tiny')
        self.assertEqual(result['step'], 3)
        self.assertEqual(result['config']['dim'], 16)
        self.assertEqual(result['panel']['mos'], 3.0)
        self.assertGreater(result['size']['moduleBytes'], result['size']['binBytes'])
        out_dir = os.path.join(distill.DISTILL_DIR, 'eval', 'tiny')
        with open(os.path.join(out_dir, 'result.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file), result)
        with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file)['defaultSpeaker'], 1)
        # (no panel)
        result = distill.evaluate_model('tiny', self.student, config, listen_panel=False)
        self.assertNotIn('panel', result)

    def test_evaluate_module(self):
        config = distill.load_config(self.config)
        model, state = export.load_model(self.student)
        export_dir = os.path.join(self.root, 'module-export')
        export.export(model, state, 'model', export_dir)
        text, _ = package.package(os.path.join(export_dir, 'model'))
        module_path = os.path.join(self.root, 'voiceModel.js')
        with open(module_path, 'w', encoding='utf-8') as file:
            file.write(text)
        with unittest.mock.patch.object(distill.panel, 'score_model', side_effect=self.score_model):
            result = distill.evaluate_model('baseline', module_path, config, reference={'speakers': {}}, whisper_workers=2)
        self.assertEqual(result['checkpoint'], module_path)
        self.assertIsNone(result['step'])
        self.assertEqual(result['parameters'], export.parameter_count(model))
        self.assertEqual(result['size']['encoding'], 'module')
        self.assertEqual(result['size']['moduleBytes'], len(text.encode('utf-8')))
        self.assertEqual(result['size']['binBytes'], os.path.getsize(os.path.join(export_dir, 'model.bin')))
        self.assertEqual(result['panel']['mos'], 3.5)
        result = distill.evaluate_module('baseline', module_path, config, None, False, 2,
                                         os.path.join(distill.DISTILL_DIR, 'eval', 'baseline'), time.time())
        self.assertNotIn('panel', result)

    def write_export(self, name):
        out_dir = os.path.join(distill.DISTILL_DIR, 'eval', name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'model.json'), 'w', encoding='utf-8') as file:
            json.dump({'speakers': [{'key': key} for key in model_fixtures.SPEAKERS]}, file)
        return out_dir

    def test_measure_speed(self):
        config = distill.load_config(self.config)
        for name in ('a', 'b'):
            self.write_export(name)
        reports = iter([
            {'networkSeconds': 2.0, 'renderSeconds': 4.0}, {'networkSeconds': 2.0, 'renderSeconds': 2.0},
            {'networkSeconds': 1.0, 'renderSeconds': 5.0}, {'networkSeconds': 3.0, 'renderSeconds': 1.0},
            {'networkSeconds': 1.5, 'renderSeconds': 9.0}, {'networkSeconds': 3.0, 'renderSeconds': 3.0}
        ])

        def speak(model_path, batch, batch_path, rate):
            self.assertTrue(model_path.endswith('model'))
            self.assertTrue(batch_path.endswith('speed-batch.json'))
            self.assertEqual(rate, 16000)
            self.assertEqual({item['speaker'] for item in batch}, {1})
            for item in batch:
                with open(item['wav'], 'wb') as file:
                    file.write(b'RIFF')
            return {'audioSeconds': 12.0, **next(reports)}
        with unittest.mock.patch.object(distill.panel, 'speak', side_effect=speak), \
                unittest.mock.patch.object(distill.panel, 'panel_texts', return_value=[{'id': 'x', 'text': 'Hello.'}]):
            best = distill.measure_speed(['a', 'b'], config)
        self.assertEqual((best['a']['networkSeconds'], best['a']['renderSeconds']), (1.0, 4.0))
        self.assertEqual((best['b']['networkSeconds'], best['b']['renderSeconds']), (2.0, 1.0))
        self.assertEqual(best['a']['realTimeNetwork'], 12.0)
        self.assertEqual(best['b']['realTimeTotal'], 4.0)
        self.assertEqual(os.listdir(os.path.join(distill.DISTILL_DIR, 'eval', 'a')), ['model.json'])

    def test_render_samples(self):
        config = distill.load_config(self.config)
        self.write_export('tiny')
        with unittest.mock.patch.object(distill.panel, 'speak') as speak:
            paths = distill.render_samples('tiny', config)
        self.assertEqual(paths, [os.path.join(distill.SAMPLES_DIR, 'distill-tiny-arctic-bdl-melville-moby.wav')])
        batch = speak.call_args[0][1]
        self.assertEqual(batch[0]['speaker'], 1)
        self.assertTrue(batch[0]['text'].startswith('Call me Ishmael.'))

    def test_brief(self):
        result = {'name': 'x', 'utterances': {'a': 1}, 'speakers': {'s': {'mos': 3, 'embedding': [1]}}}
        self.assertEqual(distill.brief(result), {'name': 'x', 'speakers': {'s': {'mos': 3}}})
        self.assertEqual(distill.brief({'name': 'y'}), {'name': 'y'})

    def test_evaluate(self):
        calls = []

        def evaluate_model(name, checkpoint, config, reference, whisper_workers):
            self.assertIn('eval', config)
            calls.append((name, checkpoint, reference, whisper_workers))
            return {'name': name, 'parameters': 10, 'size': {'moduleBytes': 100, 'moduleGzipBytes': 90}, 'seconds': 1.0,
                    **({'panel': {'mos': 3.0}} if name != 'tiny' else {})}
        with unittest.mock.patch.object(distill, 'evaluate_model', side_effect=evaluate_model):
            self.assertEqual(self.run_main(['eval', '--config', self.config, '--whisper-workers', '3']), 0)
            self.assertEqual([call[0] for call in calls], ['teacher', 'tiny', 'small'])
            self.assertIsNone(calls[0][2])
            self.assertEqual(calls[1][2]['name'], 'teacher')
            self.assertEqual(calls[1][1], self.student)
            self.assertEqual(calls[1][3], 3)
            # (a student without the teacher - the teacher's result from its result.json; NAME=CHECKPOINT)
            calls.clear()
            out_dir = os.path.join(distill.DISTILL_DIR, 'eval', 'teacher')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as file:
                json.dump({'name': 'saved'}, file)
            self.assertEqual(self.run_main(['eval', '--config', self.config, 'x=/a/b.pt', 'tiny']), 0)
            self.assertEqual(calls[0][:3], ('x', '/a/b.pt', {'name': 'saved'}))
            self.assertEqual(calls[1][2], {'name': 'saved'})

    def write_result(self, name, parameters, mos, wer, similarity=None):
        out_dir = os.path.join(distill.DISTILL_DIR, 'eval', name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump({'name': name, 'checkpoint': f'{name}.pt', 'parameters': parameters,
                       'size': {'moduleBytes': parameters * 2, 'moduleGzipBytes': parameters},
                       'panel': {'mos': mos, 'wer': wer, **({'similarity': similarity} if similarity is not None else {})},
                       'speakers': {'arctic-bdl': {'mos': mos, 'embedding': [1.0]}}}, file)

    def test_pick(self):
        self.write_result('teacher', 1000, 3.5, 2.0)
        self.write_result('big', 500, 3.48, 2.5, 0.95)
        self.write_result('big-polish', 500, 3.49, 2.4, 0.95)
        self.write_result('small', 100, 3.0, 2.0, 0.95)
        os.makedirs(os.path.join(distill.DISTILL_DIR, 'eval', 'stray'))
        speed = {name: {'realTimeNetwork': 100.0, 'realTimeTotal': 10.0} for name in ('teacher', 'big', 'big-polish', 'small')}
        with unittest.mock.patch.object(distill, 'measure_speed', return_value=speed) as measure, \
                unittest.mock.patch.object(distill, 'render_samples', side_effect=lambda name, config: [f'{name}.wav']):
            self.assertEqual(self.run_main(['pick', '--config', self.config, '--speed-repeats', '2']), 0)
        self.assertEqual(measure.call_args[0][2], 2)
        with open(os.path.join(distill.DISTILL_DIR, 'summary.json'), encoding='utf-8') as file:
            summary = json.load(file)
        self.assertEqual(summary['chosen'], 'big-polish')
        self.assertEqual(summary['against'], 'teacher')
        rows = {row['name']: row for row in summary['rows']}
        self.assertNotIn('within', rows['teacher'])
        self.assertEqual(rows['small']['within'], {'mos': False, 'wer': True, 'similarity': True})
        self.assertEqual(rows['big']['realTimeTotal'], 10.0)
        self.assertEqual(summary['samples'], ['teacher.wav', 'big-polish.wav', 'big.wav', 'small.wav'])
        self.assertNotIn('embedding', summary['results']['big']['speakers']['arctic-bdl'])

        # Against a baseline, no speed, the samples named - and no student within the tolerance fails
        self.write_result('baseline', 800, 3.9, 1.0, 1.0)
        with unittest.mock.patch.object(distill, 'measure_speed') as measure, \
                unittest.mock.patch.object(distill, 'render_samples', side_effect=lambda name, config: [f'{name}.wav']):
            self.assertEqual(self.run_main(['pick', '--config', self.config, '--no-speed', '--against', 'baseline',
                                            '--samples', 'small']), 1)
        measure.assert_not_called()
        with open(os.path.join(distill.DISTILL_DIR, 'summary.json'), encoding='utf-8') as file:
            summary = json.load(file)
        self.assertIsNone(summary['chosen'])
        self.assertEqual(summary['samples'], ['teacher.wav', 'small.wav'])
        self.assertNotIn('within', {row['name']: row for row in summary['rows']}['baseline'])
        with unittest.mock.patch.object(distill, 'render_samples') as render:
            self.run_main(['pick', '--config', self.config, '--no-speed', '--no-samples'])
        render.assert_not_called()

    def test_ladder(self):
        calls = []

        def main(argv):
            calls.append(argv)
            return 7 if argv[0] == 'pick' else 0
        with unittest.mock.patch.object(distill, 'main', side_effect=main):
            args = unittest.mock.Mock(config=self.config, device='cpu', polish=['tiny'])
            self.assertEqual(distill.ladder(args), 7)
            common = ['--config', self.config, '--device', 'cpu']
            self.assertEqual(calls, [
                ['targets'] + common, ['train', '--student', 'small'] + common, ['polish', '--student', 'tiny'] + common,
                ['eval', '--config', self.config, 'teacher', 'tiny', 'small', 'tiny-polish'], ['pick', '--config', self.config]
            ])
            # (the targets present, no device, nothing to polish)
            calls.clear()
            os.makedirs(os.path.join(distill.DISTILL_DIR, 'targets'), exist_ok=True)
            with open(os.path.join(distill.DISTILL_DIR, 'targets', 'index.json'), 'w', encoding='utf-8') as file:
                file.write('{}')
            distill.ladder(unittest.mock.Mock(config=self.config, device=None, polish=None))
            self.assertEqual(calls[0], ['train', '--student', 'small', '--config', self.config])
            self.assertEqual(len(calls), 3)

    def test_main_ladder(self):
        with unittest.mock.patch.object(distill, 'ladder', return_value=0) as ladder:
            self.assertEqual(distill.main(['ladder', '--config', self.config, '--polish', 'a']), 0)
        self.assertEqual(ladder.call_args[0][0].polish, ['a'])

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['distill', '--help']), helpers.captured() as (stdout, _), \
                self.assertRaises(SystemExit) as raised, warnings.catch_warnings():
            # (runpy's warning that the module is imported already - the tests imported it)
            warnings.filterwarnings('ignore', "'picovox_train.distill' found in sys.modules", RuntimeWarning)
            runpy.run_module('picovox_train.distill', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('usage:', stdout.getvalue())
