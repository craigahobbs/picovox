# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error
import warnings

import numpy

from picovox_train import books, ctc, data, frontend, readings, tts
from tests import ctc_fixtures, data_fixtures, helpers
from tests.ctc_fixtures import fake_recognizer, greedy_of, tts_corpus, tts_settings
from tests.data_fixtures import FakeResponse
from tests.helpers import write_wav


# A Gutenberg-format book of paragraphs
def write_book(path, paragraphs):
    with open(path, 'w', encoding='utf-8') as file:
        file.write('The Project Gutenberg eBook\n\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n')
        file.write('\n\n'.join(paragraphs))
        file.write('\n\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nThe license.\n')


# The WAV file bytes of samples
def wav_bytes(samples, rate=24000):
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'a.wav')
        write_wav(path, samples, rate)
        with open(path, 'rb') as file:
            return file.read()


# A fake Ollama - urlopen answering the model list (the model at digest), a pull, the speech endpoint (the WAV
# bytes of speech), and the unload; its requests [(path, body)]

# The process pools in this process (the workers' code covered)
def setUpModule():  # pylint: disable=invalid-name
    patcher = unittest.mock.patch('concurrent.futures.ProcessPoolExecutor', ctc_fixtures.FakeExecutor)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)

class FakeOllama:

    def __init__(self, digest='0123456789abcdef', speech=None, models=None):
        self.digest = digest
        self.speech = speech if speech is not None else wav_bytes(data_fixtures.buzz(0.5, 24000))
        self.models = models
        self.requests = []

    def __call__(self, request, timeout):
        path = request.full_url.split('11434', 1)[1]
        body = None if request.data is None else json.loads(request.data)
        self.requests.append((path, body, timeout))
        if path == '/api/tags':
            models = self.models if self.models is not None else [{'name': 'other:latest', 'digest': 'ffff'},
                                                                  {'name': 'phonon:latest', 'digest': self.digest}]
            return FakeResponse(json.dumps({'models': models}).encode('utf-8'))
        if path == '/v1/audio/speech':
            return FakeResponse(self.speech)
        return FakeResponse(b'{}')

    # The requests' paths
    def paths(self):
        return [path for path, _, _ in self.requests]


# The commands' arguments
def generate_args(jobs=1):
    return argparse.Namespace(recipe=tts.RECIPE_PATH, jobs=jobs)


def align_args(batch=2, device='cpu'):
    return argparse.Namespace(recipe=tts.RECIPE_PATH, jobs=1, batch=batch, device=device)


def build_args(out, limit=None):
    return argparse.Namespace(recipe=tts.RECIPE_PATH, jobs=1, out=out, limit=limit)


# Rewrite a corpus's alignments.jsonl - {id: record changes}
def edit_alignments(directory, changes):
    path = os.path.join(directory, 'alignments.jsonl')
    with open(path, encoding='utf-8') as file:
        records = [json.loads(line) for line in file]
    with open(path, 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(dict(record, **changes.get(record['id'], {}))) + '\n')


class TestBasics(unittest.TestCase):

    def test_log(self):
        with helpers.captured() as (stdout, stderr):
            tts.log('hello')
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ('', 'hello\n'))

    def test_load_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'train.json')
            with open(path, 'w', encoding='utf-8') as file:
                json.dump({'tts': tts_settings()}, file)
            self.assertEqual(tts.load_settings(path), tts_settings())
        self.assertIn('engine', tts.load_settings())

    def test_corpus_dir_and_paths(self):
        settings = tts_settings()
        self.assertEqual(tts.corpus_dir(settings), os.path.join(tts.TTS_DIR, 'phonon-0123456789ab'))
        settings['engine'] = dict(settings['engine'], model='phonon:1b')
        self.assertEqual(tts.corpus_dir(settings), os.path.join(tts.TTS_DIR, 'phonon-1b-0123456789ab'))
        self.assertEqual(tts.clip_id('Trevor', 'Call me Ishmael.'), tts.clip_id('Trevor', 'Call me Ishmael.'))
        self.assertNotEqual(tts.clip_id('Trevor', 'Call me Ishmael.'), tts.clip_id('Harper', 'Call me Ishmael.'))
        self.assertEqual(len(tts.clip_id('Trevor', 'Call me Ishmael.')), 16)
        self.assertEqual(tts.wav_path('dir', {'voice': 'Trevor', 'id': 'abc'}), os.path.join('dir', 'wav', 'Trevor', 'abc.wav'))

    def test_text_split(self):
        splits = [tts.text_split(f'Sentence {number}.', {'test': 5, 'val': 5}) for number in range(1000)]
        self.assertEqual(set(splits), {'train', 'val', 'test'})
        self.assertLess(splits.count('test'), 100)
        self.assertGreater(splits.count('train'), 800)
        self.assertEqual(tts.text_split('Sentence 1.', {'test': 5, 'val': 5}), splits[1])
        self.assertEqual(set(tts.text_split(f'Sentence {number}.', {'test': 0, 'val': 0}) for number in range(100)), {'train'})

    def test_dealing_order(self):
        voices = {'a': {'sex': 'male'}, 'b': {'sex': 'male'}, 'c': {'sex': 'female'}, 'd': {'sex': 'male'}}
        self.assertEqual(tts.dealing_order(voices), ['a', 'c', 'b', 'd'])
        self.assertEqual(tts.dealing_order({}), [])


class TestSelection(unittest.TestCase):

    def test_engine_text(self):
        result = frontend.tokenize('i think, Mr. Smith, i am here!')
        self.assertEqual(tts.engine_text(result['tokens'], result['words']), 'I think, mister smith, I am here!')
        result = frontend.tokenize('Who said that to us?')
        self.assertEqual(tts.engine_text(result['tokens'], result['words']), 'Who said that to us?')
        self.assertEqual(tts.engine_text(['<bos>', '<comma>'], []), '')

    def test_book_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'pg1.txt')
            write_book(path, ['CHAPTER IV', 'THE BEGINNING.', 'Call me Ishmael. Some years ago, never mind how long, I went to sea!',
                              '"Is it you?" she asked. And then', 'He saw Chapter II there.'])
            self.assertEqual(tts.book_candidates(path), [
                (2, 'Call me Ishmael.'), (2, 'Some years ago, never mind how long, I went to sea!'), (3, '"Is it you?"'),
                (3, 'she asked.')
            ])

    def test_check_sentences(self):
        sentences = ['Call me, Ishmael.', 'Who said that to us?', 'Is it? Yes it is.', 'Go now.', 'The cat sat on the mat today.',
                     'He said, and then']
        self.assertEqual(tts.check_sentences((sentences, 3, 6)),
                         ['Call me, ishmael.', 'Who said that to us?', None, None, None, 'He said, and then.'])

        # An engine text that doesn't tokenize back to its sentence's tokens
        tokenize = frontend.tokenize

        def changed(text, *args, **kwargs):
            result = tokenize(text, *args, **kwargs)
            return dict(result, tokens=result['tokens'][:1] + result['tokens'][2:]) if text == 'Call me, ishmael.' else result
        with unittest.mock.patch.object(frontend, 'tokenize', changed):
            self.assertEqual(tts.check_sentences((['Call me, Ishmael.', 'Call me.'], 1, 6)), [None, 'Call me.'])

    def test_checked(self):
        settings = tts_settings()
        with unittest.mock.patch('concurrent.futures.ProcessPoolExecutor', ctc_fixtures.FakeExecutor), \
                unittest.mock.patch.object(tts, 'CHECK_BATCH', 2):
            self.assertEqual(tts.checked(['Call me Ishmael.', 'Go.', 'Call me Ishmael.', 'The cat sat on the mat.'], settings, 1),
                             {'Call me Ishmael.': 'Call me ishmael.', 'Go.': None, 'The cat sat on the mat.': 'The cat sat on the mat.'})
            self.assertEqual(tts.checked([], settings, None), {})

    def test_excluded_texts(self):
        files = {tts.PANEL_TEXTS_PATH: {'texts': [{'text': 'Mr. Smith is here.'}]},
                 tts.RECIPE_PATH: {'samples': {'texts': [{'text': 'Call me, Ishmael!'}, {'text': 'Don\'t go.'}]}}}
        with helpers.json_files(tts, files):
            self.assertEqual(tts.excluded_texts(), ['mister smith is here', 'call me ishmael', 'don\'t go'])
        # The repository's texts
        self.assertTrue(tts.excluded_texts())

    def test_book_paths(self):
        with unittest.mock.patch.object(books, 'pinned_paths', return_value=['a.txt']) as pinned:
            self.assertEqual(tts.book_paths({'books': 'build/nn/books', 'pins': 'train/data/books.json', 'texts': ['a']}), ['a.txt'])
            tts.book_paths({'books': '/books', 'pins': '/pins.json', 'texts': ['b']})
        self.assertEqual(pinned.call_args_list[0].args, (os.path.join(tts.ROOT, 'build/nn/books'), ['a'],
                                                         os.path.join(tts.ROOT, 'train/data/books.json')))
        self.assertEqual(pinned.call_args_list[1].args, ('/books', ['b'], '/pins.json'))

    def test_selection_key(self):
        recipe = {'context': {'labels': {'texts': ['a']}, 'gate': {'words': ['lead']}}, 'distill': {'targets': {'texts': ['b']}}}
        with tempfile.TemporaryDirectory() as directory:
            pins = os.path.join(directory, 'books.json')
            with open(pins, 'w', encoding='utf-8') as file:
                file.write('{}')
            settings = tts_settings(texts=dict(tts_settings()['texts'], pins=pins))
            key = tts.selection_key(settings, recipe)
            self.assertEqual(len(key), 12)
            # The engine and the workers aren't the selection's
            self.assertEqual(tts.selection_key(dict(settings, workers=7, engine={}), recipe), key)
            self.assertNotEqual(tts.selection_key(dict(settings, seed=2), recipe), key)
            self.assertNotEqual(tts.selection_key(settings, dict(recipe, distill={'targets': {'texts': ['c']}})), key)
            with open(pins, 'w', encoding='utf-8') as file:
                file.write('{"a": 1}')
            self.assertNotEqual(tts.selection_key(settings, recipe), key)
        # The repository's pins (relative to the root)
        self.assertEqual(len(tts.selection_key(tts_settings(), recipe)), 12)


# The selection's books - the coverage book's and the fill book's paragraphs
COVERAGE_BOOK = [
    'A lead pipe lay on the record.',
    'They will lead the dogs home tonight. The lead singer sang very well. He was the 1st to take the lead.',
    'Lead on. Then the cat sat on the mat. The black cat ran away fast. A cat was here today.',
    'He came in 1st again.'
]
FILL_BOOK = [
    'The sun rose over the quiet hills. Birds sang in the tall green trees. The river ran cold and clear.',
    'A small boat drifted down the stream. Go. The old man smiled at the children. They played until the evening came.',
    'The lead singer sang very well. Rain fell softly on the roof all night. The fire burned low in the hearth.',
    'She read the letter twice before she slept.'
]


class TestSelectSentences(unittest.TestCase):

    RECIPE = {'context': {'labels': {'texts': ['cov.txt']}, 'gate': {'words': ['lead']}}, 'distill': {'targets': {'texts': ['fill.txt']}}}
    QUOTAS = {'lead': 4, 'record': 1, 'cat': 1, 'st': 1, 'tear': 1}

    # The books (pinned_paths patched to them), the quotas, the excluded texts, and the workers in this process
    @contextlib.contextmanager
    def books(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(books, 'pinned_paths',
                                           lambda books_dir, names, pins: [os.path.join(directory, name) for name in names]), \
                unittest.mock.patch.object(readings, 'coverage_quotas', return_value=self.QUOTAS) as quotas, \
                unittest.mock.patch.object(tts, 'excluded_texts', return_value=['they will lead the dogs home tonight']), \
                unittest.mock.patch('concurrent.futures.ProcessPoolExecutor', ctc_fixtures.FakeExecutor), \
                unittest.mock.patch.object(tts, 'CHECK_BATCH', 1), \
                helpers.captured() as (_, stderr):
            write_book(os.path.join(directory, 'cov.txt'), COVERAGE_BOOK)
            write_book(os.path.join(directory, 'fill.txt'), FILL_BOOK)
            yield quotas, stderr

    def test_select_sentences(self):
        with self.books() as (quotas, stderr):
            # Words for 40 words of fill
            settings = tts_settings(hoursPerVoice=(26 + 40) / 3600 / 3.0 / 2)
            chosen = tts.select_sentences(settings, self.RECIPE, 1)
        self.assertEqual(quotas.call_args.args, (settings['coverage'], ['lead']))
        coverage = [item['text'] for item in chosen if item['coverage']]
        fill = [item['text'] for item in chosen if not item['coverage']]
        self.assertEqual(chosen[:len(coverage)], [{'book': 'cov.txt', 'text': text, 'coverage': True} for text in coverage])
        self.assertEqual(coverage[0], 'A lead pipe lay on the record.')
        self.assertEqual(sum('cat' in text.split() for text in coverage), 1)
        self.assertEqual(coverage[1:], ['A cat was here today.', 'He was the first to take the lead.', 'The lead singer sang very well.'])
        coverage_words = sum(len(text.split()) for text in coverage)
        fill_words = sum(len(text.split()) for text in fill)
        self.assertEqual(coverage_words, 26)
        self.assertTrue(40 <= fill_words < 40 + 9)
        self.assertNotIn('Go.', fill)
        self.assertNotIn('The lead singer sang very well.', fill)
        self.assertEqual({item['book'] for item in chosen[len(coverage):]}, {'fill.txt'})
        self.assertIn('tts select: 4 of 5 quota words in the coverage books', stderr.getvalue())
        self.assertIn('4 coverage sentences (26 words)', stderr.getvalue())
        self.assertIn('2 quota words short of their quota, 1 not in the books', stderr.getvalue())

    def test_select_sentences_all(self):
        # More hours than the books - every fill sentence (the duplicate and the failed check left out)
        with self.books():
            chosen = tts.select_sentences(tts_settings(hoursPerVoice=1.0), self.RECIPE, None)
        fill = [item['text'] for item in chosen if not item['coverage']]
        self.assertEqual(len(fill), 9)
        self.assertEqual(len({item['text'] for item in chosen}), len(chosen))

    def test_selection(self):
        sentences = [{'book': 'pg1.txt', 'text': f'Sentence number {number}.', 'coverage': number < 2} for number in range(5)]
        settings = tts_settings()
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(tts, 'TTS_DIR', os.path.join(directory, 'tts')), \
                unittest.mock.patch.object(tts, 'selection_key', return_value='abc') as key, \
                unittest.mock.patch.object(tts, 'select_sentences', return_value=sentences) as select:
            recipe_path = os.path.join(directory, 'train.json')
            with open(recipe_path, 'w', encoding='utf-8') as file:
                json.dump({'recipe': 1}, file)
            clips = tts.selection(settings, 3, recipe_path)
            again = tts.selection(settings, 3, recipe_path)
            path = os.path.join(tts.corpus_dir(settings), 'selection-abc.json')
            with open(path, encoding='utf-8') as file:
                saved = json.load(file)
            self.assertFalse(os.path.exists(path + '.part'))
        self.assertEqual(select.call_count, 1)
        self.assertEqual(select.call_args.args, (settings, {'recipe': 1}, 3))
        self.assertEqual(key.call_args.args, (settings, {'recipe': 1}))
        self.assertEqual(again, clips)
        self.assertEqual(saved, {'engine': settings['engine'], 'clips': clips})
        # (each clip's voice by a hash of its text)
        self.assertEqual([clip['key'] for clip in clips], [tts.text_voice(clip['text'], settings['voices']) for clip in clips])
        self.assertEqual({clip['key'] for clip in clips}, {'male', 'female'})
        self.assertEqual(clips[1], {'id': tts.clip_id('Harper', 'Sentence number 1.'), 'key': 'female', 'voice': 'Harper',
                                    'sex': 'female', 'book': 'pg1.txt', 'text': 'Sentence number 1.',
                                    'split': tts.text_split('Sentence number 1.', settings['split']), 'coverage': True})
        self.assertFalse(clips[4]['coverage'])


class TestEngine(unittest.TestCase):

    ENGINE = tts_settings()['engine']

    def test_engine_request(self):
        with unittest.mock.patch('urllib.request.urlopen', side_effect=[FakeResponse(b'body'), FakeResponse(b'body')]) as urlopen:
            self.assertEqual(tts.engine_request(dict(self.ENGINE, url='http://localhost:11434/'), '/api/tags'), b'body')
            self.assertEqual(tts.engine_request(self.ENGINE, '/api/pull', {'model': 'phonon'}, timeout=5), b'body')
        first, second = (call.args[0] for call in urlopen.call_args_list)
        self.assertEqual((first.full_url, first.data, first.get_method()), ('http://localhost:11434/api/tags', None, 'GET'))
        self.assertEqual((second.full_url, json.loads(second.data), second.get_method()),
                         ('http://localhost:11434/api/pull', {'model': 'phonon'}, 'POST'))
        self.assertEqual(second.get_header('Content-type'), 'application/json')
        self.assertEqual([call.kwargs['timeout'] for call in urlopen.call_args_list], [60, 5])

    def test_engine_request_errors(self):
        error = urllib.error.HTTPError('http://localhost:11434/v1/audio/speech', 400, 'Bad Request', {}, io.BytesIO(b'unknown voice\n'))
        with unittest.mock.patch('urllib.request.urlopen', side_effect=error), \
                self.assertRaisesRegex(tts.EngineError, r'^http://localhost:11434/v1/audio/speech: HTTP 400 unknown voice$'):
            tts.engine_request(self.ENGINE, '/v1/audio/speech', {})
        error.close()
        with unittest.mock.patch('urllib.request.urlopen', side_effect=urllib.error.URLError('Connection refused')), \
                self.assertRaisesRegex(tts.EngineError, r'^Ollama isn\'t answering at http://localhost:11434 \(Connection refused\)'):
            tts.engine_request(self.ENGINE, '/api/tags')

    def test_model_digest(self):
        ollama = FakeOllama()
        with unittest.mock.patch('urllib.request.urlopen', ollama):
            self.assertEqual(tts.model_digest(self.ENGINE), '0123456789ab')
            self.assertIsNone(tts.model_digest(dict(self.ENGINE, model='phonon:1b')))
            ollama.models = [{'name': 'phonon:1b', 'digest': 'abcdef0123456789'}]
            self.assertEqual(tts.model_digest(dict(self.ENGINE, model='phonon:1b')), 'abcdef012345')

    def test_check_engine(self):
        ollama = FakeOllama()
        with unittest.mock.patch('urllib.request.urlopen', ollama):
            tts.check_engine(self.ENGINE)
        self.assertEqual(ollama.paths(), ['/api/tags'])

        # Missing - pulled
        ollama = FakeOllama(models=[])

        def pull(request, timeout):
            if request.full_url.endswith('/api/pull'):
                ollama.models = None
            return ollama(request, timeout)
        with unittest.mock.patch('urllib.request.urlopen', pull), helpers.captured() as (_, stderr):
            tts.check_engine(self.ENGINE)
        self.assertEqual(ollama.paths(), ['/api/tags', '/api/pull', '/api/tags'])
        self.assertEqual(ollama.requests[1][1:], ({'model': 'phonon', 'stream': False}, 3600))
        self.assertEqual(stderr.getvalue(), 'tts: pulling phonon (ollama pull)\n')

        # Another digest
        with unittest.mock.patch('urllib.request.urlopen', FakeOllama(digest='fedcba9876543210')), \
                self.assertRaisesRegex(tts.EngineError, '^phonon is digest fedcba987654, not the recipe\'s 0123456789ab'):
            tts.check_engine(self.ENGINE)

    def test_speak(self):
        ollama = FakeOllama(speech=b'RIFF')
        with unittest.mock.patch('urllib.request.urlopen', ollama):
            self.assertEqual(tts.speak(self.ENGINE, 'Trevor', 'Hello.'), b'RIFF')
        self.assertEqual(ollama.requests, [('/v1/audio/speech', {'model': 'phonon', 'voice': 'Trevor', 'input': 'Hello.'}, 300)])

        # A transient failure retried
        refused = urllib.error.URLError('refused')
        with unittest.mock.patch('urllib.request.urlopen', side_effect=[refused, refused, FakeResponse(b'RIFF')]), \
                unittest.mock.patch('time.sleep') as sleep, helpers.captured() as (_, stderr):
            self.assertEqual(tts.speak(self.ENGINE, 'Trevor', 'Hello.'), b'RIFF')
        self.assertEqual([call.args for call in sleep.call_args_list], [(1,), (2,)])
        self.assertEqual(stderr.getvalue().count('  retry (Ollama isn\'t answering'), 2)

        # Every attempt failed
        with unittest.mock.patch('urllib.request.urlopen', side_effect=refused) as urlopen, \
                unittest.mock.patch('time.sleep'), helpers.captured(), self.assertRaises(tts.EngineError):
            tts.speak(self.ENGINE, 'Trevor', 'Hello.', attempts=3)
        self.assertEqual(urlopen.call_count, 3)

        # An HTTP 4xx isn't retried
        error = urllib.error.HTTPError('http://x', 404, 'Not Found', {}, io.BytesIO(b'no such voice'))
        with unittest.mock.patch('urllib.request.urlopen', side_effect=error) as urlopen, \
                self.assertRaisesRegex(tts.EngineError, 'HTTP 404 no such voice'):
            tts.speak(self.ENGINE, 'Nobody', 'Hello.')
        error.close()
        self.assertEqual(urlopen.call_count, 1)

        # An HTTP 5xx is
        error = urllib.error.HTTPError('http://x', 500, 'Server Error', {}, io.BytesIO(b'busy'))
        with unittest.mock.patch('urllib.request.urlopen', side_effect=[error, FakeResponse(b'RIFF')]), \
                unittest.mock.patch('time.sleep'), helpers.captured():
            self.assertEqual(tts.speak(self.ENGINE, 'Trevor', 'Hello.'), b'RIFF')
        error.close()

        # No attempts
        self.assertIsNone(tts.speak(self.ENGINE, 'Trevor', 'Hello.', attempts=0))

    def test_unload(self):
        ollama = FakeOllama()
        with unittest.mock.patch('urllib.request.urlopen', ollama):
            tts.unload(self.ENGINE)
        self.assertEqual(ollama.requests, [('/api/generate', {'model': 'phonon', 'keep_alive': 0}, 60)])
        with unittest.mock.patch('urllib.request.urlopen', side_effect=urllib.error.URLError('refused')), \
                helpers.captured() as (_, stderr):
            tts.unload(self.ENGINE)
        self.assertIn('tts: the model wasn\'t unloaded (Ollama isn\'t answering', stderr.getvalue())


# The test corpus's clips - (voice key, text, split)
TEXTS = [
    ('male', 'Call me Ishmael.', 'train'),
    ('female', 'A lead pipe lay on the record.', 'train'),
    ('male', 'The cat sat on the mat.', 'val'),
    ('female', 'It is cold here tonight.', 'test'),
    ('male', 'Look at the dark night.', 'train'),
    ('female', 'The best home is here.', 'train'),
    ('male', 'Here is the best cat.', 'train'),
    ('female', 'It is dark at night.', 'train')
]


class TestGenerate(unittest.TestCase):

    def test_generate_command(self):
        with tts_corpus(TEXTS[:4], aligned=False) as (directory, _, clips), helpers.captured() as (_, stderr):
            for clip in clips[1:]:
                os.remove(tts.wav_path(directory, clip))
            os.rmdir(os.path.dirname(tts.wav_path(directory, clips[1])))
            ollama = FakeOllama()
            with unittest.mock.patch('urllib.request.urlopen', ollama), unittest.mock.patch.object(tts, 'GENERATE_PROGRESS', 2):
                self.assertEqual(tts.generate_command(generate_args()), 0)
            present = [os.path.exists(tts.wav_path(directory, clip)) for clip in clips]
            parts = [name for _, _, names in os.walk(directory) for name in names if name.endswith('.part')]
            # Every clip present - the engine not asked
            with unittest.mock.patch('urllib.request.urlopen') as urlopen:
                self.assertEqual(tts.generate_command(generate_args()), 0)
            self.assertEqual(urlopen.call_count, 0)
        self.assertEqual(present, [True] * 4)
        self.assertEqual(parts, [])
        self.assertEqual(ollama.paths(), ['/api/tags'] + ['/v1/audio/speech'] * 3 + ['/api/generate'])
        self.assertEqual(sorted(body['input'] for path, body, _ in ollama.requests if path == '/v1/audio/speech'),
                         sorted(clip['text'] for clip in clips[1:]))
        output = stderr.getvalue()
        self.assertIn('tts generate: 4 clips in ', output)
        self.assertIn(', 3 to generate\n', output)
        self.assertIn('  2/3 clips, 0.00 hours, ', output)
        self.assertIn('  3/3 clips, 0.00 hours, ', output)
        self.assertIn(', 0 to generate\n', output)

    def test_generate_command_errors(self):
        # Not a mono WAV
        stereo = wav_bytes(numpy.zeros((2400, 2)))
        with tts_corpus(TEXTS[:2], aligned=False) as (directory, _, clips), helpers.captured() as (_, stderr):
            os.remove(tts.wav_path(directory, clips[0]))
            with unittest.mock.patch('urllib.request.urlopen', FakeOllama(speech=stereo)):
                self.assertEqual(tts.main(['generate']), 1)
            self.assertFalse(os.path.exists(tts.wav_path(directory, clips[0])))
            # An empty WAV
            with unittest.mock.patch('urllib.request.urlopen', FakeOllama(speech=wav_bytes(numpy.zeros(0)))), \
                    self.assertRaisesRegex(tts.EngineError, 'not a mono WAV'):
                tts.generate_command(generate_args())
        self.assertIn(f'tts: {clips[0]["id"]}: not a mono WAV\n', stderr.getvalue())


class TestAlign(unittest.TestCase):

    # The corpus's alignment records, in order
    def read_records(self, directory):
        with open(os.path.join(directory, 'alignments.jsonl'), encoding='utf-8') as file:
            return [json.loads(line) for line in file]

    def test_align_command(self):
        with tts_corpus(TEXTS[:4], aligned=False) as (directory, _, clips), fake_recognizer(), \
                unittest.mock.patch.object(readings, 'overrides', return_value={}) as overrides, \
                helpers.captured() as (_, stderr):
            # A clip too short to align, and one not generated
            write_wav(tts.wav_path(directory, clips[2]), numpy.zeros(4800), 24000)
            os.remove(tts.wav_path(directory, clips[3]))
            self.assertEqual(tts.align_command(align_args()), 0)
            first = self.read_records(directory)
            # Nothing changed - nothing to align
            self.assertEqual(tts.align_command(align_args()), 0)
            again = self.read_records(directory)
            # A measured reading changes a clip's phones; a new aligner realigns every clip
            overrides.return_value = {clips[1]['id']: {'1': 'P AY1 P'}}
            self.assertEqual(tts.align_command(align_args(batch=256)), 0)
            reading = self.read_records(directory)
            with unittest.mock.patch.object(tts, 'aligner_key', return_value='new'):
                self.assertEqual(tts.align_command(align_args()), 0)
            realigned = self.read_records(directory)
            latest = tts.read_alignments(directory)
        self.assertEqual(overrides.call_args.args, (directory,))
        self.assertEqual([record['id'] for record in first], [clip['id'] for clip in clips[:3]])
        by_id = {record['id']: record for record in first}
        self.assertEqual(by_id[clips[2]['id']], {'id': clips[2]['id'], 'aligner': tts.aligner_key(),
                                                 'keys': tts.phone_keys(frontend.tokenize(clips[2]['text'])['tokens']),
                                                 'tokens': frontend.tokenize(clips[2]['text'])['tokens'], 'seconds': 0.2,
                                                 'labels': None})
        record = by_id[clips[0]['id']]
        self.assertEqual(set(record), {'id', 'aligner', 'keys', 'tokens', 'seconds', 'labels', 'confidences', 'score', 'greedy'})
        self.assertEqual(record['tokens'], frontend.tokenize('Call me Ishmael.')['tokens'])
        self.assertEqual(record['keys'], 'K AO L M IY IH SH M IY L')
        self.assertEqual(len(record['confidences']), len(record['keys'].split()))
        self.assertEqual([label[0] for label in record['labels'] if label[0] != 'pau'],
                         [token for token in record['tokens'] if token not in frontend.BOUNDARIES])
        self.assertEqual(again, first)
        self.assertEqual(len(reading), 4)
        self.assertEqual(reading[-1]['id'], clips[1]['id'])
        self.assertIn('P AY P', reading[-1]['keys'])
        self.assertEqual(len(realigned), 7)
        self.assertEqual({record['aligner'] for record in realigned[4:]}, {'new'})
        self.assertEqual(latest[clips[1]['id']], realigned[5])
        output = stderr.getvalue()
        self.assertIn('tts align: 3 clips, 3 to align\n', output)
        self.assertIn('  2/3 clips, ', output)
        self.assertIn('tts align: aligned 2 of 3 clips (1 failed)', output)
        self.assertIn('tts align: 3 clips, 0 to align\n', output)
        self.assertIn('tts align: 3 clips, 1 to align\n', output)

    def test_read_alignments(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(tts.read_alignments(directory), {})
            with open(os.path.join(directory, 'alignments.jsonl'), 'w', encoding='utf-8') as file:
                file.write('{"id": "a", "n": 1}\n{"id": "b", "n": 2}\n{"id": "a", "n": 3}\n')
            self.assertEqual(tts.read_alignments(directory), {'a': {'id': 'a', 'n': 3}, 'b': {'id': 'b', 'n': 2}})

    def test_aligner_key(self):
        key = tts.aligner_key()
        self.assertEqual(len(key), 12)
        with unittest.mock.patch.object(ctc, 'CTC_SHIFT', 0.0):
            self.assertNotEqual(tts.aligner_key(), key)

    def test_clip_fronts_and_phone_keys(self):
        clip = {'id': 'a', 'text': 'Call me Ishmael.'}
        tokens = frontend.tokenize('Call me Ishmael.')['tokens']
        self.assertEqual(tts.clip_fronts([clip], {}, 2), {'a': {'tokens': tokens, 'words': ['call', 'me', 'ishmael']}})
        self.assertEqual(tts.clip_fronts([clip], {'b': {'0': 'P AY1'}}, 2)['a']['tokens'], tokens)
        read = tts.clip_fronts([clip], {'a': {'1': 'N IY1'}}, 2)['a']['tokens']
        self.assertEqual(read, frontend.tokenize('Call me Ishmael.', readings={1: ['N', 'IY1']})['tokens'])
        self.assertEqual(read[:6] + read[7:], tokens[:5] + ['N'] + tokens[7:])
        self.assertEqual(tts.phone_keys(tokens), 'K AO L M IY IH SH M IY L')


class TestBuild(unittest.TestCase):

    # The fake acoustic analysis, and multiprocessing.Pool in this process
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS))
        cls.enterClassContext(unittest.mock.patch('multiprocessing.Pool', data_fixtures.FakePool))

    def test_clip_checks(self):
        checks = tts_settings()['checks']
        text = 'Don\'t forget a jacket.'
        tokens = frontend.tokenize(text)['tokens']
        labels, _ = data_fixtures.text_labels(text)
        labels = [list(label) for label in labels]
        record = {'labels': labels + [['pau', 2.0, 2.1]], 'tokens': tokens, 'greedy': greedy_of(text)}
        self.assertEqual(tts.clip_checks(None, checks), (['no alignment'], None))
        self.assertEqual(tts.clip_checks(dict(record, labels=None), checks), (['no alignment'], None))
        self.assertEqual(tts.clip_checks(record, checks), ([], 0.0))
        # Three final phones missing - cut short when its trailing silence is short (or none - the last label a phone)
        short = dict(record, greedy=greedy_of(text, tail=3))
        self.assertEqual(tts.clip_checks(short, checks)[0], [])
        self.assertEqual(tts.clip_checks(dict(short, labels=labels + [['pau', 2.0, 2.01]]), checks)[0],
                         ['cut short (3 final phones missing)'])
        self.assertEqual(tts.clip_checks(dict(short, labels=labels), checks)[0], ['cut short (3 final phones missing)'])
        # Five at any silence - and garbled
        reasons, error = tts.clip_checks(dict(record, greedy=greedy_of(text, tail=5)), checks)
        self.assertEqual((reasons, error), (['cut short (5 final phones missing)'], 5 / 15))
        reasons, error = tts.clip_checks(dict(record, greedy='h h h'), checks)
        self.assertEqual(reasons[-1], 'garbled (greedy phone error 1.00 > 0.35)')
        self.assertEqual(error, 1.0)

    def test_pitch_range(self):
        self.assertEqual(tts.pitch_range(numpy.array([100.0, 120.0, 140.0, 160.0, 180.0])), (90.0, 240.0, 140.0))
        self.assertEqual(tts.pitch_range(numpy.array([40.0, 40.0, 40.0, 700.0, 700.0])), (50.0, 900.0, 40.0))

    def test_process_clip(self):
        labels, seconds = data_fixtures.text_labels('Call me.')
        rows = [['pau', 0.0, data_fixtures.EDGE_SECONDS]] + [list(label) for label in labels] + [['pau', labels[-1][2], seconds]]
        tokens = frontend.tokenize('Call me.')['tokens']
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            # A 16 kHz clip - resampled to 24 kHz; its edge silences (0.25 s) cut to MAX_EDGE_SILENCE
            write_wav(path, data_fixtures.labels_audio(labels, seconds, 16000), 16000)
            result = tts.process_clip((path, tokens, rows, {}, (75.0, 300.0, 5000)))
        self.assertAlmostEqual(result['cutStart'], data_fixtures.EDGE_SECONDS - data.MAX_EDGE_SILENCE, places=3)
        self.assertAlmostEqual(result['seconds'][0], data.MAX_EDGE_SILENCE, places=3)
        self.assertEqual(len(result['frames']), len(tokens))
        self.assertEqual(len(result['audio']), sum(result['frames']) * data.HOP)
        self.assertEqual(result['audio'].dtype, numpy.int16)
        self.assertEqual(result['tracks'].shape, (sum(result['frames']), len(data.CHANNELS)))
        self.assertEqual(result['clipped'], 0.0)

    # The build, with a phone of the last clip given no frames
    def run_build(self, args):
        real = tts.process_clip

        def process_clip(task):
            result = real(task)
            if task[0].endswith(tts.clip_id('Harper', TEXTS[7][1]) + '.wav'):
                index = next(index for index, token in enumerate(task[1]) if token not in frontend.BOUNDARIES)
                result['frames'][index + 1] += result['frames'][index]
                result['frames'][index] = 0
            return result
        with unittest.mock.patch.object(tts, 'process_clip', process_clip), helpers.captured() as (stdout, stderr):
            self.assertEqual(tts.build_command(args), 0)
        return stdout.getvalue(), stderr.getvalue()

    # The test corpus - a garbled clip, one cut short, one not aligned, one not generated, one clipped; a voice with no
    # clips
    @contextlib.contextmanager
    def corpus(self):
        settings = tts_settings(voices={**tts_settings()['voices'], 'extra': {'voice': 'Nova', 'sex': 'female'}})
        with tts_corpus(TEXTS, settings) as (directory, settings, clips), \
                unittest.mock.patch.object(readings, 'overrides', return_value={clips[0]['id']: {}}):
            edit_alignments(directory, {clips[2]['id']: {'greedy': 'h h h h'},
                                        clips[3]['id']: {'greedy': greedy_of(clips[3]['text'], tail=5)},
                                        clips[4]['id']: {'labels': None}})
            os.remove(tts.wav_path(directory, clips[5]))
            labels, seconds = data_fixtures.text_labels(clips[6]['text'])
            samples = data_fixtures.labels_audio(labels, seconds, 24000)
            write_wav(tts.wav_path(directory, clips[6]), numpy.clip(samples * 20, -1, 1), 24000)
            yield directory, settings, clips

    def test_build_command(self):
        with self.corpus() as (_, settings, clips), tempfile.TemporaryDirectory() as out:
            with unittest.mock.patch.object(tts, 'BUILD_PROGRESS', 3):
                stdout, stderr = self.run_build(build_args(out))
            corpus = data.CorpusData(out)
            arrays = [corpus.arrays(entry) for entry in corpus.index['utterances']]
            with open(os.path.join(out, 'alignment.jsonl'), encoding='utf-8') as file:
                alignment = [json.loads(line) for line in file]
            raw = [name for name in os.listdir(out) if name.endswith('.raw')]
        index = corpus.index
        entries = {entry['id']: entry for entry in index['utterances']}
        ids = [f'{clip["key"]}/{clip["id"]}' for clip in clips]
        self.assertEqual(list(entries), [ids[0], ids[1], ids[2], ids[3], ids[6], ids[7]])
        self.assertEqual({entry_id: entry['reason'] for entry_id, entry in entries.items()}, {
            ids[0]: '', ids[1]: '',
            ids[2]: entries[ids[2]]['reason'],
            ids[3]: 'cut short (5 final phones missing)',
            ids[6]: entries[ids[6]]['reason'],
            ids[7]: 'a phone with no frames'
        })
        self.assertRegex(entries[ids[2]]['reason'],
                         r'^cut short \(\d+ final phones missing\); garbled \(greedy phone error 1\.00 > 0\.35\)$')
        self.assertRegex(entries[ids[6]]['reason'], r'^clipped \(\d+\.\d\d% of samples at full scale\)$')
        self.assertEqual([entry['include'] for entry in entries.values()], [True, True, False, False, False, False])
        first = entries[ids[0]]
        self.assertEqual({key: first[key] for key in ('speaker', 'sentence', 'split', 'text', 'book', 'tokenOffset', 'frameOffset',
                                                      'phoneError', 'clipped', 'readings')},
                         {'speaker': 'male', 'sentence': clips[0]['id'], 'split': 'train', 'text': 'Call me Ishmael.',
                          'book': 'pg1.txt', 'tokenOffset': 0, 'frameOffset': 0, 'phoneError': 0.0, 'clipped': 0.0, 'readings': {}})
        self.assertEqual(entries[ids[1]]['tokenOffset'], first['tokenCount'])
        self.assertEqual(entries[ids[1]]['frameOffset'], first['frameCount'])
        for entry, entry_arrays in zip(index['utterances'], arrays):
            self.assertEqual(entry_arrays['tokens'].tolist(), frontend.tokenize(entry['text'])['ids'])
            self.assertEqual(int(entry_arrays['durations'].sum()), entry['frameCount'])
            self.assertEqual(len(entry_arrays['audio']), entry['sampleCount'])
            self.assertEqual(entry_arrays['frames'].shape, (entry['frameCount'], len(data.CHANNELS)))
        self.assertEqual(raw, [])
        self.assertEqual([row['id'] for row in alignment], list(entries))
        self.assertEqual(alignment[0]['tokens'], ' '.join(frontend.tokenize('Call me Ishmael.')['tokens']))
        self.assertEqual(alignment[0]['greedy'], greedy_of('Call me Ishmael.'))

        # The index - the voices with clips, keyed by voice key, their rows the recipe's order
        speakers = index['speakers']
        self.assertEqual(list(speakers), ['male', 'female'])
        self.assertEqual({key: speakers['female'][key] for key in ('index', 'sex', 'corpus', 'engine', 'sourceSampleRate', 'bandwidth',
                                                                     'formantCeiling')},
                         {'index': 1, 'sex': 'female', 'corpus': 'tts',
                          'engine': {'model': 'phonon', 'digest': '0123456789ab', 'voice': 'Harper'}, 'sourceSampleRate': 24000,
                          'bandwidth': 12000, 'formantCeiling': 5500})
        self.assertTrue(speakers['male']['pitchFloor'] < speakers['male']['medianF0'] < speakers['male']['pitchCeiling'])
        self.assertAlmostEqual(speakers['male']['medianF0'], 150.0, delta=5.0)
        self.assertEqual((index['corpus'], index['license'], index['version']),
                         ('tts (phonon 0123456789ab)', settings['engine']['license'], data.FORMAT_VERSION))
        self.assertEqual(index['vocab'], frontend.vocabulary())
        self.assertEqual(index['alignment']['aligner'], tts.aligner_key())
        self.assertEqual(index['checks'], settings['checks'])
        self.assertIn(f'Built {out}: 6 clips (2 included), ', stdout)
        self.assertIn('  male (Trevor): F0 ', stdout)
        self.assertIn('1 included, excluded {\'clipped\': 1, \'cut short\': 1, \'garbled\': 1}', stdout)
        self.assertIn('1 included, excluded {\'a phone with no frames\': 1, \'cut short\': 1}', stdout)
        self.assertIn('  3/6 clips, ', stderr)
        self.assertIn('  6/6 clips, ', stderr)

    def test_build_limit(self):
        with self.corpus() as (_, _, clips), tempfile.TemporaryDirectory() as out:
            self.run_build(build_args(os.path.join(out, 'data'), limit=3))
            index = data.CorpusData(os.path.join(out, 'data')).index
        # Every other of the 7 generated clips, the first 3 - the third not aligned
        self.assertEqual([entry['sentence'] for entry in index['utterances']], [clips[0]['id'], clips[2]['id']])

    def test_build_stale_alignment(self):
        with self.corpus() as (_, _, clips), tempfile.TemporaryDirectory() as out, \
                unittest.mock.patch.object(readings, 'overrides', return_value={clips[1]['id']: {'1': 'P AY1 P'}}), \
                self.assertRaisesRegex(ValueError, f'^{clips[1]["id"]}: the front end\'s phones differ from its alignment\'s'):
            self.run_build(build_args(out))


class TestMain(unittest.TestCase):

    def test_main(self):
        handlers = {name: unittest.mock.patch.object(tts, name, return_value=0) for name in
                    ('generate_command', 'align_command', 'build_command')}
        with contextlib.ExitStack() as stack:
            mocks = {name: stack.enter_context(patch) for name, patch in handlers.items()}
            self.assertEqual(tts.main(['generate', '-j', '2']), 0)
            self.assertEqual(tts.main(['--recipe', 'r.json', 'align', '--batch', '8', '--device', 'cpu']), 0)
            self.assertEqual(tts.main(['build', '--out', 'x', '--limit', '5', '-j', '3']), 0)
        self.assertEqual(mocks['generate_command'].call_args.args[0],
                         argparse.Namespace(recipe=tts.RECIPE_PATH, command='generate', jobs=2, func=mocks['generate_command']))
        self.assertEqual(mocks['align_command'].call_args.args[0],
                         argparse.Namespace(recipe='r.json', command='align', batch=8, device='cpu', jobs=None,
                                            func=mocks['align_command']))
        self.assertEqual(mocks['build_command'].call_args.args[0],
                         argparse.Namespace(recipe=tts.RECIPE_PATH, command='build', out='x', limit=5, jobs=3,
                                            func=mocks['build_command']))

    def test_main_engine_error(self):
        with unittest.mock.patch.object(tts, 'generate_command', side_effect=tts.EngineError('no engine')), \
                helpers.captured() as (_, stderr):
            self.assertEqual(tts.main(['generate']), 1)
        self.assertEqual(stderr.getvalue(), 'tts: no engine\n')

    def test_module_main(self):
        with unittest.mock.patch.object(sys, 'argv', ['tts']), self.assertRaises(SystemExit), helpers.captured(), \
                warnings.catch_warnings():
            # (runpy warns that the module is already imported - it runs a fresh copy as __main__)
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.tts', run_name='__main__', alter_sys=False)
