# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import base64
import datetime
import hashlib
import io
import json
import lzma
import os
import runpy
import shutil
import sys
import tarfile
import tempfile
import unittest
import unittest.mock
import urllib.error

from picovox_train import ROOT, books

from .helpers import captured


# The texts of two revisions of a book
PINNED = b'The Project Gutenberg eBook\n                Most recently updated: January 10, 2026\n*** START OF THE BOOK ***\nI am.\n'
REVISED = b'The Project Gutenberg eBook\n                Most recently updated: October 16, 2026\n*** START OF THE BOOK ***\nI\'m.\n'


# A pin of the pinned text
def pin_of(data, archive='https://web.archive.org/web/20260709214932id_/https://www.gutenberg.org/cache/epub/7/pg7.txt'):
    return {'id': 7, 'url': 'https://www.gutenberg.org/cache/epub/7/pg7.txt', 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'pinned': '2026-10-15', 'updated': 'January 10, 2026', 'archive': archive}


# A urlopen that serves the URLs' bytes (an exception instance is raised), recording the requests
class FakeUrls:

    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def __call__(self, request, timeout):
        assert timeout == 60
        self.requests.append(request.full_url)
        response = self.responses[request.full_url]
        if isinstance(response, Exception):
            raise response
        opened = unittest.mock.MagicMock()
        opened.__enter__.return_value.read.return_value = response
        return opened


class BooksCase(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.books = os.path.join(self.directory, 'books')
        self.pins_path = os.path.join(self.directory, 'books.json')
        self.write_pins({'pg7': pin_of(PINNED)})

    def tearDown(self):
        shutil.rmtree(self.directory)

    def write_pins(self, texts):
        with open(self.pins_path, 'w', encoding='utf-8') as file:
            json.dump({'description': 'The pins', 'texts': texts}, file)

    def write_book(self, name, data):
        os.makedirs(self.books, exist_ok=True)
        with open(os.path.join(self.books, f'{name}.txt'), 'wb') as file:
            file.write(data)

    def read_book(self, name, suffix='.txt'):
        with open(os.path.join(self.books, name + suffix), 'rb') as file:
            return file.read()


class TestPins(BooksCase):

    def test_recipe_pins(self):
        # The recipe's books are pinned - each a Gutenberg URL, a SHA-256, and a date; the labels' books are pg<id> (their
        # names in the labels)
        with open(os.path.join(ROOT, 'data', 'nn', 'train.json'), encoding='utf-8') as file:
            recipe = json.load(file)
        pins = books.load_pins()
        labels, targets = recipe['context']['labels'], recipe['distill']['targets']
        for settings in (labels, targets):
            self.assertEqual(settings['pins'], 'data/nn/books.json')
            self.assertTrue(set(settings['texts']) <= set(pins))
        self.assertEqual(len(labels['texts']), 132)
        self.assertEqual(len(targets['texts']), 22)
        self.assertTrue(set(labels['heldOutBooks']) <= set(labels['texts']))
        self.assertTrue(all(pins[name]['id'] == int(name[2:]) for name in labels['texts']))
        self.assertEqual(set(pins), set(labels['texts']) | set(targets['texts']))
        for name, pin in pins.items():
            self.assertEqual(pin['url'], f'https://www.gutenberg.org/cache/epub/{pin["id"]}/pg{pin["id"]}.txt', name)
            self.assertRegex(pin['sha256'], r'^[0-9a-f]{64}$', name)
            self.assertRegex(pin['pinned'], r'^\d{4}-\d\d-\d\d$', name)
            self.assertTrue(pin['archive'] is None or pin['archive'].startswith('https://web.archive.org/web/'), name)
        # (the distillation's pg2554 is a later revision than the labels')
        self.assertNotEqual(pins['pg2554']['sha256'], pins['pg2554-2026-10-16']['sha256'])

        # The file is as write_pins writes it
        with open(books.PINS_PATH, encoding='utf-8') as file:
            text = file.read()
        with open(self.pins_path, 'w', encoding='utf-8') as file:
            file.write(text)
        books.write_pins(books.load_pins(self.pins_path), self.pins_path)
        with open(self.pins_path, encoding='utf-8') as file:
            self.assertEqual(file.read(), text)

    def test_write_pins(self):
        books.write_pins({'pg10': {'id': 10}, 'pg9-2026-10-16': {'id': 9}, 'pg9': {'id': 9}, 'other': {'id': 1}}, self.pins_path)
        with open(self.pins_path, encoding='utf-8') as file:
            self.assertEqual(file.read(), '{\n    "description": "The pins",\n    "texts": {\n'
                                          '        "pg9": {"id": 9},\n        "pg9-2026-10-16": {"id": 9},\n'
                                          '        "pg10": {"id": 10},\n        "other": {"id": 1}\n    }\n}\n')
        self.assertFalse(os.path.exists(self.pins_path + '.tmp'))

    def test_display_path(self):
        self.assertEqual(books.display_path(os.path.join(ROOT, 'build', 'x.txt')), os.path.join('build', 'x.txt'))
        self.assertEqual(books.display_path('/elsewhere/x.txt'), '/elsewhere/x.txt')

    def test_updated_date(self):
        self.assertEqual(books.updated_date(PINNED), 'January 10, 2026')
        self.assertIsNone(books.updated_date(b'No header'))

    def test_fetch_url(self):
        urls = FakeUrls({'https://example.com/a': b'bytes'})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            self.assertEqual(books.fetch_url('https://example.com/a'), b'bytes')


class TestPinnedPath(BooksCase):

    def test_cached(self):
        # A cached pinned text is verified, not fetched
        self.write_book('pg7', PINNED)
        urls = FakeUrls({})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            self.assertEqual(books.pinned_paths(self.books, ['pg7'], self.pins_path), [os.path.join(self.books, 'pg7.txt')])
        self.assertEqual(urls.requests, [])

    def test_cached_mismatch(self):
        # A cached copy of another revision fails - the book, both hashes, and how to proceed
        self.write_book('pg7', REVISED)
        with self.assertRaises(books.PinError) as raised:
            books.pinned_paths(self.books, ['pg7'], self.pins_path)
        message = str(raised.exception)
        self.assertTrue(message.startswith(f'book pg7 (Project Gutenberg #7): the copy {self.books}/pg7.txt has sha256 '
                                           f'{hashlib.sha256(REVISED).hexdigest()}, not the pinned sha256 '
                                           f'{hashlib.sha256(PINNED).hexdigest()} ({len(PINNED)} bytes, pinned 2026-10-15).'))
        self.assertIn('Remove it to fetch the pinned text. To proceed, either restore the pinned text', message)
        self.assertIn('python -m picovox_train.books pin pg7', message)
        self.assertEqual(self.read_book('pg7'), REVISED)

    def test_fetch(self):
        # A missing text is fetched from its URL
        urls = FakeUrls({'https://www.gutenberg.org/cache/epub/7/pg7.txt': PINNED})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            path = books.pinned_path(self.books, 'pg7', books.load_pins(self.pins_path))
        self.assertEqual(path, os.path.join(self.books, 'pg7.txt'))
        self.assertEqual(self.read_book('pg7'), PINNED)
        self.assertEqual(os.listdir(self.books), ['pg7.txt'])

    def test_fetch_archive(self):
        # Gutenberg revised the book (or is unreachable): the archived copy of the pinned text
        pin = pin_of(PINNED)
        for gutenberg in (REVISED, urllib.error.URLError('down')):
            shutil.rmtree(self.books, ignore_errors=True)
            urls = FakeUrls({pin['url']: gutenberg, pin['archive']: PINNED})
            with unittest.mock.patch('urllib.request.urlopen', urls):
                books.pinned_path(self.books, 'pg7', {'pg7': pin})
            self.assertEqual(urls.requests, [pin['url'], pin['archive']])
            self.assertEqual(self.read_book('pg7'), PINNED)
            self.assertEqual(os.listdir(self.books), ['pg7.txt'])

    def test_fetch_mismatch(self):
        # No source has the pinned text: fails with the hashes, the first text saved for comparison, and how to proceed
        pin = pin_of(PINNED)
        urls = FakeUrls({pin['url']: REVISED, pin['archive']: OSError('timed out')})
        with unittest.mock.patch('urllib.request.urlopen', urls), self.assertRaises(books.PinError) as raised:
            books.pinned_path(self.books, 'pg7', {'pg7': pin})
        self.assertEqual(str(raised.exception).split('\nProject Gutenberg revises', maxsplit=1)[0],
                         f'book pg7 (Project Gutenberg #7): no source has the pinned text, sha256 {pin["sha256"]} '
                         f'({len(PINNED)} bytes, pinned 2026-10-15):\n'
                         f'  {pin["url"]}: sha256 {hashlib.sha256(REVISED).hexdigest()} ({len(REVISED)} bytes)\n'
                         f'  {pin["archive"]}: timed out\n'
                         f'  (the first of those texts is saved as {self.books}/pg7.txt.unpinned)')
        self.assertIn('To proceed, either restore the pinned text', str(raised.exception))
        self.assertEqual(self.read_book('pg7', '.txt.unpinned'), REVISED)
        self.assertFalse(os.path.exists(os.path.join(self.books, 'pg7.txt')))

        # Both sources revised: the first text is kept for comparison
        os.remove(os.path.join(self.books, 'pg7.txt.unpinned'))
        urls = FakeUrls({pin['url']: REVISED, pin['archive']: b'Another'})
        with unittest.mock.patch('urllib.request.urlopen', urls), self.assertRaises(books.PinError) as raised:
            books.pinned_path(self.books, 'pg7', {'pg7': pin})
        self.assertIn(f'  {pin["archive"]}: sha256 {hashlib.sha256(b"Another").hexdigest()} (7 bytes)\n', str(raised.exception))
        self.assertEqual(self.read_book('pg7', '.txt.unpinned'), REVISED)

        # No archived copy, and Gutenberg unreachable
        pin = pin_of(PINNED, archive=None)
        urls = FakeUrls({pin['url']: urllib.error.URLError('down')})
        with unittest.mock.patch('urllib.request.urlopen', urls), self.assertRaises(books.PinError) as raised:
            books.pinned_path(os.path.join(self.directory, 'other'), 'pg7', {'pg7': pin})
        self.assertIn(f'  {pin["url"]}: <urlopen error down>\n  (no archived copy of the pinned text is known)\n', str(raised.exception))

    def test_not_pinned(self):
        with self.assertRaises(books.PinError) as raised:
            books.pinned_path(self.books, 'pg8', {})
        self.assertEqual(str(raised.exception), f'book pg8: not pinned in data/nn/books.json - fetch it to {self.books}/pg8.txt and pin it '
                                                '(python -m picovox_train.books pin pg8)')


class TestArchive(BooksCase):

    def test_archived_copy(self):
        digest = base64.b32encode(hashlib.sha1(PINNED).digest()).decode('ascii')
        header = ['urlkey', 'timestamp', 'original', 'mimetype', 'statuscode', 'digest', 'length']
        found = [header, ['org,gutenberg)/cache/epub/7/pg7.txt', '20260709214932', 'https://gutenberg.org/cache/epub/7/pg7.txt',
                          'text/plain', '200', digest, '100']]
        cdx = 'https://web.archive.org/cdx/search/cdx?url={}&output=json&filter=digest:' + digest + '&filter=statuscode:200'
        www, bare = cdx.format('www.gutenberg.org/cache/epub/7/pg7.txt'), cdx.format('gutenberg.org/cache/epub/7/pg7.txt')
        url = 'https://www.gutenberg.org/cache/epub/7/pg7.txt'
        urls = FakeUrls({www: b'', bare: json.dumps(found).encode()})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            self.assertEqual(books.archived_copy(url, PINNED),
                             'https://web.archive.org/web/20260709214932id_/https://gutenberg.org/cache/epub/7/pg7.txt')
        self.assertEqual(urls.requests, [www, bare])
        urls = FakeUrls({www: json.dumps(found[:1]).encode(), bare: b'[]'})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            self.assertIsNone(books.archived_copy(url, PINNED))


class TestCommands(BooksCase):

    def test_fetch(self):
        self.write_book('pg7', PINNED)
        with captured() as (stdout, _):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'fetch']), 0)
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'fetch', 'pg7']), 0)
        self.assertEqual(stdout.getvalue(), f'books: 1 pinned texts in {self.books}\n' * 2)

        # A text that isn't the pinned one fails the command
        self.write_book('pg7', REVISED)
        with captured() as (stdout, stderr):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'fetch']), 1)
        self.assertEqual(stdout.getvalue(), '')
        self.assertTrue(stderr.getvalue().startswith('books: book pg7 (Project Gutenberg #7): the copy '))

    def test_pin(self):
        # Re-pin a text to the books directory's copy - its hash, the date, and its archived copy
        self.write_book('pg7', REVISED)
        archive = 'https://web.archive.org/web/20261017000000id_/https://www.gutenberg.org/cache/epub/7/pg7.txt'
        with unittest.mock.patch.object(books, 'archived_copy', return_value=archive) as archived, captured() as (stdout, _):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'pin', 'pg7']), 0)
        archived.assert_called_once_with('https://www.gutenberg.org/cache/epub/7/pg7.txt', REVISED)
        digest = hashlib.sha256(REVISED).hexdigest()
        self.assertEqual(stdout.getvalue(), f'books: pinned pg7 - sha256 {digest} ({len(REVISED)} bytes), archive {archive} '
                                            f'(was sha256 {hashlib.sha256(PINNED).hexdigest()})\n')
        self.assertEqual(books.load_pins(self.pins_path), {'pg7': {
            'id': 7, 'url': 'https://www.gutenberg.org/cache/epub/7/pg7.txt', 'sha256': digest, 'bytes': len(REVISED),
            'pinned': datetime.date.today().isoformat(), 'updated': 'October 16, 2026', 'archive': archive}})

        # A new text (its id from its name, or --id), a given date, and no archived copy (the archive unreachable)
        self.write_book('pg8-2026-10-17', PINNED)
        self.write_book('other', b'other')
        with unittest.mock.patch.object(books, 'archived_copy', side_effect=urllib.error.URLError('down')), \
                captured() as (stdout, stderr):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'pin', 'pg8-2026-10-17',
                                         '--date', '2026-10-17']), 0)
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'pin', 'other', '--id', '9']), 0)
        self.assertEqual(stderr.getvalue(), 'books: no archived copy of pg8-2026-10-17 (<urlopen error down>)\n'
                                            'books: no archived copy of other (<urlopen error down>)\n')
        pins = books.load_pins(self.pins_path)
        self.assertEqual((pins['pg8-2026-10-17']['id'], pins['pg8-2026-10-17']['pinned'], pins['pg8-2026-10-17']['archive']),
                         (8, '2026-10-17', None))
        self.assertEqual((pins['other']['id'], pins['other']['updated']), (9, None))

    def test_main_module(self):
        urls = FakeUrls({'https://www.gutenberg.org/cache/epub/7/pg7.txt': PINNED})
        with unittest.mock.patch.dict(sys.modules), \
                unittest.mock.patch.object(sys, 'argv', ['books', '--pins', self.pins_path, '--books', self.books, 'fetch']), \
                unittest.mock.patch('urllib.request.urlopen', urls), captured() as (stdout, _):
            sys.modules.pop('picovox_train.books')
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.books', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue(), f'books: 1 pinned texts in {self.books}\n')


# An xz-compressed tar of texts ({name: bytes} - name.txt, or a file name with its extension) - a hand-made bundle
def archive_of(texts):
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode='w') as archive:
        for name, data in texts.items():
            info = tarfile.TarInfo(name if '.' in name else f'{name}.txt')
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return lzma.compress(tar_bytes.getvalue())


# The bundle of the pins' texts at a URL - its pin, built by the bundle command from a books directory with the texts
class TestBundle(BooksCase):

    URL = 'https://craigahobbs.github.io/picovox/books/books-v1.tar.xz'

    def build(self, texts=None):
        # Bundle the texts (default: the pinned pg7) from a separate books directory, returning its bytes
        source = os.path.join(self.directory, 'source')
        os.makedirs(source, exist_ok=True)
        for name, data in (texts or {'pg7': PINNED}).items():
            with open(os.path.join(source, f'{name}.txt'), 'wb') as file:
                file.write(data)
        with captured():
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', source, 'bundle', 'books-v1']), 0)
        with open(os.path.join(source, 'books-v1.tar.xz'), 'rb') as file:
            return file.read()

    def test_bundle_command(self):
        # The bundle is deterministic, and recorded in the pins (which keep it when rewritten)
        data = self.build()
        self.assertEqual(self.build(), data)
        bundle = books.load_bundle(self.pins_path)
        self.assertEqual(bundle, {'tag': 'books-v1', 'url': self.URL, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        books.write_pins(books.load_pins(self.pins_path), self.pins_path)
        self.assertEqual(books.load_bundle(self.pins_path), bundle)
        with captured() as (stdout, _):
            books.main(['--pins', self.pins_path, '--books', os.path.join(self.directory, 'source'), 'bundle', 'books-v1'])
        self.assertTrue(stdout.getvalue().startswith('books: bundled 1 pinned texts as '))
        self.assertIn(f'publish it with the site (make gh-pages - {self.URL})', stdout.getvalue())

        # A pinned text that isn't in the books directory fails the command
        with captured() as (stdout, stderr), unittest.mock.patch('urllib.request.urlopen', FakeUrls({
                'https://www.gutenberg.org/cache/epub/7/pg7.txt': REVISED,
                pin_of(PINNED)['archive']: REVISED})):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'bundle', 'books-v1']), 1)
        self.assertTrue(stderr.getvalue().startswith('books: book pg7 (Project Gutenberg #7): no source has the pinned text'))

    def test_fetch_from_bundle(self):
        # A missing text comes from the bundle first - downloaded once, verified, kept in the books directory
        data = self.build()
        urls = FakeUrls({self.URL: data})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            self.assertEqual(books.pinned_paths(self.books, ['pg7'], self.pins_path), [os.path.join(self.books, 'pg7.txt')])
            os.remove(os.path.join(self.books, 'pg7.txt'))
            books.pinned_paths(self.books, ['pg7'], self.pins_path)
        self.assertEqual(urls.requests, [self.URL])
        self.assertEqual(self.read_book('pg7'), PINNED)
        self.assertEqual(sorted(os.listdir(self.books)), ['books-v1.tar.xz', 'pg7.txt'])

        # A corrupt cached bundle is downloaded again
        with open(os.path.join(self.books, 'books-v1.tar.xz'), 'wb') as file:
            file.write(b'corrupt')
        os.remove(os.path.join(self.books, 'pg7.txt'))
        urls = FakeUrls({self.URL: data})
        with unittest.mock.patch('urllib.request.urlopen', urls):
            books.pinned_paths(self.books, ['pg7'], self.pins_path)
        self.assertEqual(urls.requests, [self.URL])
        self.assertEqual(self.read_book('pg7'), PINNED)

    def test_bundle_fallbacks(self):
        # The bundle fails - not the pinned bundle, unreachable, not an archive, without the text, another revision -
        # the text's URL is next
        self.build()
        url = 'https://www.gutenberg.org/cache/epub/7/pg7.txt'
        for response in (b'tampered', urllib.error.URLError('down')):
            shutil.rmtree(self.books, ignore_errors=True)
            urls = FakeUrls({self.URL: response, url: PINNED})
            with unittest.mock.patch('urllib.request.urlopen', urls):
                books.pinned_paths(self.books, ['pg7'], self.pins_path)
            self.assertEqual(urls.requests, [self.URL, url])
            self.assertEqual(self.read_book('pg7'), PINNED)

        # An archive without the text, or with another revision of it, or not an archive at all (its pin matching)
        for bundle_bytes in (archive_of({'pg8': PINNED}), archive_of({'pg7': REVISED}), b'not an archive'):
            with open(self.pins_path, encoding='utf-8') as file:
                pins = json.load(file)
            pins['bundle']['sha256'] = hashlib.sha256(bundle_bytes).hexdigest()
            pins['texts'] = {'pg7': pin_of(PINNED, archive=None)}
            with open(self.pins_path, 'w', encoding='utf-8') as file:
                json.dump(pins, file)
            shutil.rmtree(self.books, ignore_errors=True)
            urls = FakeUrls({self.URL: bundle_bytes, url: urllib.error.URLError('down')})
            with unittest.mock.patch('urllib.request.urlopen', urls), self.assertRaises(books.PinError) as raised:
                books.pinned_paths(self.books, ['pg7'], self.pins_path)
            message = str(raised.exception)
            self.assertIn(f'  {self.URL}: ', message)
            self.assertIn(f'  {url}: <urlopen error down>', message)

    def test_publish_command(self):
        # The bundle into the site's books/ - from the books directory, else downloaded
        data = self.build()
        site = os.path.join(self.directory, 'site')
        urls = FakeUrls({self.URL: data})
        with unittest.mock.patch('urllib.request.urlopen', urls), captured() as (stdout, _):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'publish', site]), 0)
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'publish', site]), 0)
        self.assertEqual(urls.requests, [self.URL])
        self.assertEqual(stdout.getvalue(), f'books: published books-v1.tar.xz in {site}/books\n' * 2)
        with open(os.path.join(site, 'books', 'books-v1.tar.xz'), 'rb') as file:
            self.assertEqual(file.read(), data)

        # Not the pinned bundle, or no bundle
        shutil.rmtree(self.books)
        with unittest.mock.patch('urllib.request.urlopen', FakeUrls({self.URL: b'tampered'})), captured() as (_, stderr):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'publish', site]), 1)
        self.assertTrue(stderr.getvalue().startswith(f'books: the bundle {self.URL}: sha256 '))
        self.write_pins({'pg7': pin_of(PINNED)})
        with captured() as (_, stderr):
            self.assertEqual(books.main(['--pins', self.pins_path, '--books', self.books, 'publish', site]), 1)
        self.assertEqual(stderr.getvalue(), f'books: no bundle in {self.pins_path} (python -m picovox_train.books bundle TAG)\n')

    def test_unpack_bundle(self):
        # One pass over the bundle: the pinned texts the directory lacks, verified - not other files, unpinned texts, texts
        # the directory has, or other revisions
        pins = {'pg7': pin_of(PINNED), 'pg8': pin_of(PINNED), 'pg9': pin_of(PINNED)}
        bundle = os.path.join(self.directory, 'bundle.tar.xz')
        with open(bundle, 'wb') as file:
            file.write(archive_of({'pg7': PINNED, 'pg8': PINNED, 'pg9': REVISED, 'pg10': PINNED, 'README.md': b'x'}))
        self.write_book('pg8', b'kept')
        books.unpack_bundle(self.books, bundle, pins)
        self.assertEqual(sorted(os.listdir(self.books)), ['pg7.txt', 'pg8.txt'])
        self.assertEqual(self.read_book('pg7'), PINNED)
        self.assertEqual(self.read_book('pg8'), b'kept')
