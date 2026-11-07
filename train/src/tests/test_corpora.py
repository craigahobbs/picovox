# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import hashlib
import logging
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
import urllib.error

from picovox_train import corpora
from tests import data_fixtures, helpers
from tests.data_fixtures import FakeResponse


class TestDownload(unittest.TestCase):

    def test_hf_snapshot(self):
        def snapshot_download(repo, local_files_only=False, **options):
            if local_files_only and not cached:
                raise OSError('not cached')
            return f'/snapshots/{repo}{options}'
        hub = types.ModuleType('huggingface_hub')
        hub.snapshot_download = unittest.mock.Mock(side_effect=snapshot_download)
        hub_logger = logging.getLogger('huggingface_hub.utils._http')
        with data_fixtures.fake_modules(huggingface_hub=hub), unittest.mock.patch.object(hub_logger, 'filters', []):
            # From the cache, without network access, when it's there
            cached = True
            self.assertEqual(corpora.hf_snapshot('a/b', allow_patterns=['*.json']), "/snapshots/a/b{'allow_patterns': ['*.json']}")
            self.assertEqual(hub.snapshot_download.call_args_list,
                             [unittest.mock.call('a/b', local_files_only=True, allow_patterns=['*.json'])])
            self.assertEqual(hub_logger.filters, [])
            # Else downloaded - the Hub's unauthenticated-requests note filtered out, its other messages kept
            cached = False
            self.assertEqual(corpora.hf_snapshot('a/b'), '/snapshots/a/b{}')
            self.assertEqual(hub.snapshot_download.call_args_list[1:], [unittest.mock.call('a/b', local_files_only=True),
                                                                        unittest.mock.call('a/b')])

            def record(message):
                return logging.LogRecord(hub_logger.name, logging.WARNING, __file__, 0, message, None, None)
            self.assertFalse(hub_logger.filter(record('Warning: You are sending unauthenticated requests to the HF Hub.')))
            self.assertTrue(hub_logger.filter(record('Another note')))

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'file')
            with open(path, 'wb') as file:
                file.write(b'hello')
            self.assertEqual(corpora.file_sha256(path), hashlib.sha256(b'hello').hexdigest())

    def test_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.tar')
            with unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'abc')) as urlopen:
                corpora.download('https://x/a.tar', path)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header('User-agent'), corpora.USER_AGENT)
            self.assertEqual(sorted(os.listdir(directory)), ['a.tar'])
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'abc')

            # An error - nothing downloaded
            error = urllib.error.HTTPError('https://x/b.tar', 500, 'Server Error', {}, None)
            with unittest.mock.patch('urllib.request.urlopen', side_effect=error), self.assertRaises(urllib.error.HTTPError):
                corpora.download('https://x/b.tar', os.path.join(directory, 'b.tar'))
            error.close()
            self.assertEqual(sorted(os.listdir(directory)), ['a.tar'])



class TestFetch(unittest.TestCase):

    def test_fetch_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'files', 'model.bin')
            urls = []

            def urlopen(request, timeout):
                urls.append((request.full_url, timeout))
                return FakeResponse(b'model')
            files = {'model': {'url': 'https://x/model.bin', 'path': path, 'sha256': hashlib.sha256(b'model').hexdigest(),
                               'description': 'A model'},
                     'revised': {'url': 'https://x/revised.bin', 'path': os.path.join(directory, 'revised.bin'), 'sha256': '0' * 64,
                                 'description': 'A revised file'}}
            with unittest.mock.patch.dict(corpora.FILES, files, clear=True), \
                    unittest.mock.patch('urllib.request.urlopen', side_effect=urlopen), helpers.captured() as (_, stderr):
                self.assertEqual(corpora.fetch_file('model'), 5)
                self.assertEqual(corpora.fetch_file('model'), 0)
                # A file other than the pinned one fails, and is removed
                with self.assertRaisesRegex(IOError, 'revised: SHA-256 .* is not the pinned 0000'):
                    corpora.fetch_file('revised')
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'model')
            self.assertFalse(os.path.exists(os.path.join(directory, 'revised.bin')))
        self.assertEqual(urls, [('https://x/model.bin', 60), ('https://x/revised.bin', 60)])
        self.assertTrue(stderr.getvalue().startswith('model: downloading https://x/model.bin\n'))
        self.assertIn('model: present\n', stderr.getvalue())

class TestDictionary(unittest.TestCase):

    def test_parse_dictionary(self):
        lines = [';;; a comment', '', 'read R IY1 D', 'read(2) R EH1 D # past', 'read(3) R IY1 D', "o'clock AH0 K L AA1 K"]
        self.assertEqual(corpora.parse_dictionary(lines), {
            'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D'], ['R', 'IY1', 'D']], "o'clock": [['AH0', 'K', 'L', 'AA1', 'K']]
        })

    def test_read_dictionary(self):
        opened = unittest.mock.mock_open(read_data='cat K AE1 T\ncats K AE1 T S\n')
        with unittest.mock.patch('builtins.open', opened):
            self.assertEqual(corpora.read_dictionary(), {'cat': [['K', 'AE1', 'T']], 'cats': [['K', 'AE1', 'T', 'S']]})
        opened.assert_called_once_with(corpora.CMUDICT_PATH, encoding='utf-8')


class TestMain(unittest.TestCase):

    def test_main(self):
        with unittest.mock.patch.object(corpora, 'fetch_file', return_value=1000000) as fetch_file, helpers.captured() as (_, stderr):
            with unittest.mock.patch.object(sys, 'argv', ['corpora.py', 'cmudict']):
                corpora.main()
            self.assertEqual(fetch_file.call_args_list, [unittest.mock.call('cmudict')])
            with unittest.mock.patch.object(sys, 'argv', ['corpora.py']):
                corpora.main()
            self.assertEqual([call.args[0] for call in fetch_file.call_args_list[1:]], list(corpora.FILES))
            with unittest.mock.patch.object(sys, 'argv', ['corpora.py', 'nope']), self.assertRaises(SystemExit):
                corpora.main()
        self.assertIn('Downloaded 1 MB in 0 s', stderr.getvalue())
        self.assertIn('unknown files: nope', stderr.getvalue())

    def test_module_main(self):
        code, _, stderr = helpers.run_module('picovox_train.corpora', ['nope'])
        self.assertEqual(code, 2)
        self.assertIn('unknown files: nope', stderr)
