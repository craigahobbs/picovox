# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import hashlib
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error
import warnings

from picovox_train import corpora
from tests import helpers
from tests.data_fixtures import FakeResponse


class TestDownload(unittest.TestCase):

    def test_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.tar')
            with unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'abc')) as urlopen, helpers.captured():
                corpora.download('https://x/a.tar', path)
                corpora.download('https://x/a.tar', path)
            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header('User-agent'), corpora.USER_AGENT)
            self.assertIsNone(request.get_header('Range'))
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'abc')

    def test_download_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.tar')
            # A partial download resumed (206), then one restarted (the server ignores the range - 200)
            with open(path + '.part', 'wb') as file:
                file.write(b'abc')
            with unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'def', status=206)) as urlopen, \
                    helpers.captured():
                corpora.download('https://x/a.tar', path)
            self.assertEqual(urlopen.call_args.args[0].get_header('Range'), 'bytes=3-')
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'abcdef')
            os.replace(path, path + '.part')
            with unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'xyz', length=False)), helpers.captured():
                corpora.download('https://x/a.tar', path)
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'xyz')

    def test_download_complete_part(self):
        # The partial file is already complete - the server answers 416
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.tar')
            with open(path + '.part', 'wb') as file:
                file.write(b'abc')
            error = urllib.error.HTTPError('https://x/a.tar', 416, 'Range Not Satisfiable', {}, None)
            with unittest.mock.patch('urllib.request.urlopen', side_effect=error):
                corpora.download('https://x/a.tar', path)
            error.close()
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'abc')

    def test_download_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.tar')
            error = urllib.error.HTTPError('https://x/a.tar', 500, 'Server Error', {}, None)
            with unittest.mock.patch('urllib.request.urlopen', side_effect=error), self.assertRaises(urllib.error.HTTPError):
                corpora.download('https://x/a.tar', path)
            error.close()
            response = FakeResponse(b'abc')
            response.headers['Content-Length'] = '10'
            with unittest.mock.patch('urllib.request.urlopen', return_value=response), helpers.captured(), \
                    self.assertRaisesRegex(IOError, 'Incomplete download of https://x/a.tar: 3 of 10 bytes'):
                corpora.download('https://x/a.tar', path)
            self.assertFalse(os.path.exists(path))

    def test_download_progress(self):
        # A progress line every 10 seconds
        body = b'x' * ((1 << 20) * 2 + 5)
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (_, stderr), \
                unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(body)), \
                unittest.mock.patch('time.time', side_effect=[100.0, 111.0, 111.0, 112.0, 130.0, 130.0]):
            corpora.download('https://x/a.tar', os.path.join(directory, 'a.tar'))
        self.assertEqual(stderr.getvalue(), '  a.tar: 1 MB (50%), 0.1 MB/s\n  a.tar: 2 MB (100%), 0.1 MB/s\n')

    def test_download_progress_unknown(self):
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (_, stderr), \
                unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'abc', length=False)), \
                unittest.mock.patch('time.time', side_effect=[100.0, 111.0, 111.0]):
            corpora.download('https://x/a.tar', os.path.join(directory, 'a.tar'))
        self.assertEqual(stderr.getvalue(), '  a.tar: 0 MB, 0.0 MB/s\n')


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

    def test_list(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.dict(corpora.FILES, {'one': {'path': directory, 'description': 'One'},
                                                         'two': {'path': os.path.join(directory, 'y'), 'description': 'Two'}},
                                         clear=True), \
                unittest.mock.patch.object(sys, 'argv', ['corpora.py', '--list']), helpers.captured() as (stdout, _):
            corpora.main()
        self.assertEqual(stdout.getvalue(), f'{"one":28s} present  One\n{"two":28s} missing  Two\n')

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
        with unittest.mock.patch.object(sys, 'argv', ['corpora.py', 'nope']), self.assertRaises(SystemExit), \
                helpers.captured(), warnings.catch_warnings():
            # (runpy warns that the module is already imported - it runs a fresh copy as __main__)
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.corpora', run_name='__main__', alter_sys=False)
