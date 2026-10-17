# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import io
import os
import runpy
import sys
import tarfile
import tempfile
import unittest
import unittest.mock
import urllib.error
import warnings

from picovox_train import corpora
from tests import helpers


# A response of the fake urlopen - its status, headers, and body
class FakeResponse(io.BytesIO):

    def __init__(self, body, status=200, length=True):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Length': str(len(body))} if length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


# Write a tar.gz archive of {name: bytes}
def write_archive(path, files):
    with tarfile.open(path, 'w:gz') as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


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


class TestArchives(unittest.TestCase):

    def test_concatenate(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = []
            for number, content in enumerate((b'ab', b'cd')):
                parts.append(os.path.join(directory, f'p{number}'))
                with open(parts[-1], 'wb') as file:
                    file.write(content)
            path = os.path.join(directory, 'all')
            corpora.concatenate(parts, path)
            os.remove(parts[0])
            corpora.concatenate(parts, path)
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'abcd')

    def test_unpack(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = os.path.join(directory, 'a.tar.gz')
            write_archive(archive, {'lab/phone/dev-clean/a.lab': b'1', './lab/word/dev-clean/a.lab': b'2', 'lab/phone/train/b.lab': b'3'})
            corpora.unpack(archive, os.path.join(directory, 'all'))
            corpora.unpack(archive, os.path.join(directory, 'some'), ('lab/phone/dev-clean/', 'lab/word/dev-clean/'))
            self.assertTrue(os.path.exists(os.path.join(directory, 'all', 'lab', 'phone', 'train', 'b.lab')))
            self.assertTrue(os.path.exists(os.path.join(directory, 'some', 'lab', 'word', 'dev-clean', 'a.lab')))
            self.assertFalse(os.path.exists(os.path.join(directory, 'some', 'lab', 'phone', 'train')))
            self.assertEqual(corpora.directory_size(os.path.join(directory, 'all')), 3)


class TestFetch(unittest.TestCase):

    # The corpora table with two test corpora under a directory, and urlopen serving their archives
    def corpora_patches(self, directory, archives):
        def urlopen(request, timeout):
            self.assertEqual(timeout, 60)
            return FakeResponse(archives[request.full_url])
        test_corpora = {
            'one': {'urls': ['https://x/one.tar.gz'], 'dest': os.path.join(directory, 'one'),
                    'check': os.path.join(directory, 'one', 'data'), 'description': 'One corpus', 'license': 'CC0'},
            'split': {'urls': ['https://x/split.tar.gz-0', 'https://x/split.tar.gz-1'], 'dest': os.path.join(directory, 'split'),
                      'check': os.path.join(directory, 'split', 'lab', 'phone'), 'members': ('lab/phone/',),
                      'sizePaths': [os.path.join(directory, 'split', 'lab')], 'description': 'A split corpus', 'license': 'MIT'},
            'doc': {'urls': ['https://x/doc.tar.gz'], 'dest': os.path.join(directory, 'doc'),
                    'check': os.path.join(directory, 'doc', 'speakers.tsv'), 'description': 'Documentation', 'license': 'CC0'}
        }
        return [unittest.mock.patch.dict(corpora.CORPORA, test_corpora, clear=True),
                unittest.mock.patch.object(corpora, 'CORPORA_DIR', directory),
                unittest.mock.patch.object(corpora, 'DOWNLOADS_DIR', os.path.join(directory, 'downloads')),
                unittest.mock.patch('urllib.request.urlopen', side_effect=urlopen)]

    def test_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            one = os.path.join(directory, 'one.tar.gz')
            write_archive(one, {'data/a.wav': b'12345'})
            split = os.path.join(directory, 'split.tar.gz')
            write_archive(split, {'lab/phone/a.lab': b'1', 'lab/word/a.lab': b'2'})
            with open(one, 'rb') as file:
                one_bytes = file.read()
            with open(split, 'rb') as file:
                split_bytes = file.read()
            archives = {'https://x/one.tar.gz': one_bytes, 'https://x/split.tar.gz-0': split_bytes[:10],
                        'https://x/split.tar.gz-1': split_bytes[10:]}
            patches = self.corpora_patches(directory, archives)
            for patch in patches:
                patch.start()
            try:
                with helpers.captured() as (_, stderr):
                    self.assertEqual(corpora.fetch('one', False), len(one_bytes))
                    self.assertEqual(corpora.fetch('one', False), 0)
                    self.assertEqual(corpora.fetch('split', True), len(split_bytes))
                downloads = sorted(os.listdir(os.path.join(directory, 'downloads')))
                self.assertTrue(os.path.exists(os.path.join(directory, 'one', 'data', 'a.wav')))
                self.assertTrue(os.path.exists(os.path.join(directory, 'split', 'lab', 'phone', 'a.lab')))
                self.assertFalse(os.path.exists(os.path.join(directory, 'split', 'lab', 'word')))
                # An archive without the corpus's check path
                archives['https://x/doc.tar.gz'] = one_bytes
                with helpers.captured(), self.assertRaisesRegex(IOError, 'doc: unpacking did not produce'):
                    corpora.fetch('doc', False)
            finally:
                for patch in reversed(patches):
                    patch.stop()
        self.assertEqual(downloads, ['split.tar.gz', 'split.tar.gz-0', 'split.tar.gz-1'])
        self.assertIn('one: downloading https://x/one.tar.gz\n', stderr.getvalue())
        self.assertIn('one: unpacking 0 MB\none: present\nsplit: downloading https://x/split.tar.gz-0\n', stderr.getvalue())

    def test_write_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            patches = self.corpora_patches(directory, {})
            for patch in patches:
                patch.start()
            try:
                os.makedirs(os.path.join(directory, 'one', 'data'))
                with open(os.path.join(directory, 'one', 'data', 'a.wav'), 'wb') as file:
                    file.write(b'x' * 2000000)
                os.makedirs(os.path.join(directory, 'split', 'lab', 'phone'))
                corpora.write_readme()
                with open(os.path.join(directory, 'README.md'), encoding='utf-8') as file:
                    first = file.read()
                # An older README (no other sections) is rewritten
                with open(os.path.join(directory, 'README.md'), 'w', encoding='utf-8') as file:
                    file.write('# An older README\n')
                corpora.write_readme()
                with open(os.path.join(directory, 'README.md'), encoding='utf-8') as file:
                    rewritten = file.read()
                # Another tool's section is kept
                with open(os.path.join(directory, 'README.md'), 'a', encoding='utf-8') as file:
                    file.write('\n<!-- other -->\n## other\n<!-- other end -->\n\n')
                os.makedirs(os.path.join(directory, 'doc'))
                with open(os.path.join(directory, 'doc', 'speakers.tsv'), 'w', encoding='utf-8') as file:
                    file.write('x')
                corpora.write_readme()
                with open(os.path.join(directory, 'README.md'), encoding='utf-8') as file:
                    second = file.read()
                corpora.write_readme()
                with open(os.path.join(directory, 'README.md'), encoding='utf-8') as file:
                    third = file.read()
            finally:
                for patch in reversed(patches):
                    patch.stop()
        self.assertTrue(first.startswith('# Speech corpora\n'))
        self.assertEqual(rewritten, first)
        self.assertIn('## one\n\n- Source: https://x/one.tar.gz\n- Contents: One corpus\n- License: CC0\n- Size: 2 MB unpacked\n', first)
        self.assertIn('## split\n\n- Source: https://x/split.tar.gz-0 (and parts)\n', first)
        self.assertIn('- Size: 0 MB unpacked\n', first)
        self.assertIn('## doc\n\n- Source: https://x/doc.tar.gz\n- Contents: Documentation\n- License: CC0\n'
                      '- Size: not downloaded\n', first)
        self.assertIn('- Size: small (documentation files)\n', second)
        self.assertTrue(second.endswith('<!-- other -->\n## other\n<!-- other end -->\n'))
        self.assertEqual(second, third)


class TestMain(unittest.TestCase):

    def test_list(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.dict(corpora.CORPORA, {'one': {'check': directory, 'description': 'One'},
                                                           'two': {'check': os.path.join(directory, 'x'), 'description': 'Two'}},
                                         clear=True), \
                unittest.mock.patch.object(sys, 'argv', ['corpora.py', '--list']), helpers.captured() as (stdout, _):
            corpora.main()
        self.assertEqual(stdout.getvalue(), f'{"one":28s} present  One\n{"two":28s} missing  Two\n')

    def test_main(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(corpora, 'CORPORA_DIR', os.path.join(directory, 'corpora')), \
                unittest.mock.patch.object(corpora, 'fetch', return_value=2000000) as fetch, \
                unittest.mock.patch.object(corpora, 'write_readme') as write_readme, helpers.captured() as (_, stderr):
            with unittest.mock.patch.object(sys, 'argv', ['corpora.py', 'arctic-slt', '--keep-archives']):
                corpora.main()
            self.assertEqual(fetch.call_args_list, [unittest.mock.call('arctic-slt', True)])
            with unittest.mock.patch.object(sys, 'argv', ['corpora.py']):
                corpora.main()
            self.assertEqual([call.args[0] for call in fetch.call_args_list[1:]], list(corpora.CORPORA))
            self.assertTrue(os.path.isdir(os.path.join(directory, 'corpora')))
            with unittest.mock.patch.object(sys, 'argv', ['corpora.py', 'nope']), self.assertRaises(SystemExit):
                corpora.main()
        self.assertEqual(write_readme.call_count, 2)
        self.assertIn('Downloaded 2 MB in 0 s', stderr.getvalue())
        self.assertIn('unknown corpora: nope', stderr.getvalue())

    def test_module_main(self):
        with unittest.mock.patch.object(sys, 'argv', ['corpora.py', 'nope']), self.assertRaises(SystemExit), \
                helpers.captured(), warnings.catch_warnings():
            # (runpy warns that the module is already imported - it runs a fresh copy as __main__)
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.corpora', run_name='__main__', alter_sys=False)
