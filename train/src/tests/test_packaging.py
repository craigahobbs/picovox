# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import base64
import gzip
import hashlib
import json
import os
import re
import runpy
import subprocess
import tempfile
import unittest
import unittest.mock
import warnings

from picovox_train import ROOT, export, package
from tests import helpers, model_fixtures


class TestPackageText(unittest.TestCase):

    def test_wrap_json(self):
        value = {'name': 'voice', 'list': [1, 2.5, None, True, 'x' * 150], 'nested': {'empty': [], 'text': 'é "quoted"'},
                 'tuple': (1, 2)}
        text = package.wrap_json(value, width=80, indent='  ', first=10)
        lines = text.split('\n')
        self.assertGreater(len(lines), 2)
        self.assertLessEqual(len(lines[0]), 70)
        for line in lines[1:]:
            self.assertTrue(line.startswith('  '))
            self.assertLessEqual(len(line) - 2, 78)
        self.assertIn('" +\n', text)
        self.assertEqual(helpers.js_literal(text), json.loads(json.dumps(value)))

    def test_wrap_json_unbreakable(self):
        # (a token longer than the line can't be broken)
        self.assertEqual(package.wrap_json('y' * 30, width=10), '"' + 'y' * 30 + '"')
        self.assertEqual(package.wrap_json({}), '{}')

    def test_wrap_base64(self):
        data = bytes(range(256)) * 3
        text = package.wrap_base64(data, width=50)
        self.assertTrue(all(len(line) <= 50 for line in text.split('\n')))
        self.assertEqual(base64.b64decode(text.replace('\n', '')), data)
        self.assertEqual(package.wrap_base64(b''), '')

    def test_module_text(self):
        text = package.module_text({'voice': 'multi', 'parameterCount': 123}, b'\x01\x02')
        lines = text.split('\n')
        self.assertEqual(lines[0], package.HEADER)
        self.assertEqual(lines[1], '// The multi acoustic model (123 parameters, 7-bit weights) for')
        self.assertEqual(lines[2], '// lib/package.js loadPackagedModel - the export\'s manifest, and its weights (2 bytes) as base64.')
        self.assertIn("export const encoding = 'raw';", lines)
        self.assertEqual(lines[-4:], ['export const weights = `', 'AQI=', '`;', ''])
        text = package.module_text({}, b'')
        self.assertIn('// The model acoustic model (None parameters, 7-bit weights) for', text)
        self.assertIn('(0 bytes) as base64.', text)

    def test_module_size(self):
        text = 'export const weights = `AAAA`;\n' * 10
        self.assertEqual(package.module_size(text), {'bytes': len(text), 'gzipBytes': len(gzip.compress(text.encode(), 9, mtime=0))})

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'file')
            with open(path, 'wb') as file:
                file.write(b'hello')
            self.assertEqual(package.file_sha256(path), hashlib.sha256(b'hello').hexdigest())


class TestProvenance(unittest.TestCase):

    def test_provenance_git(self):
        # (the repository's own git, read-only)
        result = package.provenance()
        self.assertRegex(result['commit'], '^[0-9a-f]{40}$')
        self.assertIsInstance(result['dirty'], bool)
        self.assertEqual(set(result), {'commit', 'dirty'})

    def test_provenance(self):
        outputs = {'rev-parse': 'abc123\n', 'status': ' M lib/x.js\n'}

        def run(command, **_kwargs):
            self.assertEqual(command[:3], ['git', '-C', ROOT])
            return subprocess.CompletedProcess(command, 0, stdout=outputs[command[3]])
        with tempfile.TemporaryDirectory() as directory:
            data_dir = os.path.join(directory, 'data', 'tts')
            os.makedirs(data_dir)
            with open(os.path.join(data_dir, 'index.json'), 'w', encoding='utf-8') as file:
                file.write('{}')
            scores = os.path.join(directory, 'result.json')
            with open(scores, 'w', encoding='utf-8') as file:
                json.dump({'panel': {'mos': 3.123456, 'wer': 2.5}}, file)
            recipe = os.path.join(ROOT, 'train', 'data', 'train.json')
            with unittest.mock.patch('subprocess.run', side_effect=run):
                result = package.provenance(recipe, [data_dir + '/'], scores)
        self.assertEqual(result, {
            'commit': 'abc123', 'dirty': True,
            'recipe': {'path': 'train/data/train.json', 'sha256': package.file_sha256(recipe)},
            'data': {'tts': hashlib.sha256(b'{}').hexdigest()},
            'scores': {'mos': 3.1235, 'wer': 2.5}
        })
        outputs['status'] = ''
        with unittest.mock.patch('subprocess.run', side_effect=run):
            self.assertEqual(package.provenance(), {'commit': 'abc123', 'dirty': False})


class TestPackage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        data_dir = cls.patches.__enter__()
        model_fixtures.write_data(data_dir)
        checkpoint = model_fixtures.write_checkpoint(os.path.join(cls.root, 'teacher.pt'))
        model, state = export.load_model(checkpoint)
        cls.export_dir = os.path.join(cls.root, 'export')
        export.export(model, state, 'model', cls.export_dir)
        cls.model_path = os.path.join(cls.export_dir, 'model')

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_package(self):
        # The weights raw, as base64; the provenance in the manifest
        text = package.package(self.model_path)
        self.assertIn("export const encoding = 'raw';", text)
        self.assertIn(f'its weights ({os.path.getsize(self.model_path + ".bin")} bytes) as base64.', text)
        self.assertNotIn('provenance', text)
        text = package.package(self.model_path, provenance_info={'commit': 'abc'})
        self.assertIn('"provenance":{"commit":"abc"}', text)
        self.assertTrue(all(len(line) <= package.LINE_LENGTH for line in text.split('\n')))

    def test_main_round_trip(self):
        # The module, unpacked, is the export (lib/package.js's loading of package.py's modules - test/testPackage.js)
        output = os.path.join(self.root, 'out', 'voiceModel.js')
        with helpers.captured() as (stdout, _), \
                unittest.mock.patch.object(package, 'provenance', return_value={'commit': 'abc', 'dirty': False}) as provenance:
            self.assertEqual(package.main([self.model_path, '-o', output, '--provenance', '--recipe', 'train/data/train.json',
                                           '--data', 'a', 'b', '--scores', 'result.json']), 0)
        provenance.assert_called_once_with('train/data/train.json', ['a', 'b'], 'result.json')
        with open(output, encoding='utf-8') as file:
            self.assertEqual(json.loads(stdout.getvalue()), {'output': output, **package.module_size(file.read())})
        manifest, weights = helpers.unpack_module(output)
        with open(self.model_path + '.bin', 'rb') as original:
            self.assertEqual(weights, original.read())
        with open(self.model_path + '.json', encoding='utf-8') as original:
            self.assertEqual(manifest, {**json.load(original), 'provenance': {'commit': 'abc', 'dirty': False}})

    def test_main_no_provenance(self):
        output = os.path.join(self.root, 'plain.js')
        with helpers.captured() as (stdout, _), unittest.mock.patch.object(package, 'provenance') as provenance:
            self.assertEqual(package.main([self.model_path, '-o', output]), 0)
        provenance.assert_not_called()
        self.assertEqual(json.loads(stdout.getvalue())['output'], output)
        with open(output, encoding='utf-8') as file:
            text = file.read()
        self.assertNotIn('provenance', text)
        self.assertTrue(re.match(r'// Generated by picovox_train\.package - do not edit\n', text))

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['package', '--help']), helpers.captured() as (stdout, _), \
                self.assertRaises(SystemExit) as raised, warnings.catch_warnings():
            # (runpy's warning that the module is imported already - the tests imported it)
            warnings.filterwarnings('ignore', "'picovox_train.package' found in sys.modules", RuntimeWarning)
            runpy.run_module('picovox_train.package', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('usage:', stdout.getvalue())
