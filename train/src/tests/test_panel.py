# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import shutil
import subprocess
import tempfile
import time
import unittest
import unittest.mock
import warnings

import numpy

from picovox_train import ROOT, export, package, panel
from tests import helpers, model_fixtures
from tests.helpers import captured, write_wav


TEXTS = [{'id': 'one', 'text': 'Call me Ishmael.'}, {'id': 'two', 'text': 'Is it cold?'}]


class TestPanel(unittest.TestCase):

    def test_listen(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            lines = [json.dumps({'file': path, 'mos': 3.0 + index, 'embedding': [1.0, 0.0]})
                     for index, path in enumerate(kwargs['input'].split('\n'))]
            return subprocess.CompletedProcess(command, 0, stdout='\n'.join(lines) + '\n\n')
        with unittest.mock.patch('subprocess.run', side_effect=run):
            scores = panel.listen(['a.wav', 'b.wav'])
        self.assertEqual(scores, {'a.wav': {'file': 'a.wav', 'mos': 3.0, 'embedding': [1.0, 0.0]},
                                  'b.wav': {'file': 'b.wav', 'mos': 4.0, 'embedding': [1.0, 0.0]}})
        command, kwargs = calls[0]
        self.assertEqual(command, [panel.LISTENERS_PYTHON, '-I', panel.LISTENERS_SCRIPT, '--device', 'cpu', '--files-from', '-'])
        self.assertTrue(os.path.isfile(panel.LISTENERS_SCRIPT))
        self.assertEqual(kwargs['input'], 'a.wav\nb.wav')

    def test_transcribe(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = os.path.join(directory, 'x.wav')
            with open(wav, 'wb') as file:
                file.write(b'RIFF')
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                self.assertEqual(kwargs['cwd'], ROOT)
                with open(command[command.index('-of') + 1] + '.json', 'w', encoding='utf-8') as file:
                    json.dump({'transcription': [{'text': ' Call me'}, {'text': ' Ishmael. '}]}, file)
            with unittest.mock.patch('subprocess.run', side_effect=run):
                self.assertEqual(panel.transcribe(wav), 'Call me Ishmael.')
                self.assertEqual(calls[0][:3], ['whisper-cli', '-m', panel.WHISPER_MODEL])
                # (cached - the transcript is newer than the WAV)
                self.assertEqual(panel.transcribe(wav), 'Call me Ishmael.')
                self.assertEqual(len(calls), 1)
                # (a WAV newer than its transcript is transcribed again)
                later = time.time() + 10
                os.utime(wav, (later, later))
                panel.transcribe(wav)
                self.assertEqual(len(calls), 2)

    def test_transcribe_all(self):
        with unittest.mock.patch.object(panel, 'transcribe', side_effect=lambda path: path.upper()):
            self.assertEqual(panel.transcribe_all(['a', 'b'], workers=2), {'a': 'A', 'b': 'B'})

    def test_words(self):
        self.assertEqual(panel.words("Don't stop - 42 times!"), ['dont', 'stop', '42', 'times'])
        self.assertEqual(panel.words(''), [])

    def test_edit_distance(self):
        self.assertEqual(panel.edit_distance(['a', 'b', 'c'], ['a', 'x', 'c', 'd']), 2)
        self.assertEqual(panel.edit_distance([], ['a']), 1)
        self.assertEqual(panel.edit_distance(['a'], []), 1)
        self.assertEqual(panel.edit_distance(['a', 'b'], ['a', 'b']), 0)

    def test_cosine(self):
        self.assertAlmostEqual(panel.cosine([1, 0], [1, 1]), 2 ** -0.5)
        self.assertAlmostEqual(panel.cosine([1, 2], [2, 4]), 1.0)

    def test_panel_texts(self):
        texts = panel.panel_texts()
        self.assertEqual(len(texts), 16)
        self.assertTrue(all(set(text) >= {'id', 'text'} for text in texts))


class TestPanelScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        data_dir = cls.patches.__enter__()
        model_fixtures.write_data(data_dir)
        model, state = export.load_model(model_fixtures.write_checkpoint(os.path.join(cls.root, 'voice.pt')))
        export.export(model, state, 'model', os.path.join(cls.root, 'export'))
        cls.model_path = os.path.join(cls.root, 'export', 'model')
        cls.enterClassContext(unittest.mock.patch.object(panel.subprocess, 'run', side_effect=cls.speak_js))

    # tools/speak.js - a tenth of a second of silence for each batch item, and its report
    @classmethod
    def speak_js(cls, command, check, capture_output, text, cwd):
        assert command[:3] == ['node', os.path.join(ROOT, 'tools', 'speak.js'), '-m'] and command[4] == '--batch'
        assert (check, capture_output, text, cwd) == (True, True, True, ROOT)
        rate = int(command[command.index('-r') + 1])
        with open(command[5], encoding='utf-8') as file:
            batch = json.load(file)
        for item in batch:
            write_wav(item['wav'], numpy.zeros(rate // 10), rate)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'count': len(batch), 'audioSeconds': 0.1 * len(batch)}) + '\n')

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_speak(self):
        out_dir = os.path.join(self.root, 'speak')
        os.makedirs(out_dir)
        batch = [{'text': 'Call me Ishmael.', 'wav': os.path.join(out_dir, 'a.wav'), 'seed': 1, 'speaker': 1}]
        report = panel.speak(self.model_path, batch, os.path.join(out_dir, 'batch.json'), 16000)
        self.assertGreater(report['audioSeconds'], 0)
        self.assertTrue(os.path.getsize(os.path.join(out_dir, 'a.wav')) > 44)
        with open(os.path.join(out_dir, 'batch.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file), batch)

    def score(self, out_dir, reference=None):
        os.makedirs(out_dir)
        embeddings = {'female': [1.0, 0.0], 'male': [0.0, 1.0]}

        def listen(paths):
            for path in paths:
                self.assertTrue(os.path.exists(path))
            return {path: {'mos': 3.0 if os.path.basename(path).startswith('female') else 4.0,
                           'embedding': embeddings[os.path.basename(path).rsplit('-', 1)[0]]}
                    for path in paths}

        def transcribe_all(paths, workers):
            self.assertEqual(workers, 2)
            return {path: 'call me ishmael' if path.endswith('one.wav') else 'is it hot' for path in paths}
        settings = {'speakers': [['female', 'female'], ['male', 'male']], 'rate': 16000}
        with unittest.mock.patch.object(panel, 'panel_texts', return_value=TEXTS), \
                unittest.mock.patch.object(panel, 'listen', side_effect=listen), \
                unittest.mock.patch.object(panel, 'transcribe_all', side_effect=transcribe_all):
            result = panel.score_model(self.model_path, out_dir, settings, reference, whisper_workers=2)
        self.assertEqual([name for name in os.listdir(out_dir) if name.endswith('.wav')], [])
        return result

    def test_score_model(self):
        result = self.score(os.path.join(self.root, 'score'))
        self.assertGreater(result['panelSpeed']['audioSeconds'], 0)
        self.assertEqual(set(result['speakers']), {'female', 'male'})
        female = result['speakers']['female']
        self.assertEqual((female['sex'], female['mos'], female['mosMin']), ('female', 3.0, 3.0))
        # (one word wrong of six)
        self.assertAlmostEqual(female['wer'], 100 / 6)
        self.assertEqual(female['embedding'], [1.0, 0.0])
        self.assertEqual(set(result['utterances']),
                         {'female-one.wav', 'female-two.wav', 'male-one.wav', 'male-two.wav'})
        self.assertEqual(result['utterances']['male-two.wav']['transcript'], 'is it hot')
        self.assertEqual(result['panel'], {'mos': 3.5, 'wer': 100 / 6})

        # Against a reference - the speaker similarity
        reference = {'speakers': {'female': {'embedding': [1.0, 1.0]}, 'male': {'embedding': [0.0, 1.0]}},
                     'utterances': {name: {'embedding': [0.0, 1.0]} for name in result['utterances']}}
        result = self.score(os.path.join(self.root, 'score-reference'), reference)
        self.assertAlmostEqual(result['speakers']['female']['similarity'], 2 ** -0.5)
        self.assertAlmostEqual(result['speakers']['female']['pairedSimilarity'], 0.0)
        self.assertAlmostEqual(result['panel']['similarity'], (2 ** -0.5 + 1) / 2)
        self.assertAlmostEqual(result['panel']['pairedSimilarity'], 0.5)
        self.assertAlmostEqual(result['panel']['similarityMin'], 2 ** -0.5)


# The evaluation's recipe - a tiny panel section
PANEL = {'speakers': [['female', 'female'], ['male', 'male']], 'rate': 16000,
         'tolerance': {'mos': 0.05, 'wer': 1.0, 'similarity': 0.85}}


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.root = cls.temp.name
        cls.patches = model_fixtures.patched_dirs(cls.root)
        data_dir = cls.patches.__enter__()
        model_fixtures.write_data(data_dir)
        cls.checkpoint = model_fixtures.write_checkpoint(os.path.join(cls.root, 'voice.pt'), seed=4)
        cls.config = os.path.join(cls.root, 'train.json')
        with open(cls.config, 'w', encoding='utf-8') as file:
            json.dump({'panel': PANEL}, file)

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    def tearDown(self):
        shutil.rmtree(panel.EVAL_DIR, ignore_errors=True)

    # The panel's scores of an export - the reference's similarity if given
    def score_model(self, model_path, out_dir, settings, reference, whisper_workers):
        self.assertTrue(os.path.exists(model_path + '.json'))
        self.assertEqual((settings, whisper_workers), (PANEL, 2))
        return {'panel': {'mos': 3.0 if reference is None else 3.5, 'wer': 2.0, **({} if reference is None else {'similarity': 0.9})},
                'panelSpeed': {'realTimeTotal': 20.0}, 'outDir': out_dir}

    def test_settings(self):
        self.assertEqual(panel.load_config(self.config), PANEL)
        self.assertEqual(len(panel.load_config()['speakers']), 12)
        self.assertEqual(panel.root_path('/a/b'), '/a/b')
        self.assertEqual(panel.root_path('train/data'), os.path.join(ROOT, 'train/data'))
        with captured() as (_, stderr):
            panel.log_line('hello')
        self.assertRegex(stderr.getvalue(), r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d hello\n$')
        with open(os.path.join(panel.EVAL_DIR, 'eval.log'), encoding='utf-8') as file:
            self.assertTrue(file.read().endswith(' hello\n'))

    def test_evaluate_model(self):
        # A checkpoint - exported and packaged; a previous evaluation's directory replaced
        os.makedirs(os.path.join(panel.EVAL_DIR, 'voice', 'stale'))
        with unittest.mock.patch.object(panel, 'score_model', side_effect=self.score_model):
            result = panel.evaluate_model('voice', self.checkpoint, PANEL, whisper_workers=2)
        self.assertEqual((result['name'], result['step'], result['config']['dim']), ('voice', 3, 16))
        self.assertEqual(result['panel']['mos'], 3.0)
        self.assertGreater(result['size']['moduleBytes'], result['size']['binBytes'])
        self.assertEqual(result['parameters'], export.parameter_count(export.load_model(self.checkpoint)[0]))
        out_dir = os.path.join(panel.EVAL_DIR, 'voice')
        self.assertFalse(os.path.exists(os.path.join(out_dir, 'stale')))
        self.assertEqual(panel.load_result('voice'), result)
        with open(os.path.join(out_dir, 'model.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file)['defaultSpeaker'], 0)

    def test_evaluate_module(self):
        # A packaged module - unpacked (tools/unpack.js), scored as packaged
        model, state = export.load_model(self.checkpoint)
        export_dir = os.path.join(self.root, 'module-export')
        export.export(model, state, 'model', export_dir)
        text, _ = package.package(os.path.join(export_dir, 'model'))
        module_path = os.path.join(self.root, 'voiceModel.js')
        with open(module_path, 'w', encoding='utf-8') as file:
            file.write(text)

        # (tools/unpack.js - its export)
        def unpack(command, check):
            self.assertEqual((command[:3], check), (['node', os.path.join(ROOT, 'tools', 'unpack.js'), module_path], True))
            manifest, weights = helpers.unpack_module(command[2])
            with open(command[3] + '.json', 'w', encoding='utf-8') as file:
                json.dump(manifest, file)
            with open(command[3] + '.bin', 'wb') as file:
                file.write(weights)
        with unittest.mock.patch.object(panel, 'score_model', side_effect=self.score_model), \
                unittest.mock.patch.object(panel.subprocess, 'run', side_effect=unpack):
            result = panel.evaluate_model('baseline', module_path, PANEL, reference={'speakers': {}}, whisper_workers=2)
        self.assertEqual((result['checkpoint'], result['step']), (module_path, None))
        self.assertEqual(result['parameters'], export.parameter_count(model))
        self.assertEqual(result['size']['encoding'], 'module')
        self.assertEqual(result['size']['moduleBytes'], len(text.encode('utf-8')))
        self.assertEqual(result['size']['binBytes'], os.path.getsize(os.path.join(export_dir, 'model.bin')))
        self.assertEqual(result['panel']['mos'], 3.5)

    def run_main(self, *argv):
        with captured() as (_, stderr):
            status = panel.main(['--config', self.config, *argv])
        return status, stderr.getvalue()

    def test_evaluate(self):
        # The baseline alone, then the voice against it (its similarity)
        calls = []

        def evaluate_model(name, checkpoint, settings, reference, whisper_workers):
            self.assertEqual(settings, PANEL)
            calls.append((name, checkpoint, reference, whisper_workers))
            result = {'name': name, 'parameters': 10, 'size': {'moduleBytes': 100, 'moduleGzipBytes': 90}, 'seconds': 1.0,
                      'panel': {'mos': 3.0, 'wer': 1.5, **({'similarity': 0.9} if reference else {})}}
            os.makedirs(os.path.join(panel.EVAL_DIR, name), exist_ok=True)
            with open(os.path.join(panel.EVAL_DIR, name, 'result.json'), 'w', encoding='utf-8') as file:
                json.dump(result, file)
            return result
        with unittest.mock.patch.object(panel, 'evaluate_model', side_effect=evaluate_model):
            status, output = self.run_main('eval', 'baseline=lib/models/voiceModel.js')
            self.assertEqual(status, 0)
            self.assertIn('baseline: 10 parameters, module 100 bytes (90 gzip); UTMOS 3.000, WER 1.50% (1 s)', output)
            status, output = self.run_main('eval', f'voice={self.checkpoint}', '--reference', 'baseline', '--whisper-workers', '3')
            self.assertEqual(status, 0)
            self.assertIn('WER 1.50%, similarity to baseline 0.900 (1 s)', output)
        self.assertEqual(calls[0], ('baseline', 'lib/models/voiceModel.js', None, 6))
        self.assertEqual(calls[1][:2], ('voice', self.checkpoint))
        self.assertEqual((calls[1][2]['name'], calls[1][3]), ('baseline', 3))

    def write_result(self, name, mos, wer, similarity=None):
        os.makedirs(os.path.join(panel.EVAL_DIR, name), exist_ok=True)
        with open(os.path.join(panel.EVAL_DIR, name, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump({'name': name, 'parameters': 100, 'size': {'moduleBytes': 200, 'moduleGzipBytes': 150},
                       'panel': {'mos': mos, 'wer': wer, **({'similarity': similarity} if similarity is not None else {})},
                       'panelSpeed': {'realTimeTotal': 30.0}}, file)

    def test_gate(self):
        # Within the tolerance - passed
        self.write_result('baseline', 4.30, 1.10)
        self.write_result('voice', 4.26, 2.00, 0.95)
        status, output = self.run_main('gate', 'voice', '--against', 'baseline')
        self.assertEqual(status, 0)
        with open(os.path.join(panel.EVAL_DIR, 'summary.json'), encoding='utf-8') as file:
            summary = json.load(file)
        self.assertEqual((summary['name'], summary['against'], summary['passed']), ('voice', 'baseline', True))
        self.assertEqual(summary['tolerance'], PANEL['tolerance'])
        voice, basis = summary['rows']
        self.assertEqual(voice['within'], {'mos': True, 'wer': True, 'similarity': True})
        self.assertEqual((basis['name'], basis['similarity'], basis['realTimeTotal']), ('baseline', None, 30.0))
        self.assertNotIn('within', basis)
        self.assertIn('baseline: 100 parameters, 200 bytes (150 gzip), UTMOS 4.300, WER 1.10%, real time 30.0x', output)
        self.assertIn('gate: voice passed - UTMOS 4.260 (baseline 4.300), WER 2.00% (baseline 1.10%), similarity 0.950 '
                      '(tolerance: UTMOS -0.05, WER +1.0, similarity >= 0.85)', output)

        # Outside it - failed (exit status 1), each failed measure named; no similarity fails
        self.write_result('voice', 4.20, 2.20, 0.80)
        status, output = self.run_main('gate', 'voice', '--against', 'baseline')
        self.assertEqual(status, 1)
        self.assertIn('gate: voice FAILED (mos, wer, similarity) - UTMOS 4.200', output)
        self.write_result('voice', 4.30, 1.10)
        status, output = self.run_main('gate', 'voice', '--against', 'baseline')
        self.assertEqual(status, 1)
        self.assertIn('gate: voice FAILED (similarity) - ', output)
        self.assertIn('similarity 0.000 (tolerance', output)

    def test_module_main(self):
        with unittest.mock.patch('sys.argv', ['panel', '--help']), captured() as (stdout, _), \
                self.assertRaises(SystemExit) as raised, warnings.catch_warnings():
            # (runpy's warning that the module is imported already - the tests imported it)
            warnings.filterwarnings('ignore', "'picovox_train.panel' found in sys.modules", RuntimeWarning)
            runpy.run_module('picovox_train.panel', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('usage:', stdout.getvalue())
