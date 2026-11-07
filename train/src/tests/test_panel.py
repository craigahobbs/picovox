# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import unittest.mock

import numpy

from picovox_train import ROOT, export, package, panel, tts
from tests import ctc_fixtures, helpers, model_fixtures
from tests.helpers import captured, write_wav


TEXTS = [{'id': 'one', 'text': 'Call me Ishmael.'}, {'id': 'two', 'text': 'Is it cold?'}]


class TestPanel(unittest.TestCase):

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

    def test_score_model(self):
        out_dir = os.path.join(self.root, 'score')
        os.makedirs(out_dir)

        def transcribe_all(paths, workers):
            self.assertEqual(workers, 2)
            for path in paths:
                self.assertTrue(os.path.exists(path))
            return {path: 'call me ishmael' if path.endswith('one.wav') else 'is it hot' for path in paths}
        settings = {'speakers': ['female', 'male'], 'rate': 16000}
        with unittest.mock.patch.object(panel, 'panel_texts', return_value=TEXTS), \
                unittest.mock.patch.object(panel, 'transcribe_all', side_effect=transcribe_all):
            result = panel.score_model(self.model_path, out_dir, settings, whisper_workers=2)
        # (the renderings removed after)
        self.assertEqual([name for name in os.listdir(out_dir) if name.endswith('.wav')], [])
        self.assertGreater(result['panelSpeed']['audioSeconds'], 0)
        self.assertEqual(set(result['speakers']), {'female', 'male'})
        # (one word wrong of six)
        self.assertAlmostEqual(result['speakers']['female']['wer'], 100 / 6)
        self.assertEqual(set(result['utterances']), {'female-one.wav', 'female-two.wav', 'male-one.wav', 'male-two.wav'})
        self.assertEqual(result['utterances']['male-two.wav'], {'transcript': 'is it hot'})
        self.assertEqual(set(result['panel']), {'wer'})
        self.assertAlmostEqual(result['panel']['wer'], 100 / 6)


# The evaluation's recipe - a tiny panel section, and the training speech's voices (their keys the panel's speakers)
PANEL = {'rate': 16000, 'gap': {'perVoice': 3}, 'tolerance': {'wer': 1.0, 'gap': 0.02}}
VOICES = {'female': {'voice': 'Harper', 'sex': 'female'}, 'male': {'voice': 'Trevor', 'sex': 'male'}}
SETTINGS = {**PANEL, 'speakers': ['female', 'male']}

# A fake gap (quality.gap's result)
GAP = {'model': None, 'perVoice': 3, 'clips': 4, 'fdL12': 2000.0, 'fdL6': 1500.0,
       'voices': {'female': {'fdL12': 2100.0, 'fdL6': 1600.0}, 'male': {'fdL12': 1900.0, 'fdL6': 1400.0}}}


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
            json.dump({'tts': {'voices': VOICES}, 'panel': {'description': 'the panel', **PANEL}}, file)

    @classmethod
    def tearDownClass(cls):
        cls.patches.__exit__(None, None, None)
        cls.temp.cleanup()

    def tearDown(self):
        shutil.rmtree(panel.EVAL_DIR, ignore_errors=True)

    # The panel's scores of an export
    def score_model(self, model_path, out_dir, settings, whisper_workers):
        self.assertTrue(os.path.exists(model_path + '.json'))
        self.assertEqual((settings, whisper_workers), (SETTINGS, 2))
        return {'panel': {'wer': 2.0}, 'panelSpeed': {'realTimeTotal': 20.0}, 'outDir': out_dir}

    # The gap of the module (quality.gap) - the calls recorded
    def gap(self, calls):
        def gap(model, out_dir, per_voice, jobs):
            calls.append((model, out_dir, per_voice, jobs))
            return {**GAP, 'model': model}
        return gap

    def test_settings(self):
        self.assertEqual(panel.load_config(self.config), SETTINGS)
        self.assertEqual(panel.load_config()['speakers'][:2], ['male', 'female'])
        self.assertEqual(set(panel.load_config()), {'rate', 'gap', 'tolerance', 'speakers'})
        self.assertEqual(panel.root_path('/a/b'), '/a/b')
        self.assertEqual(panel.root_path('train/data'), os.path.join(ROOT, 'train/data'))
        with captured() as (_, stderr):
            panel.log_line('hello')
        self.assertRegex(stderr.getvalue(), r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d hello\n$')
        with open(os.path.join(panel.EVAL_DIR, 'eval.log'), encoding='utf-8') as file:
            self.assertTrue(file.read().endswith(' hello\n'))

    def test_evaluate_model(self):
        # A checkpoint - exported (the package stage's export, the same as export.py's), packaged (its module, which the
        # gap renders with), scored, and its gap measured; a previous evaluation's directory replaced
        os.makedirs(os.path.join(panel.EVAL_DIR, 'voice', 'stale'))
        calls = []
        with unittest.mock.patch.object(panel, 'score_model', side_effect=self.score_model), \
                unittest.mock.patch.object(panel.quality, 'gap', side_effect=self.gap(calls)):
            result = panel.evaluate_model('voice', self.checkpoint, SETTINGS, whisper_workers=2, jobs=3)
        self.assertEqual((result['name'], result['checkpoint']), ('voice', self.checkpoint))
        out_dir = os.path.join(panel.EVAL_DIR, 'voice')
        module_path = os.path.join(out_dir, 'voiceModel.js')
        self.assertEqual(calls, [(module_path, os.path.join(out_dir, 'gap'), 3, 3)])
        self.assertEqual(result['panel'], {'wer': 2.0, 'gap': 2000.0})
        self.assertEqual(result['gap']['voices'], GAP['voices'])
        export_dir = os.path.join(self.root, 'voice-export')
        with helpers.captured():
            export.main([self.checkpoint, '--voice', 'multi', '--out', export_dir])
        for suffix in ('json', 'bin'):
            with open(os.path.join(out_dir, f'multi.{suffix}'), 'rb') as file, \
                    open(os.path.join(export_dir, f'multi.{suffix}'), 'rb') as exported:
                self.assertEqual(file.read(), exported.read(), suffix)
        text = package.package(os.path.join(out_dir, 'multi'))
        with open(module_path, encoding='utf-8') as file:
            self.assertEqual(file.read(), text)
        self.assertEqual(result['size'], {'moduleBytes': len(text.encode('utf-8')),
                                          'moduleGzipBytes': package.module_size(text)['gzipBytes']})
        self.assertEqual(result['parameters'], export.parameter_count(export.load_model(self.checkpoint)[0]))
        self.assertFalse(os.path.exists(os.path.join(out_dir, 'stale')))
        self.assertEqual(panel.load_result('voice'), result)

    def test_evaluate_module(self):
        # A packaged module - unpacked (tools/unpack.js), scored as packaged, its gap rendered with the module itself
        model, state = export.load_model(self.checkpoint)
        export_dir = os.path.join(self.root, 'module-export')
        export.export(model, state, 'model', export_dir)
        text = package.package(os.path.join(export_dir, 'model'))
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
        calls = []
        with unittest.mock.patch.object(panel, 'score_model', side_effect=self.score_model), \
                unittest.mock.patch.object(panel.quality, 'gap', side_effect=self.gap(calls)), \
                unittest.mock.patch.object(panel.subprocess, 'run', side_effect=unpack):
            result = panel.evaluate_model('baseline', module_path, SETTINGS, whisper_workers=2)
        self.assertEqual(result['checkpoint'], module_path)
        self.assertEqual(calls, [(module_path, os.path.join(panel.EVAL_DIR, 'baseline', 'gap'), 3, 8)])
        self.assertEqual(result['parameters'], export.parameter_count(model))
        self.assertEqual(result['size'], {'moduleBytes': len(text.encode('utf-8')),
                                          'moduleGzipBytes': package.module_size(text)['gzipBytes']})
        self.assertEqual(result['panel'], {'wer': 2.0, 'gap': 2000.0})

    def test_cached_result_path(self):
        # A module's cached result - in the training speech's directory, keyed by the module's bytes and the settings
        with ctc_fixtures.tts_corpus([('male', 'Hello there.', 'test')]) as (directory, _, _):
            module_path = os.path.join(self.root, 'cached.js')
            with open(module_path, 'w', encoding='utf-8') as file:
                file.write('export const manifest = {};\n')
            path = panel.cached_result_path(module_path, SETTINGS)
            self.assertEqual(os.path.dirname(path), directory)
            self.assertRegex(os.path.basename(path), r'^panel-[0-9a-f]{12}\.json$')
            self.assertEqual(panel.cached_result_path(module_path, SETTINGS), path)
            self.assertNotEqual(panel.cached_result_path(module_path, {**SETTINGS, 'rate': 24000}), path)
            with open(module_path, 'w', encoding='utf-8') as file:
                file.write('export const manifest = {"a": 1};\n')
            self.assertNotEqual(panel.cached_result_path(module_path, SETTINGS), path)

    def run_main(self, *argv):
        with captured() as (_, stderr):
            status = panel.main(['--config', self.config, *argv])
        return status, stderr.getvalue()

    def test_evaluate(self):
        # The baseline (a module - its result cached in the training speech's directory, read the second time) and
        # the voice (a checkpoint - never cached)
        calls = []

        def evaluate_model(name, checkpoint, settings, whisper_workers, jobs):
            self.assertEqual(settings, SETTINGS)
            calls.append((name, checkpoint, whisper_workers, jobs))
            result = {'name': name, 'parameters': 10, 'size': {'moduleBytes': 100, 'moduleGzipBytes': 90}, 'seconds': 1.0,
                      'panel': {'wer': 1.5, 'gap': 2051.4}}
            os.makedirs(os.path.join(panel.EVAL_DIR, name), exist_ok=True)
            with open(os.path.join(panel.EVAL_DIR, name, 'result.json'), 'w', encoding='utf-8') as file:
                json.dump(result, file)
            return result
        cached = os.path.join(self.root, 'cache', 'panel-abc.json')
        with unittest.mock.patch.object(panel, 'evaluate_model', side_effect=evaluate_model), \
                unittest.mock.patch.object(panel, 'cached_result_path', return_value=cached) as cached_path:
            status, output = self.run_main('eval', 'baseline=lib/models/voiceModel.js')
            self.assertEqual(status, 0)
            self.assertIn('baseline: evaluating lib/models/voiceModel.js', output)
            self.assertIn('baseline: 10 parameters, module 100 bytes (90 gzip); WER 1.50%, gap 2051 (1 s)', output)
            self.assertEqual(cached_path.call_args.args, (os.path.join(ROOT, 'lib/models/voiceModel.js'), SETTINGS))
            with open(cached, encoding='utf-8') as file:
                self.assertEqual(json.load(file)['panel'], {'wer': 1.5, 'gap': 2051.4})
            shutil.rmtree(panel.EVAL_DIR)
            status, output = self.run_main('eval', 'shipped=lib/models/voiceModel.js')
            self.assertEqual(status, 0)
            self.assertIn('shipped: lib/models/voiceModel.js is evaluated (', output)
            self.assertIn('shipped: 10 parameters, module 100 bytes (90 gzip); WER 1.50%, gap 2051 (1 s)', output)
            self.assertEqual(panel.load_result('shipped')['name'], 'shipped')
            status, output = self.run_main('eval', f'voice={self.checkpoint}', '--whisper-workers', '3', '-j', '2')
            self.assertEqual(status, 0)
            self.assertIn('WER 1.50%, gap 2051 (1 s)', output)
        self.assertEqual(calls, [('baseline', 'lib/models/voiceModel.js', 6, 8), ('voice', self.checkpoint, 3, 2)])

    def write_result(self, name, wer, gap):
        os.makedirs(os.path.join(panel.EVAL_DIR, name), exist_ok=True)
        with open(os.path.join(panel.EVAL_DIR, name, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump({'name': name, 'parameters': 100, 'size': {'moduleBytes': 200, 'moduleGzipBytes': 150},
                       'panel': {'wer': wer, 'gap': gap}, 'panelSpeed': {'realTimeTotal': 30.0}}, file)

    def test_gate(self):
        # Within the tolerance - passed (the word error rate a point worse, the gap 2% above)
        self.write_result('baseline', 1.10, 2000.0)
        self.write_result('voice', 2.10, 2040.0)
        status, output = self.run_main('gate', 'voice', '--against', 'baseline')
        self.assertEqual(status, 0)
        with open(os.path.join(panel.EVAL_DIR, 'summary.json'), encoding='utf-8') as file:
            summary = json.load(file)
        self.assertEqual((summary['name'], summary['against'], summary['passed']), ('voice', 'baseline', True))
        self.assertEqual(summary['tolerance'], PANEL['tolerance'])
        voice, basis = summary['rows']
        self.assertEqual(voice['within'], {'wer': True, 'gap': True})
        self.assertEqual((basis['name'], basis['gap'], basis['realTimeTotal']), ('baseline', 2000.0, 30.0))
        self.assertNotIn('within', basis)
        self.assertIn('baseline: 100 parameters, 200 bytes (150 gzip), WER 1.10%, gap 2000, real time 30.0x', output)
        self.assertIn('gate: voice passed - WER 2.10% (baseline 1.10%), gap 2040 (baseline 2000, +2.0%) '
                      '(tolerance: WER +1.0, gap +2%)', output)

        # Outside it - failed (exit status 1), each failed measure named
        self.write_result('voice', 2.20, 2041.0)
        status, output = self.run_main('gate', 'voice', '--against', 'baseline')
        self.assertEqual(status, 1)
        self.assertIn('gate: voice FAILED (wer, gap) - WER 2.20%', output)
        self.write_result('voice', 1.10, 2041.0)
        status, output = self.run_main('gate', 'voice', '--against', 'baseline')
        self.assertEqual(status, 1)
        self.assertIn('gate: voice FAILED (gap) - ', output)

    def test_module_main(self):
        code, stdout, _ = helpers.run_module('picovox_train.panel', ['--help'])
        self.assertEqual(code, 0)
        self.assertIn('usage:', stdout)
        self.assertTrue(tts.PANEL_TEXTS_PATH.endswith('panel-texts.json'))
