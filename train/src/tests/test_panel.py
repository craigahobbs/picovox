# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import subprocess
import tempfile
import time
import unittest
import unittest.mock

import numpy

from picovox_train import ROOT, export, panel
from tests import model_fixtures
from tests.helpers import write_wav


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
        model, state = export.load_model(model_fixtures.write_checkpoint(os.path.join(cls.root, 'teacher.pt')))
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
