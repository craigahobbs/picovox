# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy

from picovox_train import multi, quality
from tests import data_fixtures, helpers
from tests.helpers import write_wav


# A clean harmonic tone (harmonics below 2 kHz, no noise)
def tone(seconds, rate, f0=150.0):
    times = numpy.arange(int(seconds * rate)) / rate
    return 0.2 * sum(numpy.sin(2 * numpy.pi * f0 * harmonic * times) / harmonic for harmonic in range(1, 11))


# A buzz with an upper band - harmonics to 4.5 kHz and a little noise
def bright(seconds, rate, f0=150.0):
    times = numpy.arange(int(seconds * rate)) / rate
    signal = sum(numpy.sin(2 * numpy.pi * f0 * harmonic * times) / harmonic for harmonic in range(1, 31))
    return 0.15 * signal + 0.002 * numpy.random.default_rng(1).standard_normal(len(times))


class TestHelpers(unittest.TestCase):

    def test_moving_average(self):
        self.assertTrue(numpy.allclose(quality.moving_average(numpy.array([1.0, 1.0, 4.0, 1.0, 1.0]), 3), [1, 2, 2, 2, 1]))

    def test_voiced_runs(self):
        self.assertEqual(quality.voiced_runs([True, True, False, True, False, True, True, True], 2), [(0, 2), (5, 8)])

    def test_band_rms(self):
        # A 5 Hz wobble in a 1 s run (200 frames a second) - in the wobble band, not the flutter band; a short run is none
        times = numpy.arange(200) / 200
        track = 50 * numpy.sin(2 * numpy.pi * 5 * times)
        voiced = numpy.ones(200, dtype=bool)
        self.assertGreater(quality.band_rms(track, voiced, 200, 0.08, 0.33), 20)
        self.assertIsNone(quality.band_rms(track, numpy.zeros(200, dtype=bool), 200, 0.08, 0.33))
        self.assertIsNone(quality.band_rms(track[:55], voiced[:55], 200, 0.08, 0.33))

    def test_band_pass_envelope(self):
        rate = 8000
        times = numpy.arange(rate) / rate
        signal = numpy.sin(2 * numpy.pi * 100 * times) + numpy.sin(2 * numpy.pi * 1000 * times)
        passed = quality.band_pass(signal, rate, 500, 2000)
        self.assertTrue(numpy.allclose(passed, numpy.sin(2 * numpy.pi * 1000 * times), atol=1e-6))
        self.assertTrue(numpy.allclose(quality.envelope(passed)[100:-100], 1, atol=1e-3))
        self.assertEqual(len(quality.envelope(passed[:-1])), rate - 1)


class TestAnalyze(unittest.TestCase):

    # The fake acoustic analysis
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS))

    def analyze(self, samples, rate=24000):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            write_wav(path, samples, rate)
            return quality.analyze((path, 75, 400, rate / 2))

    def test_voiced(self):
        result = self.analyze(bright(1.5, 24000))
        self.assertEqual(set(result), {'path', 'bands', *quality.MEASURES})
        self.assertEqual(len(result['bands']), 20)
        self.assertTrue(all(result[name] is not None for name in quality.MEASURES))
        self.assertLess(result['tilt'], 0)
        self.assertGreater(result['hnr'], 0)
        self.assertLess(result['f0Flutter'], 1)

    def test_silent_and_empty_band(self):
        # Silence - nothing measured; voiced frames with an empty upper band - no crest
        silent = self.analyze(numpy.zeros(24000))
        self.assertEqual(silent['bands'], {})
        self.assertTrue(all(silent[name] is None for name in quality.MEASURES))
        with unittest.mock.patch.object(quality, 'envelope', side_effect=numpy.zeros_like):
            empty = self.analyze(tone(1.0, 24000))
        self.assertIsNotNone(empty['hnr'])
        self.assertIsNone(empty['crest2k'])


class TestPlan(unittest.TestCase):

    def test_voice_table(self):
        voices = quality.voice_table()
        self.assertEqual(len(voices), 12)
        self.assertEqual(voices[0], ('male', 'male', {'arctic-bdl': 1.0}))
        sarah = dict((key, blend) for key, _, blend in voices)['sarah']
        self.assertAlmostEqual(sum(sarah.values()), 1.0)

    def test_plan(self):
        speakers = {'r1': {'pitchFloor': 70, 'pitchCeiling': 250, 'bandwidth': 12000},
                    'r2': {'pitchFloor': 100, 'pitchCeiling': 400, 'bandwidth': 8000}}
        utterances = {'r1': {f's{n}': {'text': f'text {n}', 'wav': f'/r1/s{n}.wav'} for n in range(6)},
                      'r2': {'t0': {'text': 'other', 'wav': '/r2/t0.wav'}}}
        split = {'r1': ['s4', 'gone']}
        voices = [('v1', 'male', {'r1': 0.5, 'r2': 0.5})]
        panel = [{'id': 'p1', 'text': 'Panel.'}]
        jobs, tasks = quality.plan('/out', 3, {'add': {'shelf': -5}}, speakers, split, utterances, panel, voices)
        self.assertEqual(jobs[0], {'text': 'Panel.', 'voice': 'v1', 'wav': '/out/voices/v1-p1.wav', 'edits': {'add': {'shelf': -5}}})
        self.assertEqual(tasks[0], ('voice', 'v1', ('/out/voices/v1-p1.wav', 70, 400, quality.VOICE_BANDWIDTH)))
        # (the held-out utterance first, then the others spread; each synthesized and recorded)
        self.assertEqual([job.get('speaker') for job in jobs[1:]], ['r1', 'r1', 'r1', 'r2'])
        self.assertEqual(jobs[1]['text'], 'text 4')
        self.assertEqual(tasks[1:3], [('synth', 'r1', ('/out/readers/r1-s4.wav', 70, 250, 12000)),
                                      ('human', 'r1', ('/r1/s4.wav', 70, 250, 12000))])
        jobs, _ = quality.plan('/out', 1, None, speakers, split, utterances, panel, voices)
        self.assertNotIn('edits', jobs[0])

    def test_render(self):
        # Each group of jobs to a Node renderer (a process per non-empty group); a failed renderer raises
        children = []
        status = {'code': 0}

        def popen(command, **kwargs):
            child = unittest.mock.Mock()
            child.wait.return_value = status['code']
            children.append((command, kwargs, child))
            return child
        jobs = [{'text': 'a', 'wav': 'a.wav'}, {'text': 'b', 'wav': 'b.wav'}]
        with unittest.mock.patch('subprocess.Popen', side_effect=popen):
            quality.render(jobs, 3)
            self.assertEqual(len(children), 2)
            self.assertEqual(children[0][0], ['node', os.path.join('nn', 'js', 'renderLines.js')])
            self.assertEqual(children[0][1]['cwd'], quality.ROOT)
            self.assertEqual(children[0][2].stdin.write.call_args.args[0], json.dumps(jobs[0]) + '\n')
            status['code'] = 1
            with self.assertRaisesRegex(RuntimeError, 'renderLines.js failed'):
                quality.render(jobs, 1)


class TestMeasure(unittest.TestCase):

    # The fake acoustic analysis
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS))

    def test_measure_and_report(self):
        # One voice of one real reader (its speakers.json settings), its held-out utterances recorded (synthetic), every
        # rendering a synthetic WAV
        with tempfile.TemporaryDirectory() as directory:
            recordings = []
            for number in range(2):
                path = os.path.join(directory, f'rec{number}.wav')
                write_wav(path, bright(0.6, 16000, f0=180), 16000)
                recordings.append(path)
            with open(os.path.join(quality.ROOT, 'data', 'nn', 'multi-split.json'), encoding='utf-8') as file:
                held_out = json.load(file)['test']['arctic-slt'][:2]
            candidates = {'arctic-slt': {'utterances': [{'sentence': sentence, 'text': 'Hello.', 'wav': wav}
                                                        for sentence, wav in zip(held_out, recordings)]}}

            def render(jobs, _processes):
                for job in jobs:
                    write_wav(job['wav'], bright(0.6, 24000, f0=190), 24000)
            out = os.path.join(directory, 'quality')
            args = argparse.Namespace(out=out, per_reader=2, edits='{"add": {"shelf": -5}}', jobs=2)
            with unittest.mock.patch.object(quality, 'voice_table', return_value=[('female', 'female', {'arctic-slt': 1.0})]), \
                    unittest.mock.patch.object(multi, 'candidate_speakers', return_value=candidates), \
                    unittest.mock.patch.object(quality, 'render', side_effect=render), \
                    unittest.mock.patch('multiprocessing.Pool', data_fixtures.FakePool), helpers.captured() as (stdout, _):
                self.assertEqual(quality.measure_command(args), 0)
            with open(os.path.join(out, 'results.json'), encoding='utf-8') as file:
                results = json.load(file)
        self.assertEqual(results['edits'], {'add': {'shelf': -5}})
        self.assertEqual(results['blends'], {'female': {'arctic-slt': 1.0}})
        self.assertEqual([row['kind'] for row in results['rows']].count('voice'), 16)
        self.assertEqual([row['kind'] for row in results['rows']].count('human'), 2)
        output = stdout.getvalue()
        self.assertIn('quality: 20 measured', output)
        self.assertIn('Readers synthesized, less their recordings', output)
        self.assertRegex(output, r'arctic-slt\s+[+-]\d')
        self.assertRegex(output, r'female\s+[+-]\d')

    def test_report_missing(self):
        # A measure missing (None) in a reader's recordings or a voice's own rows, and a voice's reader unmeasured
        rows = [{'kind': 'human', 'who': 'r1', **{name: 1.0 for name in quality.MEASURES}, 'tilt': None},
                {'kind': 'synth', 'who': 'r1', **{name: 2.0 for name in quality.MEASURES}, 'alpha': None},
                {'kind': 'voice', 'who': 'v1', **{name: 3.0 for name in quality.MEASURES}, 'air': None},
                {'kind': 'voice', 'who': 'v2', **{name: 3.0 for name in quality.MEASURES}}]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'results.json')
            with open(path, 'w', encoding='utf-8') as file:
                json.dump({'blends': {'v1': {'r1': 1.0}, 'v2': {'r9': 1.0}}, 'rows': rows}, file)
            with helpers.captured() as (stdout, _):
                self.assertEqual(quality.main(['report', path]), 0)
        lines = {line.split()[0]: line.split()[1:] for line in stdout.getvalue().splitlines() if line.startswith(('r1 ', 'v1 ', 'v2 '))}
        self.assertEqual(lines['r1'][:3], ['n/a', 'n/a', '+1.00'])
        self.assertEqual(lines['v1'][:5], ['n/a', '+2.00', '+2.00', '+2.00', 'n/a'])
        self.assertEqual(set(lines['v2']), {'n/a'})

    def test_main(self):
        with unittest.mock.patch.object(quality, 'measure_command', return_value=0) as measure:
            self.assertEqual(quality.main(['measure', '--out', 'x', '--per-reader', '4', '--edits', '{}', '-j', '3']), 0)
        self.assertEqual(measure.call_args.args[0], argparse.Namespace(command='measure', out='x', per_reader=4, edits='{}', jobs=3))
        with unittest.mock.patch.object(sys, 'argv', ['quality']), self.assertRaises(SystemExit), helpers.captured(), \
                warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.quality', run_name='__main__')
