# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import json
import os
import tempfile
import types
import unittest
import unittest.mock

import numpy
import torch

from picovox_train import quality, tts
from picovox_train.wav import read_wav
from tests import ctc_fixtures, helpers
from tests.helpers import write_wav


class FakeFeatures:
    """A fake wav2vec2 recognizer's features - each 20 ms frame's level statistics, a layer's a different mix of them"""

    def __init__(self):
        self.wav2vec2 = self

    def __call__(self, inputs, output_hidden_states):
        assert output_hidden_states
        frames = inputs[0, :inputs.shape[1] // 320 * 320].reshape(-1, 320)
        base = torch.stack([frames.mean(1), frames.std(1), frames.abs().max(1).values, (frames ** 2).mean(1)], dim=1)[None]
        return types.SimpleNamespace(hidden_states=tuple(base * (index + 1) + index for index in range(25)))


# A buzz with an upper band - harmonics to 4.5 kHz and a little noise
def bright(seconds, rate, f0=150.0):
    times = numpy.arange(int(seconds * rate)) / rate
    signal = sum(numpy.sin(2 * numpy.pi * f0 * harmonic * times) / harmonic for harmonic in range(1, 31))
    return 0.15 * signal + 0.002 * numpy.random.default_rng(1).standard_normal(len(times))


class TestRender(unittest.TestCase):

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
            self.assertEqual(children[0][0], ['node', os.path.join('tools', 'renderLines.js')])
            self.assertEqual(children[0][1]['cwd'], quality.ROOT)
            self.assertEqual(children[0][2].stdin.write.call_args.args[0], json.dumps(jobs[0]) + '\n')
            # (another voice model module - its absolute path)
            quality.render(jobs[:1], 1, 'build/train/voiceModel.js')
            self.assertEqual(children[2][0], ['node', os.path.join('tools', 'renderLines.js'),
                                              os.path.abspath('build/train/voiceModel.js')])
            status['code'] = 1
            with self.assertRaisesRegex(RuntimeError, 'renderLines.js failed'):
                quality.render(jobs, 1)


class TestGap(unittest.TestCase):

    def test_gap(self):
        # The test split's clips (a training clip left out, the longest outside the clips' durations, per_voice a voice),
        # synthesized (a copy of the source - its distances 0 - but one voice's, a louder tone), against their sources;
        # the renderings removed after
        texts = [('male', 'Hello there.', 'test'), ('male', 'Good night to you all.', 'test'), ('male', 'Not this one.', 'train'),
                 ('male', 'Call me Ishmael, said the stranger at the door of the inn.', 'test'),
                 ('female', 'Call me Ishmael.', 'test'), ('female', 'It was the best of times.', 'test')]
        with ctc_fixtures.tts_corpus(texts) as (directory, _, clips):
            seconds = {clip['text']: len(read_wav(tts.wav_path(directory, clip))[0]) / 24000 for clip in clips}
            limit = (seconds['Good night to you all.'] + seconds['Call me Ishmael, said the stranger at the door of the inn.']) / 2
            renders = []

            def render(jobs, processes, model):
                renders.append((processes, model))
                for job in jobs:
                    source = next(clip for clip in clips if clip['text'] == job['text'] and clip['key'] == job['voice'])
                    samples = read_wav(tts.wav_path(directory, source))[0][:, 0]
                    write_wav(job['wav'], samples if job['voice'] == 'male' else bright(len(samples) / 24000, 24000), 24000)
            out = os.path.join(directory, 'quality')
            args = argparse.Namespace(out=out, per_voice=3, model='m.js', jobs=2)
            with unittest.mock.patch.object(quality, 'render', side_effect=render), \
                    unittest.mock.patch.object(quality, 'GAP_SECONDS', (0.0, limit)), \
                    unittest.mock.patch('picovox_train.ctc.load_ctc', return_value=(FakeFeatures(), {}, 'cpu')), \
                    helpers.captured() as (stdout, _):
                self.assertEqual(quality.gap_command(args), 0)
            with open(os.path.join(out, 'gap.json'), encoding='utf-8') as file:
                result = json.load(file)
            self.assertEqual(os.listdir(os.path.join(out, 'gap')), [])
        self.assertEqual(renders, [(2, 'm.js')])
        self.assertEqual((result['model'], result['perVoice'], result['clips']), ('m.js', 3, 4))
        self.assertEqual(sorted(result['voices']), ['female', 'male'])
        for name in quality.GAP_MEASURES:
            self.assertAlmostEqual(result['voices']['male'][name], 0.0, places=6)
            self.assertGreater(result['voices']['female'][name], 1e-6)
            self.assertAlmostEqual(result[name], result['voices']['female'][name] / 2)
        output = stdout.getvalue()
        self.assertIn('quality gap: 4 clips', output)
        self.assertRegex(output, r'voice\s+fdL12\s+fdL6\n')
        self.assertRegex(output, r'mean\s+[\d.e+-]+')

    def test_trim(self):
        # Leading and trailing silence trimmed (50 ms kept), the loud middle whole
        samples = numpy.concatenate([numpy.zeros(4800), numpy.full(4800, 0.5), numpy.zeros(4800)])
        trimmed = quality.trim(samples, 24000)
        self.assertEqual(len(trimmed), 4800 + 2 * 1200)
        self.assertEqual(float(trimmed.max()), 0.5)

    def test_frechet(self):
        # Two sets of one distribution - near 0; a shifted set - the shift's square
        rng = numpy.random.default_rng(1)
        frames = rng.normal(size=(4000, 3))
        self.assertAlmostEqual(quality.frechet(frames, frames), 0.0, places=6)
        self.assertAlmostEqual(quality.frechet(frames + 2.0, frames), 12.0, places=6)

    def test_gap_lines(self):
        result = {'voices': {'male': {'fdL12': 2051.26, 'fdL6': 1500.0}}, 'fdL12': 2051.26, 'fdL6': 1500.0}
        lines = quality.gap_lines(result)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1].split(), ['male', '2051', '1500'])
        self.assertEqual(lines[2].split(), ['mean', '2051', '1500'])

    def test_main(self):
        with unittest.mock.patch.object(quality, 'gap_command', return_value=0) as gap:
            self.assertEqual(quality.main(['gap', '--out', 'x', '--per-voice', '5', '--model', 'm.js', '-j', '2']), 0)
        self.assertEqual({key: value for key, value in vars(gap.call_args.args[0]).items() if key != 'func'},
                         {'command': 'gap', 'out': 'x', 'per_voice': 5, 'model': 'm.js', 'jobs': 2})
        code, _, stderr = helpers.run_module('picovox_train.quality')
        self.assertEqual(code, 2)
        self.assertIn('the following arguments are required', stderr)
        with tempfile.TemporaryDirectory() as directory:
            with unittest.mock.patch.object(quality, 'gap_command', return_value=0):
                self.assertEqual(quality.main(['gap', '--out', directory]), 0)
