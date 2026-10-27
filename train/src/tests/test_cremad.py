# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
import unittest.mock
import urllib.error
import warnings

import numpy
import soundfile
import torch

from picovox_train import cremad, data, frontend
from tests import data_fixtures, helpers


# A recognizer vocabulary - the blank and specials, then every IPA token
VOCAB = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
for _tokens in cremad.IPA_TOKENS.values():
    for _token in _tokens:
        VOCAB.setdefault(_token, len(VOCAB))


# The recognizer's greedy string of a text's phones (each phone's usual IPA token), less the last tail phones
def greedy_of(text, tail=0):
    keys = [key for _, key in cremad.phone_positions(frontend.tokenize(text)['tokens'])]
    return ' '.join(cremad.IPA_TOKENS[key][0] for key in keys[:len(keys) - tail])


# A clip's alignment labels [[label, start, end]] (our token names, 'pau' edges) - and its seconds
def clip_labels(text, trailing=True):
    labels, seconds = data_fixtures.text_labels(text, 'mfa')
    rows = [['pau', 0.0, data_fixtures.EDGE_SECONDS]] + [[label, start, end] for label, start, end in labels]
    if trailing:
        rows.append(['pau', labels[-1][2], seconds])
    else:
        seconds = labels[-1][2]
    return rows, seconds


# A fake wav2vec2 CTC model - random logits, one frame per 20 ms
class FakeCtc:

    def __init__(self):
        self.device = None
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, inputs, attention_mask):
        self.calls.append(tuple(inputs.shape))
        generator = torch.Generator().manual_seed(inputs.shape[1])
        logits = torch.randn(inputs.shape[0], inputs.shape[1] // 320, len(VOCAB), generator=generator) * 2
        logits[..., 0] += 3 * (attention_mask[:, ::320][:, :logits.shape[1]] == 0)
        return types.SimpleNamespace(logits=logits)

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return lengths // 320


# The fake recognizer's modules (transformers, huggingface_hub) and a snapshot directory with its vocabulary
@contextlib.contextmanager
def fake_recognizer():
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, 'vocab.json'), 'w', encoding='utf-8') as file:
            json.dump(VOCAB, file)
        model = FakeCtc()
        transformers = types.ModuleType('transformers')
        transformers.Wav2Vec2ForCTC = types.SimpleNamespace(from_pretrained=unittest.mock.Mock(return_value=model))
        hub = types.ModuleType('huggingface_hub')
        hub.snapshot_download = unittest.mock.Mock(return_value=directory)
        with data_fixtures.fake_modules(transformers=transformers, huggingface_hub=hub), \
                unittest.mock.patch.dict(os.environ), \
                unittest.mock.patch.object(torch.backends.mps, 'is_available', return_value=False):
            yield model, hub


# The tiny CREMA-D clips - {id: (the sentence read, alignment kind, greedy tail)}: kind 'pau' (labels with a trailing
# pause), 'end' (labels ending with a phone, a clipped recording), 'none' (no alignment, a silent recording)
CLIPS = {
    '1001_IEO_ANG_HI': ('IEO', 'pau', 0),
    '1002_TIE_NEU_XX': ('DFA', 'pau', 0),
    '1001_IOM_HAP_XX': ('IOM', 'none', 0),
    '1001_WSI_DIS_XX': ('WSI', 'pau', 3),
    '1002_IWW_SAD_XX': ('IWW', 'pau', 5),
    '1002_TAI_ANG_XX': ('TAI', 'garbled', 0),
    '1002_MTI_FEA_XX': ('MTI', 'end', 3)
}


# A CREMA-D layout in a temporary directory (AudioWAV, the demographics and ratings, alignments.jsonl), the module
# constants patched to it, and multiprocessing.Pool in this process
@contextlib.contextmanager
def tiny_cremad():
    with tempfile.TemporaryDirectory() as directory:
        cremad_dir = os.path.join(directory, 'crema_d')
        wav_dir = os.path.join(cremad_dir, 'AudioWAV')
        build_dir = os.path.join(directory, 'build')
        os.makedirs(wav_dir)
        os.makedirs(build_dir)
        records = []
        for number, (clip_id, (read, kind, tail)) in enumerate(CLIPS.items()):
            intended = clip_id.split('_')[1]
            labels, seconds = clip_labels(cremad.SENTENCES[read], trailing=kind != 'end')
            samples = data_fixtures.labels_audio([tuple(label) for label in labels], seconds, cremad.CTC_SAMPLE_RATE, number)
            if kind == 'none':
                samples = numpy.zeros_like(samples)
            elif kind == 'end':
                samples = numpy.clip(samples * 20, -1, 1)
            soundfile.write(os.path.join(wav_dir, clip_id + '.wav'), samples, cremad.CTC_SAMPLE_RATE)
            intended_labels, _ = clip_labels(cremad.SENTENCES[intended], trailing=kind != 'end')
            record = {'id': clip_id, 'seconds': round(seconds, 4), 'tokens': frontend.tokenize(cremad.SENTENCES[intended])['tokens'],
                      'labels': None if kind == 'none' else intended_labels if read == intended else labels}
            if kind != 'none':
                record.update({'confidences': [0.9, 0.2, 0.5], 'score': -1.5,
                               'greedy': 'h h h h' if kind == 'garbled' else greedy_of(cremad.SENTENCES[read], tail)})
            records.append(record)
        with open(os.path.join(wav_dir, 'notes.txt'), 'w', encoding='utf-8') as file:
            file.write('not a clip')
        with open(os.path.join(build_dir, 'alignments.jsonl'), 'w', encoding='utf-8') as file:
            file.write(''.join(json.dumps(record) + '\n' for record in records))
        with open(os.path.join(cremad_dir, 'VideoDemographics.csv'), 'w', encoding='utf-8') as file:
            file.write('ActorID,Age,Sex,Race,Ethnicity\n1001,51,Male,Caucasian,Not Hispanic\n1002,21,Female,Asian,Not Hispanic\n')
        with open(os.path.join(cremad_dir, 'finishedResponses.csv'), 'w', encoding='utf-8') as file:
            file.write('localid,sessionNums,queryType,questNum,respEmo,respLevel,dispEmo,dispVal,clipNum,clipName\n'
                       '1,1,1,1,A,80,A,80,1,1001_IEO_ANG_HI\n2,1,1,1,A,60,A,60,1,1001_IEO_ANG_HI\n'
                       '3,1,1,1,N,,N,0,1,1001_IEO_ANG_HI\n4,1,2,1,A,50,A,50,1,1001_IEO_ANG_HI\n'
                       '5,1,1,1,X,50,A,50,1,1001_IEO_ANG_HI\n6,1,1,1,H,40,H,40,1,1002_IWW_SAD_XX\n'
                       '7,1,1,1,N,,N,0,1,1002_IWW_SAD_XX\n')
        with unittest.mock.patch.object(cremad, 'CREMAD_DIR', cremad_dir), \
                unittest.mock.patch.object(cremad, 'WAV_DIR', wav_dir), \
                unittest.mock.patch.object(cremad, 'BUILD_DIR', build_dir), \
                unittest.mock.patch.object(cremad, 'ALIGNMENTS_PATH', os.path.join(build_dir, 'alignments.jsonl')), \
                unittest.mock.patch.object(cremad, 'SUMMARY_PATH', os.path.join(directory, 'cremad.json')), \
                unittest.mock.patch('multiprocessing.Pool', data_fixtures.FakePool):
            yield directory


# The realign subprocess - writes realigned.jsonl (the relabeled clip's alignment to the sentence it reads)
def fake_realign(command, **_kwargs):
    assert command[1:5] == ['-B', '-m', 'picovox_train.cremad', 'realign'], command
    with open(os.path.join(cremad.BUILD_DIR, 'realigned.jsonl'), 'w', encoding='utf-8') as file:
        for clip in command[5:]:
            clip_id, sentence = clip.split('=')
            labels, seconds = clip_labels(cremad.SENTENCES[sentence])
            file.write(json.dumps({'id': clip_id, 'sentence': sentence, 'seconds': seconds,
                                   'tokens': frontend.tokenize(cremad.SENTENCES[sentence])['tokens'], 'labels': labels,
                                   'confidences': [0.8], 'score': -1.0, 'greedy': greedy_of(cremad.SENTENCES[sentence])}) + '\n')


def build_args(out, limit=None, realign=False):
    return argparse.Namespace(out=out, limit=limit, realign=realign, jobs=1)


# A response of the fake urlopen
class FakeResponse(io.BytesIO):

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class TestDownload(unittest.TestCase):

    def test_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'sub', 'a.wav')
            with unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'RIFF1234')) as urlopen:
                self.assertEqual(cremad.download('https://x/a.wav', path), 8)
                self.assertEqual(cremad.download('https://x/a.wav', path), 0)
            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_header('User-agent'), cremad.USER_AGENT)
            with open(path, 'rb') as file:
                self.assertEqual(file.read(), b'RIFF1234')
            self.assertFalse(os.path.exists(path + '.part'))

    def test_download_retry(self):
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (_, stderr), \
                unittest.mock.patch('time.sleep') as sleep, \
                unittest.mock.patch('urllib.request.urlopen', side_effect=[TimeoutError('slow'), FakeResponse(b'ok')]):
            self.assertEqual(cremad.download('https://x/a', os.path.join(directory, 'a')), 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(stderr.getvalue(), '  retry https://x/a: slow\n')

    def test_download_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a')
            with unittest.mock.patch('urllib.request.urlopen', return_value=FakeResponse(b'version https://git-lfs.github.com/spec/v1')), \
                    self.assertRaisesRegex(IOError, 'an LFS pointer'):
                cremad.download('https://x/a', path)
            not_found = urllib.error.HTTPError('https://x/a', 404, 'Not Found', {}, None)
            with unittest.mock.patch('urllib.request.urlopen', side_effect=not_found) as urlopen, self.assertRaises(urllib.error.HTTPError):
                cremad.download('https://x/a', path)
            self.assertEqual(urlopen.call_count, 1)
            not_found.close()
            with unittest.mock.patch('urllib.request.urlopen', side_effect=ConnectionError('reset')), \
                    unittest.mock.patch('time.sleep'), helpers.captured(), self.assertRaises(ConnectionError):
                cremad.download('https://x/a', path, attempts=2)
            self.assertEqual(cremad.download('https://x/a', path, attempts=0), 0)

    def test_fetch(self):
        wavs = [f'AudioWAV/{1001 + number // 100}_IEO_ANG_{number:03d}.wav' for number in range(500)]
        tree = {'tree': [{'type': 'blob', 'path': path} for path in wavs] +
                [{'type': 'tree', 'path': 'AudioWAV'}, {'type': 'blob', 'path': 'README.md'}]}

        def urlopen(request, timeout):
            self.assertEqual(timeout, 60)
            return FakeResponse(json.dumps(tree).encode() if request.full_url == cremad.TREE_URL else b'data')
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (_, stderr), \
                unittest.mock.patch.object(cremad, 'CORPORA_DIR', directory), \
                unittest.mock.patch.object(cremad, 'CREMAD_DIR', os.path.join(directory, 'crema_d')), \
                unittest.mock.patch('urllib.request.urlopen', side_effect=urlopen):
            self.assertEqual(cremad.fetch(argparse.Namespace(jobs=4)), 0)
            present = sorted(os.listdir(os.path.join(directory, 'crema_d', 'AudioWAV')))
            metadata = os.path.exists(os.path.join(directory, 'crema_d', 'processedResults', 'summaryTable.csv'))
            with open(os.path.join(directory, 'README.md'), encoding='utf-8') as file:
                readme = file.read()
            # Some missing
            with unittest.mock.patch.object(cremad, 'download', return_value=0):
                os.remove(os.path.join(directory, 'crema_d', wavs[0]))
                self.assertEqual(cremad.fetch(argparse.Namespace(jobs=4)), 1)
        self.assertEqual(len(present), 500)
        self.assertTrue(metadata)
        self.assertIn('## crema-d', readme)
        self.assertIn('  500/500 files, 0 MB downloaded', stderr.getvalue())
        self.assertIn('crema-d: 500/500 WAV files present', stderr.getvalue())
        self.assertIn('crema-d: 499/500 WAV files present', stderr.getvalue())

    def test_write_readme(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(cremad, 'CORPORA_DIR', directory), \
                unittest.mock.patch.object(cremad, 'CREMAD_DIR', os.path.join(directory, 'crema_d')):
            path = os.path.join(directory, 'README.md')
            self.assertEqual(cremad.write_readme(), 0)
            with open(path, encoding='utf-8') as file:
                first = file.read()
            # A rewrite replaces the section, keeping the rest
            os.makedirs(os.path.join(directory, 'crema_d'))
            with open(os.path.join(directory, 'crema_d', 'a.wav'), 'wb') as file:
                file.write(b'x' * 3000000)
            with open(path, 'w', encoding='utf-8') as file:
                file.write('# Corpora\n\n## arctic\n\n' + first[first.index(cremad.README_START):] + '\n## after\n')
            cremad.write_readme()
            with open(path, encoding='utf-8') as file:
                second = file.read()
        self.assertTrue(first.startswith('# Speech corpora\n\n' + cremad.README_START + '\n## crema-d\n'))
        self.assertIn('- Size: 0 MB', first)
        self.assertTrue(second.startswith('# Corpora\n\n## arctic\n## after\n\n' + cremad.README_START))
        self.assertIn('- Size: 3 MB', second)
        self.assertEqual(second.count(cremad.README_START), 1)
        self.assertTrue(second.endswith(cremad.README_END + '\n'))


class TestClips(unittest.TestCase):

    def test_clips_and_metadata(self):
        with tiny_cremad():
            items = cremad.clips()
            actors = cremad.actors()
            ratings = cremad.voice_ratings()
        self.assertEqual([item['id'] for item in items], sorted(CLIPS))
        self.assertEqual({key: value for key, value in items[0].items() if key != 'wav'},
                         {'id': '1001_IEO_ANG_HI', 'actor': '1001', 'sentence': 'IEO', 'emotion': 'angry', 'intensity': 'high',
                          'text': 'It\'s eleven o\'clock.'})
        self.assertEqual(actors, {'1001': {'sex': 'male', 'age': 51, 'race': 'Caucasian', 'ethnicity': 'Not Hispanic'},
                                  '1002': {'sex': 'female', 'age': 21, 'race': 'Asian', 'ethnicity': 'Not Hispanic'}})
        self.assertEqual(ratings['1001_IEO_ANG_HI'], {
            'votes': {'angry': 2, 'disgusted': 0, 'fearful': 0, 'happy': 0, 'neutral': 1, 'sad': 0}, 'count': 3, 'vote': 'angry',
            'agreement': 0.667, 'level': 70.0})
        self.assertEqual(ratings['1002_IWW_SAD_XX'], {
            'votes': {'angry': 0, 'disgusted': 0, 'fearful': 0, 'happy': 1, 'neutral': 1, 'sad': 0}, 'count': 2,
            'vote': 'happy:neutral', 'agreement': 0.0, 'level': None})

    def test_clip_split(self):
        splits = [cremad.clip_split(f'{1001 + number}', 'IEO') for number in range(200)]
        self.assertEqual(set(splits), {'train', 'val', 'test'})

    def test_read_16k(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.wav')
            samples = data_fixtures.buzz(0.5, 16000)
            soundfile.write(path, samples, 16000, subtype='FLOAT')
            self.assertTrue(numpy.allclose(cremad.read_16k(path), samples.astype(numpy.float32)))
            soundfile.write(path, numpy.stack([samples[::2], samples[::2]], axis=1), 8000)
            resampled = cremad.read_16k(path)
        self.assertEqual(len(resampled), 8000)
        self.assertTrue(resampled.flags['C_CONTIGUOUS'])
        self.assertEqual(resampled.dtype, numpy.float64)


class TestAlignment(unittest.TestCase):

    def test_ctc_viterbi(self):
        self.assertIsNone(cremad.ctc_viterbi(numpy.zeros((2, 3)), numpy.zeros(2)))
        # Two positions: blank, A, A, blank, B, blank - and ending on a phone
        low, high = numpy.log(0.05), numpy.log(0.9)
        positions = numpy.full((6, 2), low)
        positions[1:3, 0] = high
        positions[4, 1] = high
        blank = numpy.full(6, low)
        blank[[0, 3, 5]] = high
        self.assertEqual(cremad.ctc_viterbi(positions, blank), [(1, 2), (4, 4)])
        positions[5, 1] = numpy.log(0.99)
        blank[5] = low
        self.assertEqual(cremad.ctc_viterbi(positions, blank), [(1, 2), (4, 5)])

    def test_phone_positions(self):
        tokens = frontend.tokenize('Call me.')['tokens']
        self.assertEqual(cremad.phone_positions(tokens), [(1, 'K'), (2, 'AO'), (3, 'L'), (5, 'M'), (6, 'IY')])

    # Log posteriors [frames, vocab] with each phone of tokens in its run of frames (blank elsewhere)
    @staticmethod
    def posteriors(tokens, runs, frames):
        values = numpy.full((frames, len(VOCAB)), -20.0)
        values[:, 0] = 0.0
        for (_, key), (first, last) in zip(cremad.phone_positions(tokens), runs):
            values[first:last + 1, 0] = -20.0
            values[first:last + 1, VOCAB[cremad.IPA_TOKENS[key][0]]] = 0.0
        return values

    def test_ctc_labels(self):
        tokens = frontend.tokenize('Call me.')['tokens']
        rate = cremad.CTC_SAMPLE_RATE
        # Speech from 0.1 s to 0.9 s of 1 s; the phones' runs from 0.2 s - the edges extend to the speech
        samples = numpy.zeros(rate)
        samples[int(0.1 * rate):int(0.9 * rate)] = data_fixtures.buzz(0.8, rate)
        runs = [(10, 13), (15, 20), (22, 25), (28, 31), (33, 36)]
        labels, confidences, score = cremad.ctc_labels(tokens, self.posteriors(tokens, runs, 50), VOCAB, samples)
        self.assertEqual([label for label, _, _ in labels], ['pau', 'K', 'AO1', 'L', 'M', 'IY0', 'pau'])
        self.assertAlmostEqual(labels[1][1], 0.1, delta=0.011)
        self.assertAlmostEqual(labels[-2][2], 0.9, delta=0.02)
        self.assertAlmostEqual(labels[2][1], (14 + 15) / 2 * 0.02 - 0.005)
        for previous, following in zip(labels, labels[1:]):
            self.assertAlmostEqual(previous[2], following[1])
        self.assertTrue(numpy.allclose(confidences, 1.0))
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_ctc_labels_edges(self):
        # Speech to the end; the first run at frame 0; a phone's start pushed past its predecessor's (5 ms apart)
        tokens = frontend.tokenize('Call me.')['tokens']
        rate = cremad.CTC_SAMPLE_RATE
        samples = data_fixtures.buzz(0.5, rate)
        runs = [(0, 3), (4, 4), (5, 9), (10, 15), (16, 24)]
        labels, _, _ = cremad.ctc_labels(tokens, self.posteriors(tokens, runs, 25), VOCAB, samples)
        self.assertEqual([label for label, _, _ in labels], ['K', 'AO1', 'L', 'M', 'IY0'])
        self.assertEqual(labels[0][1], 0.0)
        self.assertEqual(labels[-1][2], 0.5)
        # Speech ending before the last run (its end stays), a trailing pause
        samples[int(0.3 * rate):] = 0.0
        labels, _, _ = cremad.ctc_labels(tokens, self.posteriors(tokens, runs, 25), VOCAB, samples)
        self.assertEqual(labels[-1][0], 'pau')
        self.assertAlmostEqual(labels[-2][2], 25 * 0.02 - 0.005)
        # Too few frames
        self.assertIsNone(cremad.ctc_labels(tokens, self.posteriors(tokens, [], 3), VOCAB, samples))

    def test_greedy_phone_error(self):
        tokens = frontend.tokenize(cremad.SENTENCES['DFA'])['tokens']
        count = len(cremad.phone_positions(tokens))
        self.assertEqual(cremad.greedy_phone_error(greedy_of(cremad.SENTENCES['DFA']), tokens), (0.0, 0))
        self.assertEqual(cremad.greedy_phone_error(greedy_of(cremad.SENTENCES['DFA'], tail=4), tokens), (4 / count, 4))
        error, tail = cremad.greedy_phone_error('d oʊ n t zz ɡ ɛ t', tokens)
        self.assertGreater(error, 0)
        self.assertEqual((count, tail), (15, 6))
        self.assertEqual(cremad.greedy_phone_error('', tokens), (1.0, count))

    def test_greedy_decode(self):
        posteriors = numpy.full((7, len(VOCAB)), -10.0)
        for frame, token in enumerate(['<pad>', 'k', 'k', '<pad>', 'k', '<s>', 'ɔː']):
            posteriors[frame, VOCAB[token]] = 0.0
        self.assertEqual(cremad.greedy_decode(posteriors, VOCAB), 'k k ɔː')

    def test_label_durations(self):
        tokens = frontend.tokenize('Call me.')['tokens']
        labels, seconds = clip_labels('Call me.')
        samples = data_fixtures.labels_audio([tuple(label) for label in labels], seconds, 16000)
        durations, report = cremad.label_durations(tokens, [tuple(label) for label in labels], samples, 16000, {})
        self.assertAlmostEqual(sum(durations), seconds)
        self.assertEqual(report['exact'], 5)
        self.assertIn('pausesRefined', report)

    def test_load_ctc(self):
        with fake_recognizer() as (model, hub):
            self.assertEqual(cremad.load_ctc(), (model, VOCAB, 'cpu'))
            self.assertEqual(model.device, 'cpu')
            self.assertEqual(os.environ['HF_HOME'], cremad.HF_HOME)
            self.assertEqual(cremad.load_ctc('mps')[2], 'mps')
        self.assertEqual(hub.snapshot_download.call_args.args, (cremad.CTC_REPO,))

    def test_ctc_log_posteriors(self):
        model = FakeCtc()
        samples = [data_fixtures.buzz(seconds, 16000) for seconds in (0.5, 0.2, 0.8)]
        results = cremad.ctc_log_posteriors(model, 'cpu', samples, max_batch_seconds=1.0)
        self.assertEqual([len(result) for result in results], [25, 10, 40])
        self.assertTrue(numpy.allclose(numpy.exp(results[0]).sum(axis=1), 1, atol=1e-5))
        self.assertEqual(model.calls, [(2, 8000), (1, 12800)])
        self.assertEqual(cremad.ctc_log_posteriors(model, 'cpu', []), [])

    def test_align_command(self):
        with tiny_cremad() as directory, fake_recognizer(), helpers.captured() as (_, stderr):
            # A clip too short to align
            soundfile.write(os.path.join(directory, 'crema_d', 'AudioWAV', '1002_DFA_HAP_XX.wav'), numpy.zeros(3200), 16000)
            self.assertEqual(cremad.align_command(argparse.Namespace(limit=None, device=None)), 0)
            with open(cremad.ALIGNMENTS_PATH, encoding='utf-8') as file:
                records = [json.loads(line) for line in file]
            self.assertEqual(cremad.align_command(argparse.Namespace(limit=2, device='cpu')), 0)
            with open(cremad.ALIGNMENTS_PATH + '.sample', encoding='utf-8') as file:
                sample = [json.loads(line) for line in file]
        self.assertEqual(len(records), 8)
        by_id = {record['id']: record for record in records}
        self.assertIsNone(by_id['1002_DFA_HAP_XX']['labels'])
        aligned = [record for record in records if record['labels']]
        self.assertTrue(aligned)
        for record in aligned:
            self.assertEqual(record['tokens'], frontend.tokenize(cremad.SENTENCES[record['id'].split('_')[1]])['tokens'])
            self.assertEqual(set(record), {'id', 'seconds', 'tokens', 'labels', 'confidences', 'score', 'greedy'})
        self.assertEqual(len(sample), 2)
        self.assertIn('aligned', stderr.getvalue())

    def test_realign_command(self):
        with tiny_cremad(), fake_recognizer():
            self.assertEqual(cremad.realign_command(argparse.Namespace(ids=['1002_TIE_NEU_XX=DFA'], device=None)), 0)
            with open(os.path.join(cremad.BUILD_DIR, 'realigned.jsonl'), encoding='utf-8') as file:
                records = [json.loads(line) for line in file]
        self.assertEqual([(record['id'], record['sentence']) for record in records], [('1002_TIE_NEU_XX', 'DFA')])
        self.assertEqual(records[0]['tokens'], frontend.tokenize(cremad.SENTENCES['DFA'])['tokens'])


class TestBuild(unittest.TestCase):

    # The fake acoustic analysis
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS))

    # The build with a phone of one clip given no frames
    def run_build(self, args):
        real = cremad.process_clip

        def process_clip(task):
            result = real(task)
            if result['item']['id'] == '1001_WSI_DIS_XX':
                index = next(index for index, token in enumerate(result['tokens']) if token not in frontend.BOUNDARIES)
                result['frames'][index + 1] += result['frames'][index]
                result['frames'][index] = 0
            return result
        with unittest.mock.patch.object(cremad, 'process_clip', side_effect=process_clip), \
                unittest.mock.patch('subprocess.run', side_effect=fake_realign) as run, helpers.captured() as (stdout, stderr):
            self.assertEqual(cremad.build(args), 0)
        return run, stdout.getvalue(), stderr.getvalue()

    def test_build(self):
        with tiny_cremad() as directory:
            out = os.path.join(directory, 'data')
            run, stdout, stderr = self.run_build(build_args(out))
            corpus = data.ArcticData(out)
            with open(os.path.join(out, 'cremad.json'), encoding='utf-8') as file:
                summary = json.load(file)
            with open(os.path.join(out, 'alignment.jsonl'), encoding='utf-8') as file:
                alignment = [json.loads(line) for line in file]
            # The realigned clips are kept (no subprocess) unless --realign
            run_again, _, _ = self.run_build(build_args(out))
            run_realign, _, _ = self.run_build(build_args(out, realign=True))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], os.path.join(cremad.ROOT, 'build', 'venv-listeners', 'bin', 'python'))
        self.assertEqual(run.call_args.args[0][5:], ['1002_TIE_NEU_XX=DFA'])
        self.assertEqual(run.call_args.kwargs['env']['PYTHONPATH'], cremad.SOURCE_DIR)
        self.assertEqual(run_again.call_count, 0)
        self.assertEqual(run_realign.call_count, 1)
        self.assertIn('1002_TIE_NEU_XX: reads DFA', stderr)
        index = corpus.index
        reasons = {entry['id']: entry['reason'] for entry in index['utterances']}
        self.assertEqual(reasons['cremad/1001_IEO_ANG_HI'], '')
        self.assertEqual(reasons['cremad/1002_TIE_NEU_XX'], '')
        self.assertEqual(reasons['cremad/1001_WSI_DIS_XX'], 'a phone with no frames')
        self.assertEqual(reasons['cremad/1002_IWW_SAD_XX'], 'cut short (5 final phones missing)')
        self.assertRegex(reasons['cremad/1002_TAI_ANG_XX'],
                         r'^cut short \(\d+ final phones missing\); garbled \(greedy phone error 1\.00 > 0\.6\)$')
        self.assertRegex(reasons['cremad/1002_MTI_FEA_XX'], r'^cut short \(3 final phones missing\); clipped \(\d+\.\d\d% of samples')
        self.assertNotIn('cremad/1001_IOM_HAP_XX', reasons)
        entries = {entry['id']: entry for entry in index['utterances']}
        relabeled = entries['cremad/1002_TIE_NEU_XX']
        self.assertEqual((relabeled['sentence'], relabeled['text'], relabeled['relabeled']), ('DFA', cremad.SENTENCES['DFA'], True))
        angry = entries['cremad/1001_IEO_ANG_HI']
        self.assertEqual((angry['emotion'], angry['emotionId'], angry['intensity'], angry['intensityId']), ('angry', 0, 'high', 2))
        self.assertEqual((angry['voiceVote'], angry['voiceAgreement'], angry['voiceLevel'], angry['voiceRatings']),
                         ('angry', 0.667, 70.0, 3))
        self.assertEqual(entries['cremad/1002_TAI_ANG_XX']['voiceRatings'], 0)
        for entry in index['utterances']:
            arrays = corpus.arrays(entry)
            self.assertEqual(arrays['tokens'].tolist(), frontend.tokenize(entry['text'])['ids'])
            self.assertEqual(int(arrays['durations'].sum()), entry['frameCount'])
        speakers = index['speakers']
        self.assertEqual(list(speakers), ['cremad-1001', 'cremad-1002'])
        self.assertEqual(sorted(speakers['cremad-1001']['pitchRanges']), ['angry', 'disgusted'])
        self.assertEqual(sorted(speakers['cremad-1002']['pitchRanges']), ['angry', 'fearful', 'neutral', 'sad'])
        self.assertEqual((speakers['cremad-1002']['sex'], speakers['cremad-1002']['formantCeiling']), ('female', 5500))
        self.assertIsNone(speakers['cremad-1001']['accent'])
        self.assertEqual(index['vocab'], frontend.vocabulary())
        self.assertEqual(len(alignment), 6)
        self.assertEqual(summary['clips'], 6)
        self.assertEqual(summary['included'], 2)
        self.assertEqual(summary['relabeled'], ['cremad/1002_TIE_NEU_XX'])
        self.assertEqual(summary['excluded'], {'a phone with no frames': 1, 'cut short': 3, 'garbled': 1, 'clipped': 1})
        self.assertEqual(summary['counts']['emotion'], {'angry': 1, 'neutral': 1})
        self.assertEqual((summary['actors']['cremad-1001']['included'], summary['actors']['cremad-1002']['included']), (1, 1))
        self.assertIn('Built ', stdout)
        self.assertIn('6 clips (2 included)', stdout)

    def test_build_default_and_limit(self):
        # The default directory writes the committed summary; --limit samples the clips (none relabeled here)
        with tiny_cremad() as directory:
            default = os.path.join(directory, 'build', 'nn', 'data', 'cremad')
            with unittest.mock.patch.object(cremad, 'ROOT', directory), unittest.mock.patch.object(cremad, 'DATA_DIR', default):
                run, _, _ = self.run_build(build_args(None, limit=3))
            index = data.ArcticData(default).index
            committed = os.path.exists(cremad.SUMMARY_PATH)
        self.assertTrue(committed)
        self.assertEqual(run.call_count, 0)
        self.assertEqual([entry['id'] for entry in index['utterances']], ['cremad/1001_IEO_ANG_HI', 'cremad/1001_WSI_DIS_XX',
                                                                          'cremad/1002_MTI_FEA_XX'])

    def test_build_stale_alignment(self):
        with tiny_cremad():
            with open(cremad.ALIGNMENTS_PATH, encoding='utf-8') as file:
                records = [json.loads(line) for line in file]
            records[0]['tokens'] = frontend.tokenize('Call me.')['tokens']
            with open(cremad.ALIGNMENTS_PATH, 'w', encoding='utf-8') as file:
                file.write(''.join(json.dumps(record) + '\n' for record in records))
            with tempfile.TemporaryDirectory() as out, self.assertRaisesRegex(ValueError, 'differ from its alignment'):
                self.run_build(build_args(out))


class TestMain(unittest.TestCase):

    def test_log(self):
        with helpers.captured() as (stdout, stderr):
            cremad.log('hello')
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ('', 'hello\n'))

    def test_main(self):
        handlers = {name: unittest.mock.patch.object(cremad, name, return_value=None) for name in
                    ('build', 'realign_command', 'fetch', 'align_command')}
        with contextlib.ExitStack() as stack:
            mocks = {name: stack.enter_context(patch) for name, patch in handlers.items()}
            self.assertEqual(cremad.main(['fetch', '-j', '2']), 0)
            self.assertEqual(cremad.main(['align', '--limit', '5', '--device', 'cpu']), 0)
            self.assertEqual(cremad.main(['build', '--realign', '--out', 'x', '-j', '3']), 0)
            self.assertEqual(cremad.main(['realign', 'a=DFA', 'b=TIE']), 0)
        self.assertEqual(mocks['fetch'].call_args.args[0].jobs, 2)
        self.assertEqual(mocks['align_command'].call_args.args[0], argparse.Namespace(command='align', limit=5, device='cpu'))
        self.assertEqual(mocks['build'].call_args.args[0], argparse.Namespace(command='build', limit=None, realign=True, out='x', jobs=3))
        self.assertEqual(mocks['realign_command'].call_args.args[0].ids, ['a=DFA', 'b=TIE'])

    def test_module_main(self):
        with unittest.mock.patch.object(sys, 'argv', ['cremad']), self.assertRaises(SystemExit), helpers.captured(), \
                warnings.catch_warnings():
            # (runpy warns that the module is already imported - it runs a fresh copy as __main__)
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.cremad', run_name='__main__', alter_sys=False)
