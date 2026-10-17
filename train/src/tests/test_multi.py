# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import argparse
import contextlib
import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy

from picovox_train import data, frontend, multi, readings
from tests import data_fixtures, helpers


# A speaker table row (data/nn/speakers.json's)
def table_row(row_id, key, accent=True):
    return {'id': row_id, 'key': key, 'accent': {'top': 'us', 'us': 0.9, 'canada': 0.05, 'votes': {'us': 9}} if accent else None,
            'pitchFloor': 75.0, 'pitchCeiling': 400.0, 'formantCeiling': 5000, 'erF3Ratio': 0.6 if accent else None,
            'utmos': 4.0 + row_id}


ARCTIC_TEXTS = {'arctic_a0001': 'Call me Ishmael.', 'arctic_a0002': 'It was the best of times.', 'arctic_a0003': None}
LIBRITTS_TEXTS = {'84_121123_000007_000001': 'We went home.', '84_121123_000008_000000': 'Is it cold?',
                  '84_121123_000009_000000': 'A dark night.'}


# A tiny pool in a temporary directory - an ARCTIC speaker (slt) and a LibriTTS-R speaker (84, one utterance without
# labels), a speaker table, and the module constants patched to them
@contextlib.contextmanager
def tiny_pool():
    with tempfile.TemporaryDirectory() as directory:
        arctic_dir = os.path.join(directory, 'arctic')
        libritts_dir = os.path.join(directory, 'LibriTTS_R')
        label_dir = os.path.join(directory, 'libritts_label', 'lab', 'phone')
        data_fixtures.make_arctic(arctic_dir, 'slt', ARCTIC_TEXTS)
        data_fixtures.make_libritts(libritts_dir, label_dir, 'dev-clean', '84', '121123', LIBRITTS_TEXTS,
                                    without_labels=('84_121123_000009_000000',))
        with open(os.path.join(libritts_dir, 'speakers.tsv'), 'w', encoding='utf-8') as file:
            file.write('READER\tGENDER\tSUBSET\tNAME\n84\tF\tdev-clean\tChristie Nowak\n7976\tM\n')
        pool_path = os.path.join(directory, 'speakers.json')
        multi.write_json(pool_path, {'speakers': [table_row(1, 'libritts-84', accent=False), table_row(0, 'arctic-slt')]})
        committed = os.path.join(directory, 'committed', 'speakers.json')
        multi.write_json(committed, {'speakers': [{'key': 'arctic-slt', 'utmos': 4.25}, {'key': 'libritts-84'}]})
        with unittest.mock.patch.object(data, 'ARCTIC_DIR', arctic_dir), \
                unittest.mock.patch.object(multi, 'LIBRITTS_DIR', libritts_dir), \
                unittest.mock.patch.object(multi, 'LABEL_DIR', label_dir), \
                unittest.mock.patch.object(multi, 'ARCTIC_SPEAKERS', {'slt': 'female'}), \
                unittest.mock.patch.object(multi, 'SPEAKERS_PATH', committed), \
                unittest.mock.patch.object(multi, 'SPLIT_PATH', os.path.join(directory, 'committed', 'multi-split.json')), \
                unittest.mock.patch.object(multi.multiprocessing, 'Pool', data_fixtures.FakePool):
            yield directory, pool_path


def build_args(out, pool, speakers=None, limit=None):
    return argparse.Namespace(out=out, pool=pool, speakers=speakers, limit=limit, jobs=1)


class TestCorpus(unittest.TestCase):

    def test_libritts_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, 'speakers.tsv'), 'w', encoding='utf-8') as file:
                file.write('READER\tGENDER\tSUBSET\tNAME\n14\tF\ttrain-clean-360\tKristin LeMoine\n16\tM\tdev-clean\n17\tX\n')
            with unittest.mock.patch.object(multi, 'LIBRITTS_DIR', directory):
                self.assertEqual(multi.libritts_metadata(), {
                    '14': {'sex': 'female', 'subset': 'train-clean-360', 'reader': 'Kristin LeMoine'},
                    '16': {'sex': 'male', 'subset': 'dev-clean', 'reader': ''},
                    '17': {'sex': None, 'subset': None, 'reader': ''}
                })

    def test_libritts_utterances(self):
        with tempfile.TemporaryDirectory() as directory:
            chapter = data_fixtures.make_libritts(directory, os.path.join(directory, 'lab', 'phone'), 'dev-clean', '84', '121123',
                                                  {'84_1': 'We went home.'})
            # (a WAV without text, a file that isn't a WAV, and a file beside the chapters are skipped)
            with open(os.path.join(chapter, '84_2.wav'), 'wb') as file:
                file.write(b'')
            with open(os.path.join(directory, 'dev-clean', '84', 'README'), 'w', encoding='utf-8') as file:
                file.write('x')
            with unittest.mock.patch.object(multi, 'LIBRITTS_DIR', directory), \
                    unittest.mock.patch.object(multi, 'LABEL_DIR', os.path.join(directory, 'lab', 'phone')):
                utterances = multi.libritts_utterances('dev-clean', '84')
        self.assertEqual(utterances, [{
            'speaker': 'libritts-84', 'sentence': '84_1', 'wav': os.path.join(chapter, '84_1.wav'),
            'lab': os.path.join(directory, 'lab', 'phone', 'dev-clean', '84', '121123', '84_1.lab'), 'labelFormat': 'mfa',
            'text': 'We went home.'
        }])

    def test_candidate_speakers(self):
        with tiny_pool() as (directory, _):
            os.makedirs(os.path.join(directory, 'LibriTTS_R', 'dev-clean', '7976', '1'))
            os.makedirs(os.path.join(directory, 'LibriTTS_R', 'dev-clean', 'notes'))
            speakers = multi.candidate_speakers()
        self.assertEqual(list(speakers), ['arctic-slt', 'libritts-84', 'libritts-7976'])
        slt = speakers['arctic-slt']
        self.assertEqual([utterance['sentence'] for utterance in slt['utterances']], ['arctic_a0001', 'arctic_a0002'])
        self.assertEqual(slt['utterances'][0]['speaker'], 'arctic-slt')
        self.assertEqual({key: value for key, value in slt.items() if key != 'utterances'},
                         {'key': 'arctic-slt', 'corpus': 'arctic', 'source': 'CMU ARCTIC 0.95', 'speaker': 'slt', 'reader': None,
                          'sex': 'female', 'sourceSampleRate': 16000, 'bandwidth': 8000})
        reader = speakers['libritts-84']
        self.assertEqual((reader['corpus'], reader['source'], reader['reader'], reader['sex'], reader['sexCorrected']),
                         ('libritts-dev-clean', 'LibriTTS-R dev-clean', 'Christie Nowak', 'female', False))
        self.assertEqual(len(reader['utterances']), 3)
        self.assertEqual((speakers['libritts-7976']['sex'], speakers['libritts-7976']['sexCorrected']), ('female', True))

    def test_utterance_split(self):
        self.assertEqual(multi.utterance_split('arctic', 'arctic_a0001'), data.sentence_split('arctic_a0001'))
        splits = [multi.utterance_split('libritts-dev-clean', f'84_{number}') for number in range(300)]
        self.assertEqual(set(splits), {'train', 'val', 'test'})


class TestPool(unittest.TestCase):

    def test_table_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'speakers.json')
            multi.write_json(path, {'speakers': [table_row(1, 'libritts-84', accent=False), table_row(0, 'arctic-slt')]})
            kept, rows = multi.table_pool(path)
        self.assertEqual(kept, ['arctic-slt', 'libritts-84'])
        self.assertEqual(rows[0], {'key': 'arctic-slt', 'keep': True,
                                   'accent': {'top': 'us', 'posterior': {'us': 0.9, 'canada': 0.05}, 'votes': {'us': 9}},
                                   'settings': {'pitchFloor': 75.0, 'pitchCeiling': 400.0, 'formantCeiling': 5000, 'erF3Ratio': 0.6}})
        self.assertIsNone(rows[1]['accent'])

    def test_table_paths(self):
        default = os.path.join(multi.ROOT, 'build', 'nn', 'data', 'multi')
        self.assertEqual(multi.table_paths(default + os.sep), (multi.SPEAKERS_PATH, multi.SPLIT_PATH))
        self.assertEqual(multi.table_paths('/tmp/out'), ('/tmp/out/speakers.json', '/tmp/out/multi-split.json'))


class TestBuild(unittest.TestCase):

    def test_build(self):
        with tiny_pool() as (directory, pool_path), helpers.captured() as (stdout, _):
            out = os.path.join(directory, 'multi')
            multi.build_command(build_args(out, pool_path))
            corpus = data.ArcticData(out)
            with open(os.path.join(out, 'speakers.json'), encoding='utf-8') as file:
                table = json.load(file)
            with open(os.path.join(out, 'multi-split.json'), encoding='utf-8') as file:
                split = json.load(file)
            with open(os.path.join(out, 'failures.json'), encoding='utf-8') as file:
                failures = json.load(file)
            with open(os.path.join(out, 'alignment.jsonl'), encoding='utf-8') as file:
                alignment = [json.loads(line) for line in file]
            raw_left = [name for name in os.listdir(out) if name.endswith('.raw')]
        index = corpus.index
        self.assertEqual(raw_left, [])
        self.assertEqual(failures, [])
        self.assertEqual(index['speakers'], {
            'arctic-slt': {'index': 0, 'sex': 'female', 'corpus': 'CMU ARCTIC 0.95', 'sourceSampleRate': 16000, 'bandwidth': 8000,
                           'pitchFloor': 75.0, 'pitchCeiling': 400.0, 'formantCeiling': 5000},
            'libritts-84': {'index': 1, 'sex': 'female', 'corpus': 'LibriTTS-R dev-clean', 'sourceSampleRate': 24000,
                            'bandwidth': 12000, 'pitchFloor': 75.0, 'pitchCeiling': 400.0, 'formantCeiling': 5000}
        })
        self.assertEqual(index['vocab'], frontend.vocabulary())
        self.assertEqual(index['readings']['utterances'], 2)
        entries = index['utterances']
        self.assertEqual([entry['id'] for entry in entries], [
            'arctic-slt/arctic_a0001', 'arctic-slt/arctic_a0002', 'libritts-84/84_121123_000007_000001',
            'libritts-84/84_121123_000008_000000'])
        self.assertTrue(all(entry['include'] for entry in entries), entries)
        for entry in entries:
            arrays = corpus.arrays(entry)
            self.assertEqual(arrays['tokens'].tolist(), frontend.tokenize(entry['text'])['ids'])
            self.assertEqual(int(arrays['durations'].sum()), entry['frameCount'])
            self.assertEqual(entry['split'], multi.utterance_split('arctic' if entry['speaker'] == 'arctic-slt' else 'libritts',
                                                                   entry['sentence']))
        self.assertEqual(len(alignment), 4)
        self.assertEqual([row['key'] for row in table['speakers']], ['arctic-slt', 'libritts-84'])
        self.assertEqual(table['usThreshold'], multi.US_THRESHOLD)
        slt, reader = table['speakers']
        self.assertEqual(slt['utmos'], 4.25)
        self.assertNotIn('utmos', reader)
        self.assertEqual(reader['utterances']['noLabels'], 1)
        self.assertIsNone(reader['accent'])
        self.assertEqual(slt['accent'], {'top': 'us', 'us': 0.9, 'canada': 0.05, 'votes': {'us': 9}})
        self.assertAlmostEqual(slt['medianF0'], 150.0, delta=3)
        self.assertEqual(slt['phoneMatch'], 1.0)
        self.assertEqual(set(split), {'description', 'val', 'test'})
        self.assertIn('4 utterances (4 included)', stdout.getvalue())

    def test_build_limited(self):
        # --speakers and --limit - a quick test, no speaker table
        with tiny_pool() as (directory, pool_path), helpers.captured() as (stdout, _):
            out = os.path.join(directory, 'multi')
            multi.build_command(build_args(out, pool_path, speakers=['arctic-slt'], limit=1))
            index = data.ArcticData(out).index
            tables = os.path.exists(os.path.join(out, 'speakers.json'))
        self.assertEqual(list(index['speakers']), ['arctic-slt'])
        self.assertEqual(len(index['utterances']), 1)
        self.assertFalse(tables)
        self.assertIn('1 speakers, 1 utterances', stdout.getvalue())

    def test_build_results(self):
        # The exclusion reasons, failures, the progress line, and the heteronym check over its limit
        tokens = frontend.tokenize('We went home.')['tokens']
        phones = [index for index, token in enumerate(tokens) if token not in frontend.BOUNDARIES]
        self.assertEqual(len(phones), 9)

        def result(speaker, sentence, frames=None, exact=8, status=None):
            frames = frames or [2] * len(tokens)
            report = {'exact': exact, 'near': 0, 'phones': 8, 'labelPhones': 8, 'substituted': 0, 'deleted': 0, 'inserted': 0,
                      'pauseAtBoundary': 0, 'pauseInsideWord': 0, 'pausesRefined': 0, 'cost': 0.0, 'edits': ['K>G'],
                      'worstWord': None, 'worstWordMatch': 1.0, 'ids': frontend.tokenize('We went home.')['ids'],
                      'status': status or ['exact'] * len(tokens)}
            return {'utterance': {'speaker': speaker, 'sentence': sentence, 'text': 'We went home.',
                                  'readings': {3: ['W', 'EH1', 'N', 'T']}, 'alignedReadings': [{}]},
                    'tokens': tokens, 'seconds': [0.02] * len(tokens), 'frames': frames, 'cutStart': 0.0, 'report': report,
                    'audio': numpy.zeros(sum(frames) * data.HOP, dtype=numpy.int16),
                    'tracks': numpy.zeros((sum(frames), len(data.CHANNELS)))}
        no_frames = [2] * len(tokens)
        no_frames[phones[0]] = 0
        home = frontend.tokenize('We went home.')['words'][2]
        bad_word = ['exact'] * len(tokens)
        bad_word[home['start']:home['end']] = ['deleted'] * (home['end'] - home['start'])
        results = [result('arctic-slt', 'a1'), result('arctic-slt', 'a2', exact=2), result('libritts-84', 'b1', status=bad_word),
                   result('libritts-84', 'b2', frames=no_frames)]
        results.extend({'utterance': {'speaker': 'libritts-84', 'sentence': f'f{number}'}, 'error': 'ValueError: no'}
                       for number in range(1996))

        def check_built(_utterance, _tokens, totals):
            totals['tokens'] += 1
            totals['disagree'] += 1
        with tiny_pool() as (directory, pool_path), helpers.captured() as (stdout, _), \
                unittest.mock.patch.object(data_fixtures.FakePool, 'results', results), \
                unittest.mock.patch.object(readings, 'check_built', side_effect=check_built) as checked:
            out = os.path.join(directory, 'multi')
            multi.build_command(build_args(out, pool_path, limit=5))
            index = data.ArcticData(out).index
            with open(os.path.join(out, 'failures.json'), encoding='utf-8') as file:
                failures = json.load(file)
            with open(os.path.join(out, 'alignment.jsonl'), encoding='utf-8') as file:
                alignment = [json.loads(line) for line in file]
        entries = index['utterances']
        self.assertEqual([entry['reason'] for entry in entries], [
            '', 'phone match 0.250 < 0.85', 'word "home" match 0.00 < 0.5', 'a phone with no frames'])
        self.assertEqual(entries[1]['worstWord'], None)
        self.assertEqual(entries[2]['worstWord'], 'home')
        self.assertEqual(checked.call_count, 1)
        self.assertEqual(len(failures), 1996)
        self.assertEqual(failures[0], ['libritts-84', 'f0', 'ValueError: no'])
        self.assertEqual(alignment[0]['readings'], {'3': 'W EH1 N T'})
        output = stdout.getvalue()
        self.assertIn('  2000/', output)
        self.assertIn('over the 1% limit', output)

    def test_worst_content_word(self):
        tokenized = frontend.tokenize('The cat sat on the mat.')
        status = ['exact'] * len(tokenized['tokens'])
        for word in tokenized['words']:
            if word['text'] in ('the', 'sat'):
                status[word['start']] = 'substituted'
        status[tokenized['words'][1]['start']] = 'exact+inserted'
        # ("the" exempt; "sat" 2/3; "cat" 2/4 - its inserted label phone counts against it)
        self.assertEqual(multi.worst_content_word('The cat sat on the mat.', status), ('cat', 0.5))
        self.assertEqual(multi.worst_content_word('The cat.', ['exact'] * 8), (None, 1.0))

    def test_safe_process_utterance(self):
        utterance = {'text': 'x', 'wav': '/nonexistent.wav', 'lab': '/nonexistent.lab'}
        result = multi.safe_process_utterance((utterance, {}, (75, 400, 5000)))
        self.assertIs(result['utterance'], utterance)
        self.assertTrue(result['error'].startswith('LibsndfileError: ') or result['error'].startswith('FileNotFoundError'),
                        result['error'])
        with unittest.mock.patch.object(data, 'process_utterance', return_value={'tokens': []}):
            self.assertEqual(multi.safe_process_utterance((utterance, {}, None)), {'tokens': []})

    def test_raw_to_npy(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = os.path.join(directory, 'a.raw')
            values = numpy.arange(12, dtype=numpy.float32)
            values.tofile(raw)
            multi.raw_to_npy(raw, os.path.join(directory, 'a.npy'), numpy.float32, (6, 2))
            self.assertFalse(os.path.exists(raw))
            self.assertEqual(numpy.load(os.path.join(directory, 'a.npy')).tolist(), values.reshape(6, 2).tolist())

    def test_write_speakers(self):
        # The table's previous scores kept (from the table itself when it exists); a speaker with no included utterances
        with tempfile.TemporaryDirectory() as directory:
            helpers.write_corpus(os.path.join(directory, 'corpus'), speakers={'arctic-slt': {'sex': 'female', 'index': 0}}, count=1)
            corpus = data.ArcticData(os.path.join(directory, 'corpus'))
            for entry in corpus.index['utterances']:
                entry.update({'include': False, 'match': {'exact': 3, 'phones': 4}})
            speakers = {'arctic-slt': {'source': 'CMU ARCTIC 0.95', 'speaker': 'slt', 'reader': None, 'sex': 'female'}}
            row_of = {'arctic-slt': {'accent': None, 'settings': None}}
            path = os.path.join(directory, 'speakers.json')
            multi.write_json(path, {'speakers': [{'key': 'arctic-slt', 'utmos': 3.5}]})
            multi.write_speakers(corpus, speakers, row_of, {'arctic-slt': multi.collections.Counter({'K>G': 2})}, {}, path)
            table = multi.load_json(path, None)
        row = table['speakers'][0]
        self.assertEqual(row['utmos'], 3.5)
        self.assertIsNone(row['medianF0'])
        self.assertIsNone(row['erF3Ratio'])
        self.assertEqual(row['hours'], 0.0)
        self.assertEqual(row['phoneMatch'], 0.75)
        self.assertEqual(row['commonestEdits'], ['K>G 2'])
        self.assertEqual(row['utterances'], {'all': 1, 'included': 0, 'noLabels': 0, 'train': 0, 'val': 0, 'test': 0})

    def test_write_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'split.json')
            multi.write_split([{'speaker': 'a', 'sentence': 's1', 'split': 'val', 'include': True},
                               {'speaker': 'a', 'sentence': 's2', 'split': 'test', 'include': True},
                               {'speaker': 'a', 'sentence': 's3', 'split': 'test', 'include': False},
                               {'speaker': 'b', 'sentence': 's4', 'split': 'train', 'include': True}], path)
            split = multi.load_json(path, None)
        self.assertEqual((split['val'], split['test']), ({'a': ['s1']}, {'a': ['s2']}))
        self.assertIn('2% test, 3% validation', split['description'])


class TestVerifyTokens(unittest.TestCase):

    def test_verify_tokens(self):
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (stdout, _):
            helpers.write_corpus(directory)
            with unittest.mock.patch.object(multi, 'OUT_DIR', directory):
                status = multi.verify_tokens_command(argparse.Namespace(out=None, jobs=1))
        self.assertEqual(status, 0)
        texts = len(set(helpers.TEXTS[:5]))
        self.assertRegex(stdout.getvalue(), rf'verify-tokens: {texts} pool texts, \d+ tokens: {texts} identical, 0 different\n')

    def test_verify_tokens_mismatch(self):
        with tempfile.TemporaryDirectory() as directory, helpers.captured() as (stdout, _):
            helpers.write_corpus(directory, speakers={f'arctic-{number}': {'index': number} for number in range(3)})
            with unittest.mock.patch.object(data, 'js_tokenize', side_effect=lambda texts: [{'ids': [1, 2]} for _ in texts]):
                status = multi.verify_tokens_command(argparse.Namespace(out=directory, jobs=1))
        self.assertEqual(status, 1)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(sum(line.startswith('MISMATCH: ') for line in lines), 5)
        self.assertRegex(lines[-1], r'verify-tokens: (\d+) pool texts, \d+ tokens: 0 identical, \1 different')


class TestMain(unittest.TestCase):

    def test_load_json(self):
        self.assertEqual(multi.load_json('/nonexistent/x.json', {'a': 1}), {'a': 1})

    def test_main(self):
        with unittest.mock.patch.object(multi, 'build_command', return_value=None) as build, \
                unittest.mock.patch.object(multi, 'verify_tokens_command', return_value=1) as verify:
            self.assertEqual(multi.main(['build', '--limit', '2', '--speakers', 'a', 'b']), 0)
            self.assertEqual(multi.main(['verify-tokens', '--out', 'x', '-j', '2']), 1)
        self.assertEqual(build.call_args.args[0], argparse.Namespace(command='build', speakers=['a', 'b'], limit=2, jobs=None, out=None,
                                                                     pool=multi.SPEAKERS_PATH))
        self.assertEqual(verify.call_args.args[0], argparse.Namespace(command='verify-tokens', out='x', jobs=2))

    def test_module_main(self):
        with unittest.mock.patch.object(sys, 'argv', ['multi', 'verify-tokens', '--out', '/nonexistent']), \
                self.assertRaises(FileNotFoundError), warnings.catch_warnings():
            # (runpy warns that the module is already imported - it runs a fresh copy as __main__)
            warnings.filterwarnings('ignore', message='.*found in sys.modules', category=RuntimeWarning)
            runpy.run_module('picovox_train.multi', run_name='__main__', alter_sys=False)
