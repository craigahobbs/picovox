# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy

from picovox_train import heteronyms

from . import data_fixtures
from .helpers import captured, write_wav


# The tiny dictionary - word to its pronunciations
DICTIONARY = {
    'lead': [['L', 'IY1', 'D'], ['L', 'EH1', 'D']],
    'record': [['R', 'EH1', 'K', 'ER0', 'D'], ['R', 'EH0', 'K', 'ER1', 'D']],
    'the': [['DH', 'AH0'], ['DH', 'IY0']],
    'use': [['Y', 'UW1', 'S'], ['Y', 'UW1', 'Z']],
    'cat': [['K', 'AE1', 'T']]
}


# A voiced synthetic utterance (a 120 Hz harmonic series, its first resonance near 700 Hz) - silent after silence_at seconds
def voiced_audio(seconds=1.2, silence_at=0.9, rate=24000):
    times = numpy.arange(int(seconds * rate)) / rate
    signal = numpy.zeros_like(times)
    for harmonic in range(1, 40):
        signal += numpy.sin(2 * numpy.pi * 120 * harmonic * times) / harmonic * numpy.exp(-((120 * harmonic - 700) / 400.0) ** 2)
    signal[int(silence_at * rate):] = 0
    return 0.3 * signal, rate


class TestHeteronyms(unittest.TestCase):

    # The fake acoustic analysis
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(data_fixtures.fake_modules(parselmouth=data_fixtures.ANALYSIS))

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.labels = os.path.join(self.directory, 'lab')
        self.audio = os.path.join(self.directory, 'audio')
        self.out = os.path.join(self.directory, 'out')
        os.makedirs(self.out)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def write(self, path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text)

    # Write an utterance's word and phone labels - words [(word, [(phone, start, end)])]
    def write_utterance(self, speaker, number, words, subset='train-clean-100', phone_labels=True):
        name = f'{speaker}_1_{number:06d}.lab'
        word_lines = [f'{phones[0][1]}\t{phones[-1][2]}\t{word}' for word, phones in words] + ['9.0\t9.1\t']
        phone_lines = [f'{start}\t{end}\t{phone}' for _, phones in words for phone, start, end in phones] + ['9.0\t9.1\tsil']
        self.write(os.path.join(self.labels, 'word', subset, speaker, '1', name), '\n'.join(word_lines) + '\n')
        if phone_labels:
            self.write(os.path.join(self.labels, 'phone', subset, speaker, '1', name), '\n'.join(phone_lines) + '\n')
        return os.path.join(subset, speaker, '1', name)

    def test_base_and_labels(self):
        self.assertEqual(heteronyms.base('AH0'), 'AH')
        path = os.path.join(self.directory, 'a.lab')
        self.write(path, '0.0\t0.1\tthe\n0.1\t0.2\t\nshort\n')
        self.assertEqual(heteronyms.read_lab(path), [(0.0, 0.1, 'the')])

    def test_read_dictionary(self):
        dictionary = heteronyms.read_dictionary()
        self.assertEqual(dictionary['read'], [['R', 'EH1', 'D'], ['R', 'IY1', 'D']])
        self.write(os.path.join(self.directory, 'data', 'cmudict.dict'), ';;; a comment\n\nread R IY1 D\nread(2) R EH1 D # past\n')
        with unittest.mock.patch.object(heteronyms, 'ROOT', self.directory):
            self.assertEqual(dict(heteronyms.read_dictionary()), {'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']]})

    def test_edit_ops(self):
        self.assertEqual(heteronyms.edit_ops(['L', 'IY1', 'D'], ['L', 'EH1', 'D']), [('IY1', 'EH1')])
        self.assertEqual(heteronyms.edit_ops(['F', 'AO1', 'R', 'T'], ['F', 'AO1', 'R']), [('T', None)])
        self.assertEqual(heteronyms.edit_ops(['F', 'AO1'], ['S', 'F', 'AO2']), [(None, 'S'), ('AO1', 'AO2')])
        self.assertEqual(heteronyms.edit_ops([], ['K']), [(None, 'K')])
        cost, substitution = heteronyms.edit_costs(['AH0'], ['AH1'])
        self.assertEqual(cost[1][1], 0.5)
        self.assertEqual(substitution('K', 'T'), 1)

    def test_wilson_lower(self):
        self.assertEqual(heteronyms.wilson_lower(0, 0), 0.0)
        self.assertAlmostEqual(heteronyms.wilson_lower(7, 8), 0.5291, places=4)
        self.assertLess(heteronyms.wilson_lower(1, 2), 0.5)

    def test_kind(self):
        self.assertEqual(heteronyms.kind('R EH1 K ER0 D'.split(), 'R EH0 K ER1 D'.split()), 'stress')
        self.assertEqual(heteronyms.kind('B AH0 L'.split(), 'B IH0 L'.split()), 'reduced')
        self.assertEqual(heteronyms.kind('F AO1 R T'.split(), 'F AO1 R'.split()), 'consonant-drop')
        self.assertEqual(heteronyms.kind('F AE1 M L IY0'.split(), 'F AE1 M AH0 L IY0'.split()), 'syllable')
        self.assertEqual(heteronyms.kind('L IY1 D'.split(), 'L EH1 D'.split()), 'vowel')
        self.assertEqual(heteronyms.kind('Y UW1 S'.split(), 'Y UW1 Z'.split()), 'consonant')

    def test_difference(self):
        self.assertEqual(heteronyms.difference('L IY1 D'.split(), 'L EH1 D'.split()), (1, 1, 'vowel'))
        self.assertEqual(heteronyms.difference('Y UW1 S'.split(), 'Y UW1 Z'.split()), (2, 2, 'voicing'))
        self.assertEqual(heteronyms.difference('F AO1 R T'.split(), 'F AO1 R'.split()), None)
        self.assertEqual(heteronyms.difference('F AO1 R'.split(), 'S F AO1 R'.split()), None)
        self.assertEqual(heteronyms.difference('F AO1 R'.split(), 'F AO1 R T'.split()), None)

    def test_lexical(self):
        self.assertTrue(heteronyms.lexical('L IY1 D', 'L EH1 D'))
        self.assertFalse(heteronyms.lexical('R EH1 K ER0 D', 'R EH0 K ER1 D'))
        self.assertTrue(heteronyms.lexical('K AA1 N T EH0 N T', 'K AH0 N T EH1 N T'))
        self.assertFalse(heteronyms.lexical('B AH0 L', 'B IH0 L'))
        self.assertTrue(heteronyms.lexical('F AE1 M L IY0', 'F EY1 M AH0 L IY0'))

    def write_readers_corpus(self):
        # Four speakers read lead as L EH1 D (and one as L IY1 D), the the (DH IY0), record ambiguously or otherwise, and
        # use with its first variant
        number = 0
        for speaker in ('100', '200', '300', '400'):
            for reading in range(2):
                lead = 'EH1' if speaker != '400' or reading == 0 else 'IY1'
                record = [('R', 0.6, 0.65), ('EH2', 0.65, 0.7), ('K', 0.7, 0.75), ('ER2', 0.75, 0.8), ('D', 0.8, 0.85)] if reading else \
                    [('R', 0.6, 0.7), ('EH1', 0.7, 0.8), ('K', 0.8, 0.85)]
                self.write_utterance(speaker, number, [
                    ('the', [('DH', 0.0, 0.05), ('IY0', 0.05, 0.1)]), ('lead', [('L', 0.1, 0.2), (lead, 0.2, 0.3), ('D', 0.3, 0.35)]),
                    ('sp', [('sp', 0.35, 0.4)]), ('use', [('Y', 0.4, 0.45), ('UW1', 0.45, 0.5), ('S', 0.5, 0.55)]),
                    ('record', record), ('cat', [('K', 0.9, 0.95), ('AE1', 0.95, 1.0), ('T', 1.0, 1.05)])])
                number += 1
        # (an utterance without phone labels)
        self.write_utterance('500', 0, [('lead', [('L', 0.0, 0.1)])], subset='dev-clean', phone_labels=False)

    def test_utterance_tokens(self):
        rel = self.write_utterance('100', 0, [('lead', [('L', 0.0, 0.1), ('EH1', 0.1, 0.2), ('sp', 0.2, 0.25), ('D', 0.25, 0.3)]),
                                              ('cat', [('K', 0.3, 0.4)]), ('lead', [('L', 0.4, 0.5)])])
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels):
            tokens = heteronyms.utterance_tokens((os.path.join(self.labels, 'word', rel), {'lead'}))
            missing = self.write_utterance('200', 0, [('lead', [('L', 0.0, 0.1)])], phone_labels=False)
            self.assertEqual(heteronyms.utterance_tokens((os.path.join(self.labels, 'word', missing), {'lead'})), [])
        self.assertEqual([token[:6] for token in tokens],
                         [('lead', 'L EH1 D', '<s>', 'cat', 0.0, 0.3), ('lead', 'L', 'cat', '</s>', 0.4, 0.5)])
        self.assertEqual(tokens[0][6], [(0.0, 0.1, 'L'), (0.1, 0.2, 'EH1'), (0.25, 0.3, 'D')])
        self.assertEqual(tokens[0][7], os.path.join('word', rel))

    def run_readers(self):
        self.write_readers_corpus()
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels), \
                unittest.mock.patch.object(heteronyms, 'read_dictionary', lambda: DICTIONARY), \
                unittest.mock.patch.object(heteronyms.multiprocessing, 'Pool', data_fixtures.FakePool):
            return heteronyms.readers(self.out, 1)

    def test_readers_and_candidates(self):
        self.assertEqual(self.run_readers(), '9 utterances, 4 multi-pronunciation words, 4 seen')
        with open(os.path.join(self.out, 'readers.json'), encoding='utf-8') as file:
            counts = json.load(file)
        self.assertEqual(counts['lead'], {'variants': ['L IY1 D', 'L EH1 D'], 'exact': [1, 7], 'stressOnly': [0, 0], 'ambiguous': 0,
                                          'other': {}, 'otherTotal': 0, 'speakerCounts': [1, 4]})
        self.assertEqual(counts['record']['ambiguous'], 4)
        self.assertEqual(counts['record']['other'], {'R EH1 K': 4})
        self.assertEqual(counts['the']['exact'], [0, 8])
        with open(os.path.join(self.out, 'tokens.jsonl'), encoding='utf-8') as file:
            tokens = [json.loads(line) for line in file]
        self.assertEqual(len(tokens), 32)
        self.assertEqual(tokens[1], {'word': 'lead', 'phones': 'L EH1 D', 'choice': 1, 'prev': 'the', 'next': 'sp', 'start': 0.1,
                                     'end': 0.35, 'segments': [[0.1, 0.2, 'L'], [0.2, 0.3, 'EH1'], [0.3, 0.35, 'D']],
                                     'path': os.path.join('word', 'train-clean-100', '100', '1', '100_1_000000.lab')})

        self.assertEqual(heteronyms.candidates(self.out), '2 words with a non-first majority, 2 significant')
        with open(os.path.join(self.out, 'candidates.json'), encoding='utf-8') as file:
            items = json.load(file)
        self.assertEqual(sorted((item['word'], item['significant'], item['kind'], item['counts']) for item in items),
                         [('lead', True, 'vowel', [1, 7]), ('the', True, 'reduced', [0, 8])])

    def test_readers_stress_only(self):
        self.write_utterance('100', 0, [('use', [('Y', 0.0, 0.1), ('UW2', 0.1, 0.2), ('Z', 0.2, 0.3)])])
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels), \
                unittest.mock.patch.object(heteronyms, 'read_dictionary', lambda: DICTIONARY), \
                unittest.mock.patch.object(heteronyms.multiprocessing, 'Pool', data_fixtures.FakePool):
            heteronyms.readers(self.out, 1)
        with open(os.path.join(self.out, 'readers.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file)['use']['stressOnly'], [0, 1])

    def write_audio_utterance(self):
        # lead (its vowel the target), use (its fricative the target), cat (a reference vowel), and words past the audio
        rel = self.write_utterance('100', 0, [
            ('lead', [('L', 0.0, 0.05), ('EH1', 0.05, 0.25), ('D', 0.25, 0.3)]),
            ('use', [('Y', 0.3, 0.35), ('UW1', 0.35, 0.45), ('Z', 0.45, 0.6)]),
            ('cat', [('K', 0.6, 0.65), ('AE1', 0.65, 0.85), ('T', 0.85, 0.9)]),
            ('see', [('S', 1.5, 1.6), ('IY1', 1.6, 1.7)]),
            ('cue', [('K', 1.7, 1.75), ('Y', 1.75, 1.8)]),
            ('it', [('IH1', 1.8, 1.82)]),
            ('lead', [('L', 1.9, 2.0), ('EH1', 2.0, 2.1), ('D', 2.1, 2.2)])
        ])
        audio, rate = voiced_audio()
        wav = os.path.join(self.audio, 'train-clean-100', '100', '1', '100_1_000000.wav')
        os.makedirs(os.path.dirname(wav))
        write_wav(wav, audio, rate)
        return rel

    def test_measure(self):
        rel = self.write_audio_utterance()
        wanted = {(0.0, 'lead'): (1, 'vowel'), (0.3, 'use'): (2, 'voicing'), (1.7, 'cue'): (5, 'vowel'), (1.8, 'it'): (0, 'vowel'),
                  (1.9, 'lead'): (1, 'vowel'), (0.6, 'cat'): (2, 'voicing')}
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels), unittest.mock.patch.object(heteronyms, 'AUDIO', self.audio), \
                warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
            result = heteronyms.measure((rel, 'M', wanted))
            female = heteronyms.measure((rel, 'F', {}))
            self.assertIsNone(heteronyms.measure((os.path.join('train-clean-100', '100', '1', 'none.lab'), 'M', {})))
        self.assertEqual(result['rel'], rel)
        self.assertEqual(result['speaker'], '100')
        self.assertEqual([(target['word'], target['phone']) for target in result['targets']], [('lead', 'EH1'), ('use', 'Z'), ('cat', 'T')])
        self.assertEqual(len(result['targets'][0]['f']), 2)
        self.assertGreater(result['targets'][1]['voiced'], 0.5)
        self.assertEqual(result['vowels'], [])
        self.assertEqual([phone for phone, _ in female['vowels']], ['EH1', 'UW1', 'AE1'])

    def write_acoustic_inputs(self):
        candidates = [
            {'word': 'lead', 'first': 'L IY1 D', 'majority': 'L EH1 D', 'significant': True},
            {'word': 'use', 'first': 'Y UW1 Z', 'majority': 'Y UW1 S', 'significant': True},
            {'word': 'uses', 'first': 'Y UW1 S IH0 Z', 'majority': 'Y UW1 Z IH0 Z', 'significant': True},
            {'word': 'forty', 'first': 'F AO1 R T', 'majority': 'F AO1 R', 'significant': True},
            {'word': 'the', 'first': 'DH AH0', 'majority': 'DH IY0', 'significant': False}
        ]
        with open(os.path.join(self.out, 'candidates.json'), 'w', encoding='utf-8') as file:
            json.dump(candidates, file)
        self.write(os.path.join(self.audio, 'speakers.tsv'), 'READER\tGENDER\n100\tF\n200\tM\nshort\n')
        tokens = []
        for number in range(6):
            for speaker in ('100', '200', '300'):
                rel = self.write_utterance(speaker, number, [('lead', [('L', 0.0, 0.1)])])
                path = os.path.join('word', rel)
                tokens.append({'word': 'lead', 'phones': 'L EH1 D' if number % 3 else 'L IY0 D', 'start': 0.0, 'path': path})
                tokens.append({'word': 'use', 'phones': 'Y UW1 S' if number % 2 else 'Y UW1 Z', 'start': 0.5, 'path': path})
                tokens.append({'word': 'uses', 'phones': 'Y UW1 Z IH0 Z', 'start': 1.0, 'path': path})
                tokens.append({'word': 'lead', 'phones': 'L AA1 D', 'start': 1.5, 'path': path})
                tokens.append({'word': 'cat', 'phones': 'K AE1 T', 'start': 2.0, 'path': path})
        # (one more reference utterance of speaker 100, with no tokens)
        self.write_utterance('100', 99, [('cat', [('K', 0.0, 0.1)])], subset='dev-clean')
        with open(os.path.join(self.out, 'tokens.jsonl'), 'w', encoding='utf-8') as file:
            file.write(''.join(json.dumps(token) + '\n' for token in tokens))

    # A fake measure - the reference vowels (speakers 100 and 200 have 40 each: IY1, EH1, AH0, AA1; speaker 300 too few), and
    # each target's measure: lead's formants near EH1 (or AH0 for the second utterances), use's voicing
    @staticmethod
    def fake_measure(task):
        rel, _, wanted = task
        speaker = rel.split(os.sep)[1]
        if rel.endswith('000005.lab') and speaker == '300':
            return None
        formants = {'IY1': (300.0, 2300.0), 'EH1': (550.0, 1800.0), 'AH0': (500.0, 1400.0), 'AA1': (750.0, 1100.0)}
        count = 10 if speaker != '300' else 1
        vowels = [(phone, (value[0] + index, value[1] - index)) for phone, value in formants.items() for index in range(count)]
        vowels = vowels if rel.endswith('000000.lab') else []
        targets = []
        for (start, word), (_, kind) in wanted.items():
            if kind == 'vowel':
                targets.append({'word': word, 'start': start, 'phone': 'X', 'f': formants['AH0' if rel.endswith('000001.lab') else 'EH1']})
            else:
                targets.append({'word': word, 'start': start, 'phone': 'X', 'voiced': None if rel.endswith('000002.lab') else 0.8})
        return {'rel': rel, 'speaker': speaker, 'vowels': vowels, 'targets': targets}

    def run_acoustic(self, per_word=240):
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels), unittest.mock.patch.object(heteronyms, 'AUDIO', self.audio), \
                unittest.mock.patch.object(heteronyms, 'measure', self.fake_measure), \
                unittest.mock.patch.object(heteronyms.multiprocessing, 'Pool', data_fixtures.FakePool):
            return heteronyms.acoustic(self.out, 1, per_word)

    def test_acoustic(self):
        self.write_acoustic_inputs()
        self.assertEqual(self.run_acoustic(), '3 words checked, 32 tokens classified')
        with open(os.path.join(self.out, 'acoustic.json'), encoding='utf-8') as file:
            result = json.load(file)
        self.assertEqual(sorted(result['references']), ['AA1', 'AH0', 'EH1', 'IY1'])
        lead = result['words']['lead']
        self.assertEqual(lead['kind'], 'vowel')
        # (lead's second utterances measure as schwa; speaker 300 has too few reference vowels; use's third utterances no voicing)
        self.assertEqual(lead['tokens'], 12)
        self.assertEqual(lead['aligner'], {'first': 4, 'majority': 8})
        self.assertEqual(lead['acoustic'], {'majority': 10, 'schwa': 2})
        self.assertEqual(lead['agree'], 6)
        use = result['words']['use']
        self.assertEqual(use['kind'], 'voicing')
        self.assertEqual(use['tokens'], 10)
        self.assertEqual(use['acoustic'], {'first': 10})
        self.assertEqual(result['words']['uses']['acoustic'], {'majority': 10})

        # At most per_word tokens a word
        os.remove(os.path.join(self.out, 'acoustic.json'))
        summary = self.run_acoustic(per_word=2)
        self.assertLessEqual(int(summary.split(', ')[1].split()[0]), 6)

    def test_acoustic_schwa_option(self):
        # A difference to or from schwa itself - no third option; reference vowels named by their stress-1 forms
        candidates = [{'word': 'a', 'first': 'AH0', 'majority': 'EY2', 'significant': True}]
        with open(os.path.join(self.out, 'candidates.json'), 'w', encoding='utf-8') as file:
            json.dump(candidates, file)
        self.write(os.path.join(self.audio, 'speakers.tsv'), '100\tF\n')
        rel = self.write_utterance('100', 0, [('a', [('AH0', 0.0, 0.1)])])
        with open(os.path.join(self.out, 'tokens.jsonl'), 'w', encoding='utf-8') as file:
            file.write(json.dumps({'word': 'a', 'phones': 'EY1', 'start': 0.0, 'path': os.path.join('word', rel)}) + '\n')

        def measure(task):
            vowels = [('AH0', (500.0 + index, 1400.0)) for index in range(20)] + [('EY1', (450.0 + index, 2000.0)) for index in range(20)]
            targets = [{'word': 'a', 'start': 0.0, 'phone': 'EY1', 'f': (450.0, 2000.0)}] if task[2] else []
            return {'rel': task[0], 'speaker': '100', 'vowels': vowels, 'targets': targets}
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels), unittest.mock.patch.object(heteronyms, 'AUDIO', self.audio), \
                unittest.mock.patch.object(heteronyms, 'measure', measure), \
                unittest.mock.patch.object(heteronyms.multiprocessing, 'Pool', data_fixtures.FakePool):
            heteronyms.acoustic(self.out, 1, 10)
        with open(os.path.join(self.out, 'acoustic.json'), encoding='utf-8') as file:
            self.assertEqual(json.load(file)['words']['a']['acoustic'], {'majority': 1})

    def test_select(self):
        significant = [
            {'word': 'the', 'first': 'DH AH0', 'majority': 'DH IY0', 'index': 1, 'counts': [1, 9], 'kind': 'reduced'},
            {'word': 'lead', 'first': 'L IY1 D', 'majority': 'L EH1 D', 'index': 1, 'counts': [1, 9], 'kind': 'vowel'},
            {'word': 'balloon', 'first': 'B AH0 L', 'majority': 'B IH0 L', 'index': 1, 'counts': [1, 9], 'kind': 'vowel'},
            {'word': 'use', 'first': 'Y UW1 Z', 'majority': 'Y UW1 S', 'index': 1, 'counts': [4, 6], 'kind': 'consonant'},
            {'word': 'wind', 'first': 'W IH1 N D', 'majority': 'W AY1 N D', 'index': 1, 'counts': [2, 8], 'kind': 'vowel'},
            {'word': 'bass', 'first': 'B AE1 S', 'majority': 'B AH1 S', 'index': 1, 'counts': [2, 8], 'kind': 'vowel'},
            {'word': 'tear', 'first': 'T IH1 R', 'majority': 'T EH1 R', 'index': 1, 'counts': [2, 8], 'kind': 'vowel'},
            {'word': 'live', 'first': 'L IH1 V', 'majority': 'L AY1 V', 'index': 1, 'counts': [2, 8], 'kind': 'vowel'}
        ]
        with open(os.path.join(self.out, 'candidates.json'), 'w', encoding='utf-8') as file:
            json.dump([{**item, 'significant': True} for item in significant] + [{'word': 'no', 'significant': False}], file)
        checks = {
            'wind': {'kind': 'vowel', 'acoustic': {'majority': 2, 'first': 4}},
            'bass': {'kind': 'vowel', 'acoustic': {'majority': 3, 'first': 2, 'schwa': 6}},
            'tear': {'kind': 'vowel', 'acoustic': {'majority': 3, 'first': 2, 'schwa': 6}},
            'live': {'kind': 'vowel', 'acoustic': {'majority': 2, 'first': 1}}
        }
        with open(os.path.join(self.out, 'acoustic.json'), 'w', encoding='utf-8') as file:
            json.dump({'words': checks}, file)
        summary = heteronyms.select(self.out, 0.5)
        self.assertEqual(summary, '4 relabeled of 8 significant candidates: lead L EH1 D, use Y UW1 S, bass B AH1 S, live L AY1 V')
        with open(os.path.join(self.out, 'relabel.json'), encoding='utf-8') as file:
            relabel = json.load(file)
        self.assertEqual(relabel['relabel'], {'lead': 'L EH1 D', 'use': 'Y UW1 S', 'bass': 'B AH1 S', 'live': 'L AY1 V'})
        reasons = {item['word']: item['reasons'] for item in relabel['rejected']}
        self.assertEqual(reasons['the'], ['function word', 'kind reduced'])
        self.assertEqual(reasons['balloon'], ['reduction only'])
        self.assertEqual(reasons['wind'], ['acoustic share 0.33'])
        self.assertEqual(reasons['tear'], ['acoustic schwa'])
        self.assertNotIn('use', reasons)
        self.assertEqual(relabel['chosen'][2]['acoustic'], checks['bass']['acoustic'])

        # A share above a majority
        heteronyms.select(self.out, 0.7)
        with open(os.path.join(self.out, 'relabel.json'), encoding='utf-8') as file:
            reasons = {item['word']: item['reasons'] for item in json.load(file)['rejected']}
        self.assertEqual(reasons['use'], ['aligner share 0.60'])
        self.assertEqual(reasons['bass'], ['acoustic share 0.60'])
        self.assertNotIn('live', reasons)

    def test_main(self):
        self.write_readers_corpus()
        self.write(os.path.join(self.audio, 'speakers.tsv'), '100\tF\n')
        expected = os.path.join(self.directory, 'expected.json')
        self.write(expected, json.dumps({'relabel': {}}))

        def measure(task):
            return {'rel': task[0], 'speaker': task[0].split(os.sep)[1], 'vowels': [], 'targets': []}
        with unittest.mock.patch.object(heteronyms, 'LABELS', self.labels), unittest.mock.patch.object(heteronyms, 'AUDIO', self.audio), \
                unittest.mock.patch.object(heteronyms, 'read_dictionary', lambda: DICTIONARY), \
                unittest.mock.patch.object(heteronyms, 'measure', measure), \
                unittest.mock.patch.object(heteronyms.multiprocessing, 'Pool', data_fixtures.FakePool):
            with captured() as (_, stderr):
                self.assertEqual(heteronyms.main(['--out', self.out, '-j', '1', '--check', expected]), 1)
            self.assertIn('readers: 9 utterances', stderr.getvalue())
            self.assertIn('select: 1 relabeled of 2 significant candidates: lead L EH1 D', stderr.getvalue())
            self.assertIn(f'check: the relabeled words differ from {expected}: lead L EH1 D (was None)', stderr.getvalue())

            # Resumed - every step done before; the same words
            self.write(expected, json.dumps({'relabel': {'lead': 'L EH1 D'}}))
            with captured() as (_, stderr):
                self.assertEqual(heteronyms.main(['--out', self.out, '--check', expected]), 0)
            self.assertIn('readers: done before', stderr.getvalue())
            self.assertIn('check: the 1 relabeled words equal', stderr.getvalue())
            with captured():
                self.assertEqual(heteronyms.main(['--out', self.out]), 0)

    def test_main_module(self):
        with unittest.mock.patch.dict(sys.modules), unittest.mock.patch.object(sys, 'argv', ['heteronyms', '--help']), \
                captured() as (stdout, _):
            sys.modules.pop('picovox_train.heteronyms')
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.heteronyms', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('relabeled words', stdout.getvalue())
