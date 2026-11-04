# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import math
import multiprocessing
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy

from picovox_train import corpora, ctc, measure, readings, tts

from . import ctc_fixtures, data_fixtures
from .helpers import captured


# The measurement's tiny dictionary - its heteronyms (read and lead strong, use voicing, record weak, dawn merger, insult
# stress, the closed-class), its control words (went and home ctc, look formant, times and cats voicing), and words no
# control can be made of (dark - its vowel before R, night - no primary stress)
DICTIONARY = {
    'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']],
    'lead': [['L', 'EH1', 'D'], ['L', 'IY1', 'D']],
    'use': [['Y', 'UW1', 'S'], ['Y', 'UW1', 'Z']],
    'record': [['R', 'EH1', 'K', 'ER0', 'D'], ['R', 'IH0', 'K', 'AO1', 'R', 'D']],
    'dawn': [['D', 'AA1', 'N'], ['D', 'AO1', 'N']],
    'insult': [['IH1', 'N', 'S', 'AH0', 'L', 'T'], ['IH0', 'N', 'S', 'AH1', 'L', 'T']],
    'the': [['DH', 'AH0'], ['DH', 'IY0']],
    'went': [['W', 'EH1', 'N', 'T']],
    'home': [['HH', 'OW1', 'M']],
    'look': [['L', 'UH1', 'K']],
    'times': [['T', 'AY1', 'M', 'Z']],
    'cats': [['K', 'AE1', 'T', 'S']],
    'dark': [['D', 'AA1', 'R', 'K']],
    'night': [['N', 'AY0', 'T']]
}


# The corpus's clips - a male voice's (enough reference vowels for the formants) and a female voice's (too few)
MALE_TEXTS = [
    'I read the lead, we went home.', 'The cats use a record, the best went home.', 'Look, the dawn went dark.',
    'They insult the night, went the cats.', 'The best went home.', 'Then the best went.', 'The men went home.',
    'The night went dark.', 'The cats went home.', 'We went the best way.', 'Look at the times, the men went.',
    'The times went dark.'
]
FEMALE_TEXTS = ['I read the use.', 'Look at the record.']


# The fake formant tracker's vowels (F1, F2 by measure.vowel_key); any other phone's, and a pause's
FORMANTS = {
    'AA': (750.0, 1100.0), 'AE': (700.0, 1700.0), 'AH': (600.0, 1200.0), 'AH0': (500.0, 1400.0), 'AO': (600.0, 900.0),
    'AW': (700.0, 1200.0), 'AY': (700.0, 1300.0), 'EH': (550.0, 1900.0), 'ER': (500.0, 1300.0), 'EY': (450.0, 2100.0),
    'IH': (400.0, 2000.0), 'IY': (300.0, 2300.0), 'OW': (500.0, 900.0), 'OY': (500.0, 1000.0), 'UH': (450.0, 1000.0),
    'UW': (320.0, 900.0)
}
OTHER = (900.0, 2600.0)



# The process pools in this process (the workers' code covered)
def setUpModule():  # pylint: disable=invalid-name
    patcher = unittest.mock.patch('concurrent.futures.ProcessPoolExecutor', ctc_fixtures.FakeExecutor)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)

class FakeFormantTrack:
    """A fake Praat formant track - each phone label's formants (FORMANTS), a little modulated in time; NaN outside the labels"""

    def __init__(self, labels):
        self.labels = labels

    def get_value_at_time(self, number, moment):
        label = next((label for label in self.labels if label[1] <= moment < label[2]), None)
        if label is None:
            return math.nan
        return FORMANTS.get(measure.vowel_key(label[0]), OTHER)[number - 1] + 20.0 * math.sin(997.0 * moment)


class FakePraat:
    """The fake parselmouth module - its sounds' phone labels by path (a test patches them)"""

    labels = {}

    # A fake parselmouth.Sound
    class Sound:

        def __init__(self, path):
            self.path = path

        def to_formant_burg(self, time_step, max_number_of_formants, maximum_formant):
            assert time_step > 0 and max_number_of_formants == 5 and maximum_formant in (5000, 5500)
            return FakeFormantTrack(FakePraat.labels[self.path])


# A token sequence's alignment record (word_times's fields) - the phones' labels each 0.1 s, after a 0.2 s pause
def record_of(tokens, seconds=3.0):
    phones = [token for token in tokens if token not in readings.BOUNDARIES]
    labels = [['pau', 0.0, 0.2]] + [[phone, round(0.2 + 0.1 * index, 4), round(0.3 + 0.1 * index, 4)] for index, phone in enumerate(phones)]
    return {'tokens': tokens, 'labels': labels + [['pau', labels[-1][2], seconds]], 'seconds': seconds}


class TestReadings(unittest.TestCase):

    def test_unstressed(self):
        self.assertEqual(readings.unstressed('R EH1 K ER0 D'), 'R EH K ER D')

    def test_vowel_pairs(self):
        self.assertEqual(readings.vowel_pairs('R IY1 D', 'R EH1 D'), [('IY1', 'EH1')])
        self.assertEqual(readings.vowel_pairs('K AE1 T', 'K AH0 AE1 T'), [])
        self.assertEqual(readings.vowel_pairs('K AE1 T', 'K AH0 T IH0'), [('AE1', 'AH0')])

    def test_distinct(self):
        self.assertTrue(readings.distinct('R IY1 D', 'R EH1 D'))
        self.assertTrue(readings.distinct('IH1 N S AH0 L T', 'IH0 N S AH1 L T'))
        self.assertTrue(readings.distinct('K L OW1 S', 'K L OW1 Z'))
        self.assertTrue(readings.distinct('S EH1 P ER0 EY2 T', 'S EH1 P ER0 AH0 T'))
        self.assertFalse(readings.distinct('T AH0', 'T UW1'))
        self.assertFalse(readings.distinct('R IY1 D', 'R IY0 D'))

    def test_readings_of(self):
        self.assertEqual(readings.readings_of(['R IY1 D', 'R EH1 D', 'R IY0 D']), [['R IY1 D', 'R IY0 D'], ['R EH1 D']])

    def test_read_variants_inventory(self):
        dictionary = {
            'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D'], ['R', 'IY1', 'D']], 'to': [['T', 'UW1'], ['T', 'AH0']],
            'the': [['DH', 'AH0'], ['DH', 'IY1']], 'lead': [['L', 'IY1', 'D'], ['L', 'EH1', 'D']],
            'cat': [['K', 'AE1', 'T'], ['K', 'AE0', 'T']], 'mat': [['M', 'AE1', 'T']]
        }
        with unittest.mock.patch.object(corpora, 'read_dictionary', lambda: dictionary):
            variants = readings.read_variants()
            self.assertEqual(sorted(readings.inventory()), ['lead', 'read'])
        self.assertEqual(variants['read'], ['R IY1 D', 'R EH1 D'])
        self.assertEqual(variants['cat'], ['K AE1 T', 'K AE0 T'])
        self.assertEqual(readings.inventory(variants), {'read': [['R IY1 D'], ['R EH1 D']], 'lead': [['L IY1 D'], ['L EH1 D']]})

    def test_sz_only(self):
        self.assertTrue(readings.sz_only('Y UW1 S', 'Y UW1 Z'))
        self.assertTrue(readings.sz_only('Y UW1 S', 'Y UW0 Z'))
        self.assertFalse(readings.sz_only('Y UW1 S', 'Y UW1 S'))
        self.assertFalse(readings.sz_only('K L OW1 S', 'K L AW1 Z'))
        self.assertFalse(readings.sz_only('Y UW1 S', 'Y UW1 Z D'))

    def test_reading_of(self):
        word_readings = [['R IY1 D', 'R IY0 D IY0'], ['R EH1 D'], ['R IY2 D IY2']]
        self.assertEqual(readings.reading_of('R EH1 D', word_readings), 1)
        self.assertEqual(readings.reading_of('R EH0 D', word_readings), 1)
        self.assertIsNone(readings.reading_of('R IY1 D IY1', word_readings))
        self.assertIsNone(readings.reading_of('K AE1 T', word_readings))

    def test_token_words(self):
        boundaries = {'<bos>', '_', '<period>', '<comma>'}
        self.assertEqual(readings.token_words(['<bos>', 'AY1', '_', 'R', 'EH1', 'D', '<comma>', 'IH1', 'T', '<period>'], boundaries),
                         ['AY1', 'R EH1 D', 'IH1 T'])
        self.assertEqual(readings.token_words(['<bos>', 'AY1', '_', 'IH1', 'T', '<period>']), ['AY1', 'IH1 T'])

    def test_quality_tokens(self):
        self.assertEqual(readings.quality_tokens('IY1'), {'iː', 'i', 'iə'})
        self.assertEqual(readings.quality_tokens('S'), {'s', 'z'})

    def test_pair_class(self):
        self.assertEqual(readings.pair_class('IH1 N S AH0 L T', 'IH0 N S AH1 L T'), 'stress')
        self.assertEqual(readings.pair_class('D AO1 N', 'D AA1 N'), 'merger')
        self.assertEqual(readings.pair_class('K AE1 N', 'K EH1 N'), 'merger')
        self.assertEqual(readings.pair_class('R IY1 D', 'R EH1 D'), 'strong')
        self.assertEqual(readings.pair_class('P AE1 T', 'B AE1 T'), 'strong')
        self.assertEqual(readings.pair_class('Y UW1 S', 'Y UW1 Z'), 'voicing')
        self.assertEqual(readings.pair_class('R EH1 K ER0 D', 'R IH0 K AO1 R D'), 'weak')
        self.assertEqual(readings.pair_class('K AE1 T', 'K AE1 T S'), 'weak')
        self.assertEqual(readings.pair_class('K AE1 T', 'K AE2 T S'), 'weak')
        self.assertEqual(readings.pair_class('IH1 Z', 'Y Z'), 'weak')
        self.assertEqual(readings.pair_class('K AE1 T', 'G AH0 T'), 'strong')

    def test_word_class(self):
        self.assertEqual(readings.word_class([['R EH1 K ER0 D'], ['R IH0 K AO1 R D'], ['R IY1 K ER0 D']]), 'strong')
        self.assertEqual(readings.word_class([['IH1 N S AH0 L T'], ['IH0 N S AH1 L T']]), 'stress')

    def test_measured_words_coverage_quotas(self):
        dictionary = {'read': DICTIONARY['read'], 'use': DICTIONARY['use'], 'insult': DICTIONARY['insult'],
                      'dawn': DICTIONARY['dawn']}

        # The context labels' candidates - use and insult (in the inventory), and wound (with a reading the dictionary lacks)
        def candidate_words(spec, inventory):
            self.assertEqual(spec, {'words': {}})
            return {word: {'readings': inventory[word]} for word in ('use', 'insult')} | \
                {'wound': {'readings': [['W UW1 N D'], ['W AW1 N D']]}}
        with unittest.mock.patch.object(corpora, 'read_dictionary', lambda: dictionary), \
                unittest.mock.patch.object(readings, 'load_spec', return_value={'words': {}}), \
                unittest.mock.patch.object(readings, 'candidate_words', candidate_words):
            words = readings.measured_words()
            quotas = readings.coverage_quotas({'gate': 5, 'context': 3, 'other': 1}, ['read'])
        self.assertEqual(sorted(words), ['dawn', 'insult', 'read', 'use', 'wound'])
        self.assertEqual(words['wound'], [['W UW1 N D'], ['W AW1 N D']])
        self.assertEqual(quotas, {'read': 5, 'use': 3, 'dawn': 1, 'wound': 3})

    def test_measure_key(self):
        key = measure.measure_key({'read': [['R IY1 D'], ['R EH1 D']]})
        self.assertEqual(len(key), 12)
        self.assertNotEqual(measure.measure_key({'read': [['R IY1 D']]}), key)
        with unittest.mock.patch.object(measure, 'FORMANT_MARGIN', 0.5):
            self.assertNotEqual(measure.measure_key({'read': [['R IY1 D'], ['R EH1 D']]}), key)

    def test_word_times(self):
        record = record_of(['<bos>', 'R', 'EH1', 'D', '_', '<comma>', 'L', 'IY1', 'D', '<period>'])
        words = measure.word_times(record)
        self.assertEqual([word[:3] for word in words], [('R EH1 D', 0.2, 0.5), ('', None, None), ('L IY1 D', 0.5, 0.8)])
        self.assertEqual(words[0][3], [['R', 0.2, 0.3], ['EH1', 0.3, 0.4], ['D', 0.4, 0.5]])
        self.assertEqual(words[1][3], [])

    def test_word_context(self):
        timed = [('AY1', 0.0, 0.1, []), ('R EH1 D', 0.2, 0.5, []), ('', None, None, []), ('L IY1 D', 0.6, 0.9, []),
                 ('DH AH0', 1.0, 2.5, []), ('K AE1 T', 2.6, 2.95, [])]
        # (both neighbors short; the first word - no previous; a neighbor without phones; a neighbor too long; the last word)
        self.assertEqual(measure.word_context(timed, 1, 3.0), (0.0, 0.62, ['AY1'], []))
        self.assertEqual(measure.word_context(timed, 0, 3.0), (0.0, 0.5, [], ['R', 'EH1', 'D']))
        result = measure.word_context(timed, 3, 3.0)
        self.assertEqual(result[2:], ([], []))
        self.assertAlmostEqual(result[0], 0.48)
        self.assertAlmostEqual(result[1], 1.02)
        result = measure.word_context(timed, 5, 3.0)
        self.assertAlmostEqual(result[0], 2.48)
        self.assertEqual(result[1:], (3.0, [], []))

    def test_voicing_positions(self):
        self.assertEqual(measure.voicing_positions('S IH1 T S'), {0, 3})
        self.assertEqual(measure.voicing_positions('K AE1 T'), set())

    def test_clip_tasks(self):
        # read (measured), an empty word, insult (stress - not measured), went (a control), home (a control the front end
        # says otherwise), cats (a control), times (a third control - not taken), and a word past the front end's words
        tokens = ('<bos> R EH1 D _ <comma> IH1 N S AH0 L T _ W EH1 N T _ HH AH0 M _ K AE1 T S _ T AY1 M Z _ K AE1 T '
                  '<period>').split()
        record = record_of(tokens, 5.0)
        words = {'read': [['R IY1 D'], ['R EH1 D']], 'insult': [['IH1 N S AH0 L T'], ['IH0 N S AH1 L T']]}
        classes = {'read': 'strong', 'insult': 'stress'}
        controls = {'went': ('ctc', ['W EH1 N T', 'W IY1 N T']), 'home': ('ctc', ['HH OW1 M', 'HH AW1 M']),
                    'cats': ('voicing', ['K AE1 T S', 'K AE1 T Z']), 'times': ('voicing', ['T AY1 M Z', 'T AY1 M S'])}
        names = ['read', '', 'insult', 'went', 'home', 'cats', 'times']
        tasks = measure.clip_tasks(record, words, classes, controls, names)
        self.assertEqual([(task['position'], task['word'], task.get('control', False)) for task in tasks],
                         [(0, 'read', False), (3, 'went', True), (5, 'cats', True)])
        segment = tasks[0].pop('segment')
        self.assertEqual(tasks[0], {'position': 0, 'word': 'read', 'class': 'strong', 'frontEnd': 1, 'method': 'ctc', 'phones': 'R EH1 D',
                                    'labels': [['R', 0.2, 0.3], ['EH1', 0.3, 0.4], ['D', 0.4, 0.5]],
                                    'candidates': ['R IY1 D', 'R EH1 D']})
        self.assertEqual(numpy.round(segment[:2], 6).tolist(), [0.08, 0.62])
        self.assertEqual(segment[2:], ([], []))
        self.assertEqual(tasks[1]['method'], 'ctc')
        self.assertEqual(tasks[1]['candidates'], ['W EH1 N T', 'W IY1 N T'])
        self.assertEqual(tasks[2]['method'], 'voicing')

    def test_control_words(self):
        counts = {word: 6 for word in ('went', 'home', 'look', 'times', 'cats', 'dark', 'night', 'read', 'lead', 'the', 'cat', 'zzzz')}
        counts['mats'] = 1
        variants = {word: [' '.join(phones) for phones in pronunciations] for word, pronunciations in DICTIONARY.items()}
        variants.update({'cat': ['K AE1 T'], 'mats': ['M AE1 T S']})
        measured = {'read': [['R IY1 D'], ['R EH1 D']]}
        self.assertEqual(measure.control_words(counts, variants, measured), {
            'went': ('ctc', ['W EH1 N T', 'W IY1 N T']), 'home': ('ctc', ['HH OW1 M', 'HH AW1 M']),
            'look': ('formant', ['L UH1 K', 'L UW1 K']), 'times': ('voicing', ['T AY1 M Z', 'T AY1 M S']),
            'cats': ('voicing', ['K AE1 T S', 'K AE1 T Z'])
        })

        # At most CONTROL_WORDS words a method
        with unittest.mock.patch.object(measure, 'CONTROL_WORDS', 1):
            controls = measure.control_words(counts, variants, measured)
        self.assertEqual(sorted(method for method, _ in controls.values()), ['ctc', 'formant', 'voicing'])

    def test_clip_formants(self):
        labels = {'clip.wav': [['pau', 0.0, 0.2], ['DH', 0.2, 0.3], ['AH0', 0.3, 0.45]]}
        with data_fixtures.fake_modules(parselmouth=FakePraat), unittest.mock.patch.object(FakePraat, 'labels', labels), \
                warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
            values = measure.clip_formants(('clip.wav', 'male', {'male': 5000, 'female': 5500}, [(0.3, 0.45), (1.0, 1.2)]))
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0][0], 500.0, delta=20.0)
        self.assertAlmostEqual(values[0][1], 1400.0, delta=20.0)
        self.assertIsNone(values[1])

    def test_vowel_key(self):
        self.assertEqual(measure.vowel_key('AH0'), 'AH0')
        self.assertEqual(measure.vowel_key('IH0'), 'AH0')
        self.assertEqual(measure.vowel_key('ER0'), 'AH0')
        self.assertEqual(measure.vowel_key('IY0'), 'IY')
        self.assertEqual(measure.vowel_key('AH1'), 'AH')

    def test_formant_target(self):
        self.assertEqual(measure.formant_target({'phones': 'R EH1 K ER0 D', 'candidates': ['R EH1 K ER0 D', 'R IH0 K AO1 R D']}),
                         (1, {1: 'AH0', 0: 'EH'}))
        # (a candidate differing elsewhere, and one by a consonant, aside)
        self.assertEqual(measure.formant_target({'phones': 'B AE1 T IY0',
                                                  'candidates': ['B AE1 T IY0', 'B EH1 T IY0', 'B AE1 T AH0', 'P AE1 T IY0']}),
                         (1, {1: 'EH', 0: 'AE'}))
        # (the token isn't a candidate; no vowel differs)
        self.assertIsNone(measure.formant_target({'phones': 'K AE1 T', 'candidates': ['K AH0 T', 'K EH1 T']}))
        self.assertIsNone(measure.formant_target({'phones': 'Y UW1 S', 'candidates': ['Y UW1 S', 'Y UW1 Z']}))

    def test_formant_vote(self):
        norm = (numpy.array([500.0, 1500.0]), numpy.array([100.0, 400.0]))
        references = {'EH': numpy.array([1.0, 1.0]), 'AH0': numpy.array([-1.0, -1.0]), 'IY': numpy.array([1.2, 1.1])}
        task = {'target': (1, {0: 'EH', 1: 'AH0'}), 'formants': [600.0, 1900.0]}
        self.assertEqual(measure.formant_vote(task, norm, references), 0)
        self.assertEqual(measure.formant_vote(dict(task, formants=[400.0, 1100.0]), norm, references), 1)
        self.assertIsNone(measure.formant_vote(dict(task, target=None), norm, references))
        self.assertIsNone(measure.formant_vote({'formants': [600.0, 1900.0]}, norm, references))
        self.assertIsNone(measure.formant_vote(dict(task, formants=None), norm, references))
        self.assertIsNone(measure.formant_vote(task, None, {}))
        # (one vowel; a vowel without a reference; the nearest not by the margin)
        self.assertIsNone(measure.formant_vote(dict(task, target=(1, {0: 'EH', 1: 'EH'})), norm, references))
        self.assertIsNone(measure.formant_vote(dict(task, target=(1, {0: 'EH', 1: 'UW'})), norm, references))
        self.assertIsNone(measure.formant_vote(dict(task, target=(1, {0: 'EH', 1: 'IY'})), norm, references))

    def test_measure_formants(self):
        # A clip - record (a formant task with its target, and one whose phones aren't a candidate), the (a reference
        # vowel, too short a vowel), and a word past the front end's words; a CTC task
        record = {'tokens': '<bos> R EH1 K ER0 D _ DH AH0 _ IH1 T <period>'.split(), 'seconds': 2.0,
                  'labels': [['pau', 0.0, 0.2], ['R', 0.2, 0.3], ['EH1', 0.3, 0.4], ['K', 0.4, 0.5], ['ER0', 0.5, 0.6], ['D', 0.6, 0.7],
                             ['DH', 0.7, 0.8], ['AH0', 0.8, 0.9], ['IH1', 0.9, 0.92], ['T', 0.92, 1.0], ['pau', 1.0, 2.0]]}
        clip = {'id': 'c1', 'key': 'male', 'voice': 'Trevor', 'sex': 'male'}
        candidates = ['R EH1 K ER0 D', 'R IH0 K AO1 R D']
        measured = {'method': 'formant', 'phones': 'R EH1 K ER0 D', 'labels': record['labels'][1:6], 'candidates': candidates}
        unmatched = {'method': 'formant', 'phones': 'R AA1 K', 'labels': record['labels'][1:4], 'candidates': candidates}
        voted = {'method': 'ctc', 'phones': 'R EH1 K ER0 D', 'labels': record['labels'][1:6], 'candidates': candidates}
        settings = {'analysis': {'formantCeilings': {'male': 5000, 'female': 5500}}}
        with unittest.mock.patch.object(multiprocessing, 'Pool', data_fixtures.FakePool), \
                unittest.mock.patch.object(data_fixtures.FakePool, 'results', [[[550.0, 1900.0], None]]):
            voices = measure.measure_formants([(clip, measured), (clip, unmatched), (clip, voted)], [clip], {'c1': record},
                                               {'record': [[candidates[0]], [candidates[1]]]}, {'c1': ['record', 'the']},
                                               'corpus', settings, 1)
        self.assertEqual(voices, {'male': (None, {})})
        self.assertEqual(measured['target'], (1, {1: 'AH0', 0: 'EH'}))
        self.assertEqual(measured['formants'], [550.0, 1900.0])
        self.assertIsNone(unmatched['target'])
        self.assertNotIn('formants', unmatched)
        self.assertNotIn('target', voted)

    def test_read_readings_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(readings.read_readings(directory))
            self.assertEqual(readings.overrides(directory), {})
            os.makedirs(os.path.join(directory, 'readings'))
            with open(os.path.join(directory, 'readings', 'words.json'), 'w', encoding='utf-8') as file:
                json.dump({'words': {'read': [['R IY1 D'], ['R EH1 D']], 'record': [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']]}}, file)
            tokens = [{'position': 1, 'word': 'read', 'class': 'strong', 'frontEnd': 1, 'reading': 0},
                      {'position': 3, 'word': 'read', 'class': 'strong', 'frontEnd': 1, 'reading': 1},
                      {'position': 4, 'word': 'read', 'class': 'voicing', 'frontEnd': 0, 'reading': None},
                      {'position': 5, 'word': 'record', 'class': 'weak', 'frontEnd': 0, 'reading': 1}]
            with open(os.path.join(directory, 'readings', 'readings.jsonl'), 'w', encoding='utf-8') as file:
                file.write(json.dumps({'id': 'a', 'tokens': tokens}) + '\n' + json.dumps({'id': 'b', 'tokens': tokens[1:]}) + '\n')
            self.assertEqual(len(readings.read_readings(directory)), 2)
            self.assertEqual(readings.overrides(directory), {'a': {'1': 'R IY1 D'}})

    def test_measure_command(self):
        texts = [('male', text, 'train') for text in MALE_TEXTS] + [('female', text, 'val') for text in FEMALE_TEXTS]
        with ctc_fixtures.tts_corpus(texts) as (directory, _, clips), ctc_fixtures.fake_recognizer(), \
                tempfile.TemporaryDirectory() as temp:
            recipe = os.path.join(temp, 'train.json')
            with open(recipe, 'w', encoding='utf-8') as file:
                file.write('{}')
            records = tts.read_alignments(directory)
            labels = {tts.wav_path(directory, clip): records[clip['id']]['labels'] for clip in clips}
            with data_fixtures.fake_modules(parselmouth=FakePraat), unittest.mock.patch.object(FakePraat, 'labels', labels), \
                    unittest.mock.patch.object(corpora, 'read_dictionary', lambda: DICTIONARY), \
                    unittest.mock.patch.object(readings, 'load_spec', return_value={}), \
                    unittest.mock.patch.object(readings, 'candidate_words', return_value={}), \
                    unittest.mock.patch.object(multiprocessing, 'Pool', data_fixtures.FakePool), \
                    unittest.mock.patch.object(ctc, 'vote', return_value=0), \
                    unittest.mock.patch.object(measure, 'CONTROL_TOKENS', 2), \
                    unittest.mock.patch.object(measure, 'MIN_CONTROL_VOTES', 1):
                with captured() as (_, stderr):
                    self.assertEqual(measure.main(['--recipe', recipe, '--batch', '4', '-j', '1']), 0)
                self.assertIn('measure: 14 clips', stderr.getvalue())
                out = os.path.join(directory, 'readings')
                with open(os.path.join(out, 'summary.json'), encoding='utf-8') as file:
                    summary = json.load(file)
                with open(os.path.join(out, 'words.json'), encoding='utf-8') as file:
                    words = json.load(file)
                measured = readings.read_readings(directory)
                overrides = readings.overrides(directory)

                # Current - not measured again
                with captured() as (_, stderr):
                    self.assertEqual(measure.main(['--recipe', recipe]), 0)
                self.assertIn('is current', stderr.getvalue())

                # A sample - measured, not kept as current
                with captured() as (_, stderr):
                    self.assertEqual(measure.main(['--recipe', recipe, '--limit', '3']), 0)
                self.assertIn('measure: 3 clips', stderr.getvalue())
                self.assertIsNone(readings.read_json(os.path.join(out, 'summary.json'))['key'])

        # The tokens - read, lead, and use by the CTC vote (its controls pass), record and dawn by the formants (the male
        # voice's record near its EH reference; no AA or AO reference for dawn, the female voice too few references) but not
        # used (no formant control voted)
        self.assertEqual(summary['clips'], 14)
        self.assertEqual(summary['key'], words['key'])
        self.assertEqual(summary['byClass'], {'strong': 3, 'voicing': 2, 'weak': 2, 'merger': 1})
        self.assertEqual(summary['votes'], {'strong': 3, 'voicing': 2, 'weak': 1})
        self.assertEqual(summary['controls'], {'ctc': {'votes': 4, 'false': 0, 'error': 0.0, 'used': True},
                                               'voicing': {'votes': 4, 'false': 0, 'error': 0.0, 'used': True},
                                               'formant': {'votes': 0, 'false': 0, 'error': 0.0, 'used': False}})
        self.assertEqual(summary['classes'], {'strong': True, 'voicing': True, 'weak': False, 'merger': False})
        self.assertEqual(sorted(summary['references']['male']), ['AH0', 'EH'])
        self.assertEqual(summary['references']['female'], {})
        self.assertEqual(words['classes'], {'read': 'strong', 'lead': 'strong', 'use': 'voicing', 'record': 'weak', 'dawn': 'merger',
                                            'insult': 'stress'})
        self.assertEqual([(record['text'], [(token['word'], token['frontEnd'], token['vote'], token['reading'])
                                            for token in record['tokens']])
                          for record in measured], [
            ('I read the lead, we went home.', [('read', 1, 0, 0), ('lead', 1, 0, 0)]),
            ('The cats use a record, the best went home.', [('use', 1, 0, 0), ('record', 0, 0, None)]),
            ('Look, the dawn went dark.', [('dawn', 1, None, None)]),
            ('I read the use.', [('read', 1, 0, 0), ('use', 0, 0, 0)]),
            ('Look at the record.', [('record', 0, None, None)])
        ])
        self.assertEqual(len(measured[0]['tokens'][0]['scores']), 2)
        self.assertEqual(len(measured[1]['tokens'][1]['formants']), 2)
        self.assertEqual(measured[3]['voice'], 'Harper')

        # The voice's tokens take the CTC readings that differ from the front end's
        self.assertEqual(overrides, {measured[0]['id']: {'1': 'R IY1 D', '3': 'L EH1 D'}, measured[1]['id']: {'2': 'Y UW1 S'},
                                     measured[3]['id']: {'1': 'R IY1 D'}})

    def test_main_module(self):
        with unittest.mock.patch.object(sys, 'argv', ['measure', '--bad']), captured() as (_, stderr), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.measure', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('unrecognized arguments: --bad', stderr.getvalue())
