# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import tempfile
import unittest
import unittest.mock

import numpy

from picovox_train import corpora, ctc, measure, readings, tts

from . import ctc_fixtures
from .helpers import captured, run_module


# The measurement's tiny dictionary - its heteronyms (read and lead strong, use voicing, record weak, dawn merger, insult
# stress, the closed-class), its control words (went and home ctc, times and cats voicing), and words no control can be
# made of (look - its vowel has no CTC partner, dark - its vowel before R, night - no primary stress)
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


# The corpus's clips
MALE_TEXTS = [
    'I read the lead, we went home.', 'The cats use a record, the best went home.', 'Look, the dawn went dark.',
    'They insult the night, went the cats.', 'The best went home.', 'Then the best went.', 'The men went home.',
    'The night went dark.', 'The cats went home.', 'We went the best way.', 'Look at the times, the men went.',
    'The times went dark.'
]
FEMALE_TEXTS = ['I read the use.', 'Look at the record.']


# The process pools in this process (the workers' code covered)
def setUpModule():  # pylint: disable=invalid-name
    patcher = unittest.mock.patch('concurrent.futures.ProcessPoolExecutor', ctc_fixtures.FakeExecutor)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


# A token sequence's alignment record (word_times's fields) - the phones' labels each 0.1 s, after a 0.2 s pause
def record_of(tokens, seconds=3.0):
    phones = [token for token in tokens if token not in readings.BOUNDARIES]
    labels = [['pau', 0.0, 0.2]] + [[phone, round(0.2 + 0.1 * index, 4), round(0.3 + 0.1 * index, 4)] for index, phone in enumerate(phones)]
    return {'tokens': tokens, 'labels': labels + [['pau', labels[-1][2], seconds]], 'seconds': seconds}


class TestReadings(unittest.TestCase):

    def test_base(self):
        self.assertEqual(readings.base('R EH1 K ER0 D'), 'R EH K ER D')
        self.assertEqual(readings.base('AH0'), 'AH')

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
        self.assertEqual(readings.token_words(['<bos>', 'AY1', '_', 'R', 'EH1', 'D', '<comma>', 'IH1', 'T', '<period>']),
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
            words, candidates = readings.measured_words()
            quotas = readings.coverage_quotas({'gate': 5, 'context': 3, 'other': 1}, ['read'])
        self.assertEqual(sorted(words), ['dawn', 'insult', 'read', 'use', 'wound'])
        self.assertEqual(sorted(candidates), ['insult', 'use', 'wound'])
        self.assertEqual(words['wound'], [['W UW1 N D'], ['W AW1 N D']])
        self.assertEqual(quotas, {'read': 5, 'use': 3, 'dawn': 1, 'wound': 3})

    def test_measure_key(self):
        settings = ctc_fixtures.tts_settings()
        clips = [{'id': 'a'}, {'id': 'b'}]
        key = measure.measure_key(settings, clips)
        self.assertEqual(len(key), 12)
        self.assertEqual(measure.measure_key(settings, clips), key)
        # The settings, the clips (a bigger corpus), the aligner, the checks, and the front end, the dictionary, and the spec
        with unittest.mock.patch.object(measure, 'NEIGHBOR_SECONDS', 0.5):
            self.assertNotEqual(measure.measure_key(settings, clips), key)
        self.assertNotEqual(measure.measure_key(settings, clips + [{'id': 'c'}]), key)
        with unittest.mock.patch.object(tts, 'aligner_key', return_value='new'):
            self.assertNotEqual(measure.measure_key(settings, clips), key)
        self.assertNotEqual(measure.measure_key(dict(settings, checks={}), clips), key)
        self.assertIn(corpora.CMUDICT_PATH, tts.code_files('readings'))
        for name in ('g2pModel.js', 'context-labels.json', 'readings.py', 'frontend.py'):
            with unittest.mock.patch.object(tts, 'file_key', side_effect=lambda path, end=name: 'z' if path.endswith(end) else 'y'):
                changed = measure.measure_key(settings, clips)
            with unittest.mock.patch.object(tts, 'file_key', return_value='y'):
                self.assertNotEqual(changed, measure.measure_key(settings, clips), name)

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
        tasks = measure.clip_tasks(record, words, classes, controls, {'tokens': tokens, 'words': names})
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

        # A token aligned with its measured reading - frontEnd is still the front end's own
        aligned = tokens[:2] + ['IY1'] + tokens[3:]
        tasks = measure.clip_tasks(record_of(aligned, 5.0), words, classes, controls, {'tokens': tokens, 'words': names})
        self.assertEqual((tasks[0]['phones'], tasks[0]['frontEnd']), ('R IY1 D', 1))

    def test_control_words(self):
        counts = {word: 6 for word in ('went', 'home', 'look', 'times', 'cats', 'dark', 'night', 'read', 'lead', 'the', 'cat', 'zzzz')}
        counts['mats'] = 1
        variants = {word: [' '.join(phones) for phones in pronunciations] for word, pronunciations in DICTIONARY.items()}
        variants.update({'cat': ['K AE1 T'], 'mats': ['M AE1 T S']})
        measured = {'read': [['R IY1 D'], ['R EH1 D']]}
        self.assertEqual(measure.control_words(counts, variants, measured), {
            'went': ('ctc', ['W EH1 N T', 'W IY1 N T']), 'home': ('ctc', ['HH OW1 M', 'HH AW1 M']),
            'times': ('voicing', ['T AY1 M Z', 'T AY1 M S']), 'cats': ('voicing', ['K AE1 T S', 'K AE1 T Z'])
        })

        # At most CONTROL_WORDS words a method
        with unittest.mock.patch.object(measure, 'CONTROL_WORDS', 1):
            controls = measure.control_words(counts, variants, measured)
        self.assertEqual(sorted(method for method, _ in controls.values()), ['ctc', 'voicing'])

    def test_read_measured_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(readings.read_measured(directory))
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
            records, words = readings.read_measured(directory)
            self.assertEqual((len(records), sorted(words['words'])), (2, ['read', 'record']))
            self.assertEqual(readings.overrides(directory), {'a': {'1': 'R IY1 D'}})

    def test_measure_command(self):
        texts = [('male', text, 'train') for text in MALE_TEXTS] + [('female', text, 'val') for text in FEMALE_TEXTS]
        with ctc_fixtures.tts_corpus(texts) as (directory, _, clips), ctc_fixtures.fake_recognizer(), \
                tempfile.TemporaryDirectory() as temp:
            recipe = os.path.join(temp, 'train.json')
            with open(recipe, 'w', encoding='utf-8') as file:
                file.write('{}')
            with unittest.mock.patch.object(corpora, 'read_dictionary', lambda: DICTIONARY), \
                    unittest.mock.patch.object(readings, 'load_spec', return_value={}), \
                    unittest.mock.patch.object(readings, 'candidate_words', return_value={}), \
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
                self.assertNotIn('references', summary)
                measured, _ = readings.read_measured(directory)
                self.assertTrue(all('scores' in token for record in measured for token in record['tokens']))
                overrides = readings.overrides(directory)

                # Current - not measured again
                with captured() as (_, stderr):
                    self.assertEqual(measure.main(['--recipe', recipe]), 0)
                self.assertIn('is current', stderr.getvalue())

                # Another selection (a clip less) - measured again
                os.remove(tts.wav_path(directory, clips[-1]))
                with captured() as (_, stderr):
                    self.assertEqual(measure.main(['--recipe', recipe, '-j', '1']), 0)
                self.assertIn('measure: 13 clips', stderr.getvalue())
                self.assertNotEqual(readings.read_json(os.path.join(out, 'summary.json'))['key'], summary['key'])

                # A sample - measured, not kept as current
                with captured() as (_, stderr):
                    self.assertEqual(measure.main(['--recipe', recipe, '--limit', '3']), 0)
                self.assertIn('measure: 3 clips', stderr.getvalue())
                self.assertIsNone(readings.read_json(os.path.join(out, 'summary.json'))['key'])

        # The tokens - read, lead, and use by the CTC vote (its controls pass); record (weak), dawn (merger), and insult
        # (stress) unmeasured
        self.assertEqual(summary['clips'], 14)
        self.assertEqual(summary['key'], words['key'])
        self.assertEqual(summary['byClass'], {'strong': 3, 'voicing': 2})
        self.assertEqual(summary['votes'], {'strong': 3, 'voicing': 2})
        self.assertEqual(summary['controls'], {'ctc': {'votes': 4, 'false': 0, 'error': 0.0, 'used': True},
                                               'voicing': {'votes': 4, 'false': 0, 'error': 0.0, 'used': True}})
        self.assertEqual(summary['classes'], {'strong': True, 'voicing': True})
        self.assertEqual(words['classes'], {'read': 'strong', 'lead': 'strong', 'use': 'voicing', 'record': 'weak', 'dawn': 'merger',
                                            'insult': 'stress'})
        self.assertEqual([(record['text'], [(token['word'], token['frontEnd'], token['vote'], token['reading'])
                                            for token in record['tokens']])
                          for record in measured], [
            ('I read the lead, we went home.', [('read', 1, 0, 0), ('lead', 1, 0, 0)]),
            ('The cats use a record, the best went home.', [('use', 1, 0, 0)]),
            ('I read the use.', [('read', 1, 0, 0), ('use', 0, 0, 0)])
        ])
        self.assertEqual(len(measured[0]['tokens'][0]['scores']), 2)
        self.assertEqual(measured[2]['voice'], 'Harper')

        # The voice's tokens take the CTC readings that differ from the front end's
        self.assertEqual(overrides, {measured[0]['id']: {'1': 'R IY1 D', '3': 'L EH1 D'}, measured[1]['id']: {'2': 'Y UW1 S'},
                                     measured[2]['id']: {'1': 'R IY1 D'}})

    def test_main_module(self):
        code, _, stderr = run_module('picovox_train.measure', ['--bad'])
        self.assertEqual(code, 2)
        self.assertIn('unrecognized arguments: --bad', stderr)
