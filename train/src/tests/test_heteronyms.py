# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import datetime
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
import unittest.mock

from picovox_train import heteronyms, readings, tts

from .helpers import captured


# The measured words' readings (readings.py's words.json) - lead (relabeled), the (a function word, reduction), use (its
# first reading the majority), wind (no significant majority), record (unmeasured - every token abstained), that (a
# function word), family (a syllable), and solo (one reading)
WORDS = {
    'lead': [['L IY1 D'], ['L EH1 D']],
    'the': [['DH AH0'], ['DH IY0']],
    'use': [['Y UW1 S'], ['Y UW1 Z']],
    'wind': [['W IH1 N D'], ['W AY1 N D']],
    'record': [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']],
    'that': [['DH AE1 T'], ['DH AH0 T']],
    'family': [['F AE1 M L IY0'], ['F AE1 M AH0 L IY0']],
    'solo': [['S OW1 L OW0']]
}


# The measured tokens - (word, class, reading, book, voice) by word
TOKENS = [
    *[('lead', 'strong', 1, f'pg{number % 4}.txt', 'male' if number % 2 else 'female') for number in range(7)],
    ('lead', 'strong', 0, 'pg9.txt', 'male'), ('lead', 'strong', None, 'pg9.txt', 'male'),
    *[('the', 'weak', 1, f'pg{number}.txt', 'male') for number in range(4)],
    *[('use', 'voicing', 0, f'pg{number}.txt', 'female') for number in range(8)],
    ('wind', 'strong', 0, 'pg1.txt', 'male'), ('wind', 'strong', 1, 'pg2.txt', 'male'), ('wind', 'strong', 1, 'pg2.txt', 'male'),
    *[('record', 'weak', None, f'pg{number}.txt', 'male') for number in range(3)],
    *[('that', 'weak', 1, f'pg{number}.txt', 'male') for number in range(4)],
    *[('family', 'weak', 1, f'pg{number}.txt', 'male') for number in range(4)],
    ('solo', 'strong', 0, 'pg1.txt', 'male')
]


# The committed relabel file the selection keeps from - lead (relabeled again), wound (not measured), use (contradicted),
# record (unmeasured), that (its majority, rejected), family (another reading than the majority), and solo (one reading)
KEEP = {
    'relabel': {'lead': 'L EH1 D', 'wound': 'W AW1 N D', 'use': 'Y UW1 Z', 'record': 'R IH0 K AO1 R D', 'that': 'DH AH0 T',
                'family': 'F EY1 M L IY0', 'solo': 'S OW1 L AH0'},
    'evidence': {'wound': {'first': 'W UW1 N D'}, 'record': {'first': 'R EH1 K ER0 D'}}
}


class TestHeteronyms(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.corpus = os.path.join(self.directory, 'phonon-0123456789ab')
        self.out = os.path.join(self.directory, 'out')
        os.makedirs(self.out)

    def tearDown(self):
        shutil.rmtree(self.directory)

    # Write the corpus's measured readings - words.json and readings.jsonl (a clip a token)
    def write_measured(self, tokens=None):
        os.makedirs(os.path.join(self.corpus, 'readings'), exist_ok=True)
        with open(os.path.join(self.corpus, 'readings', 'words.json'), 'w', encoding='utf-8') as file:
            json.dump({'key': 'k', 'words': WORDS, 'classes': {}}, file)
        with open(os.path.join(self.corpus, 'readings', 'readings.jsonl'), 'w', encoding='utf-8') as file:
            for number, (word, word_class, reading, book, voice) in enumerate(TOKENS if tokens is None else tokens):
                token = {'position': 0, 'word': word, 'class': word_class, 'frontEnd': 0, 'vote': reading, 'reading': reading}
                file.write(json.dumps({'id': f'clip{number}', 'key': voice, 'book': book, 'tokens': [token]}) + '\n')

    def read_out(self, name):
        with open(os.path.join(self.out, name), encoding='utf-8') as file:
            return json.load(file)

    def test_base(self):
        self.assertEqual(readings.base('AH0'), 'AH')
        self.assertEqual(readings.base('K'), 'K')

    def test_edit_ops(self):
        self.assertEqual(readings.edit_ops(['L', 'IY1', 'D'], ['L', 'EH1', 'D']), [('IY1', 'EH1')])
        self.assertEqual(readings.edit_ops(['F', 'AO1', 'R', 'T'], ['F', 'AO1', 'R']), [('T', None)])
        self.assertEqual(readings.edit_ops(['F', 'AO1'], ['S', 'F', 'AO2']), [(None, 'S'), ('AO1', 'AO2')])
        self.assertEqual(readings.edit_ops([], ['K']), [(None, 'K')])
        cost, substitution = readings.edit_costs(['AH0'], ['AH1'])
        self.assertEqual(cost[1][1], 0.5)
        self.assertEqual(substitution('K', 'T'), 1)

    def test_wilson_lower(self):
        self.assertEqual(heteronyms.wilson_lower(0, 0), 0.0)
        self.assertAlmostEqual(heteronyms.wilson_lower(7, 8), 0.5291, places=4)
        self.assertLess(heteronyms.wilson_lower(1, 2), 0.5)

    def test_kind(self):
        self.assertEqual(readings.kind('R EH1 K ER0 D'.split(), 'R EH0 K ER1 D'.split()), 'stress')
        self.assertEqual(readings.kind('B AH0 L'.split(), 'B IH0 L'.split()), 'reduced')
        self.assertEqual(readings.kind('F AO1 R T'.split(), 'F AO1 R'.split()), 'consonant-drop')
        self.assertEqual(readings.kind('F AE1 M L IY0'.split(), 'F AE1 M AH0 L IY0'.split()), 'syllable')
        self.assertEqual(readings.kind('L IY1 D'.split(), 'L EH1 D'.split()), 'vowel')
        self.assertEqual(readings.kind('Y UW1 S'.split(), 'Y UW1 Z'.split()), 'consonant')

    def test_difference(self):
        self.assertEqual(readings.difference('L IY1 D'.split(), 'L EH1 D'.split()), (1, 1, 'vowel'))
        self.assertEqual(readings.difference('Y UW1 S'.split(), 'Y UW1 Z'.split()), (2, 2, 'voicing'))
        self.assertEqual(readings.difference('F AO1 R T'.split(), 'F AO1 R'.split()), None)
        self.assertEqual(readings.difference('F AO1 R'.split(), 'S F AO1 R'.split()), None)
        self.assertEqual(readings.difference('F AO1 R'.split(), 'F AO1 R T'.split()), None)

    def test_lexical(self):
        self.assertTrue(heteronyms.lexical('L IY1 D', 'L EH1 D'))
        self.assertFalse(heteronyms.lexical('R EH1 K ER0 D', 'R EH0 K ER1 D'))
        self.assertTrue(heteronyms.lexical('K AA1 N T EH0 N T', 'K AH0 N T EH1 N T'))
        self.assertFalse(heteronyms.lexical('B AH0 L', 'B IH0 L'))
        self.assertTrue(heteronyms.lexical('F AE1 M L IY0', 'F EY1 M AH0 L IY0'))

    def test_votes(self):
        self.write_measured()
        self.assertEqual(heteronyms.votes(self.out, self.corpus), '8 measured words, 36 tokens')
        votes = self.read_out('votes.json')
        self.assertEqual(votes['corpus'], 'phonon-0123456789ab')
        self.assertEqual(votes['words']['lead'], {
            'variants': ['L IY1 D', 'L EH1 D'], 'class': 'strong', 'tokens': 9, 'abstained': 1, 'votes': [1, 7],
            'books': [['pg9.txt'], ['pg0.txt', 'pg1.txt', 'pg2.txt', 'pg3.txt']], 'voices': [['male'], ['female', 'male']]
        })
        self.assertEqual(votes['words']['record']['votes'], [0, 0])
        self.assertEqual(votes['words']['record']['abstained'], 3)

        # (the corpus not measured)
        shutil.rmtree(os.path.join(self.corpus, 'readings'))
        with self.assertRaises(FileNotFoundError):
            heteronyms.votes(self.out, self.corpus)

    def test_significant(self):
        books = [['pg1', 'pg2', 'pg3'], ['pg1', 'pg2', 'pg3']]
        self.assertEqual(heteronyms.significant({'votes': [1, 9, 2], 'books': books + [['pg1']]}), 1)
        self.assertEqual(heteronyms.significant({'votes': [1, 2, 9], 'books': [['pg1'], ['pg1'], books[0]]}), 2)
        self.assertEqual(heteronyms.significant({'votes': [9, 1], 'books': books}), 0)
        # (too few books either way; no significant majority)
        self.assertIsNone(heteronyms.significant({'votes': [1, 9], 'books': [['pg1'], ['pg1', 'pg2']]}))
        self.assertIsNone(heteronyms.significant({'votes': [9, 1], 'books': [['pg1', 'pg2'], ['pg1']]}))
        self.assertIsNone(heteronyms.significant({'votes': [3, 4], 'books': books}))
        self.assertIsNone(heteronyms.significant({'votes': [0, 0], 'books': [[], []]}))

    def test_candidates(self):
        self.write_measured()
        heteronyms.votes(self.out, self.corpus)
        self.assertEqual(heteronyms.candidates(self.out), '5 words with a non-first majority, 4 significant')
        items = self.read_out('candidates.json')
        self.assertEqual([(item['word'], item['significant'], item['kind']) for item in items],
                         [('lead', True, 'vowel'), ('the', True, 'reduced'), ('that', True, 'vowel'), ('family', True, 'syllable'),
                          ('wind', False, 'vowel')])
        self.assertEqual({key: value for key, value in items[0].items() if key != 'lower'}, {
            'word': 'lead', 'first': 'L IY1 D', 'majority': 'L EH1 D', 'index': 1, 'class': 'strong', 'votes': [1, 7],
            'books': [1, 4], 'voices': [1, 2], 'significant': True, 'kind': 'vowel'
        })
        self.assertAlmostEqual(items[0]['lower'], 0.5291, places=4)

    def test_select(self):
        self.write_measured()
        heteronyms.votes(self.out, self.corpus)
        heteronyms.candidates(self.out)
        self.assertEqual(heteronyms.select(self.out, 0.5),
                         '1 relabeled of 4 significant candidates, 0 kept, 0 contradicted: lead L EH1 D')
        result = self.read_out('relabel.json')
        self.assertEqual(result['relabel'], {'lead': 'L EH1 D'})
        self.assertEqual([item['word'] for item in result['chosen']], ['lead'])
        self.assertEqual({item['word']: item['reasons'] for item in result['rejected']},
                         {'the': ['function word', 'kind reduced'], 'that': ['function word', 'reduction only'],
                          'family': ['kind syllable']})
        self.assertEqual((result['kept'], result['contradicted']), ([], []))

        # The least share above the majority's
        heteronyms.select(self.out, 0.9)
        result = self.read_out('relabel.json')
        self.assertEqual(result['relabel'], {})
        self.assertEqual((result['rejected'][0]['word'], result['rejected'][0]['reasons']), ('lead', ['share 0.88']))

    def test_select_keep(self):
        self.write_measured()
        heteronyms.votes(self.out, self.corpus)
        heteronyms.candidates(self.out)
        self.assertEqual(heteronyms.select(self.out, 0.5, KEEP),
                         '1 relabeled of 4 significant candidates, 4 kept, 1 contradicted: lead L EH1 D')
        result = self.read_out('relabel.json')
        self.assertEqual(result['relabel'], {'lead': 'L EH1 D', 'wound': 'W AW1 N D', 'record': 'R IH0 K AO1 R D',
                                             'family': 'F EY1 M L IY0', 'solo': 'S OW1 L AH0'})
        self.assertEqual(result['kept'], [
            {'word': 'wound', 'phones': 'W AW1 N D', 'evidence': {'first': 'W UW1 N D'}, 'votes': None},
            {'word': 'record', 'phones': 'R IH0 K AO1 R D', 'evidence': {'first': 'R EH1 K ER0 D'}, 'votes': [0, 0]},
            {'word': 'family', 'phones': 'F EY1 M L IY0', 'evidence': None, 'votes': [0, 4]},
            {'word': 'solo', 'phones': 'S OW1 L AH0', 'evidence': None, 'votes': [1]}
        ])
        self.assertEqual(result['contradicted'], [{'word': 'use', 'phones': 'Y UW1 Z', 'votes': [8, 0]}])

    def test_select_context_and_names(self):
        # A context word isn't relabeled - a committed one kept, whatever the corpus says; a name isn't relabeled
        self.write_measured()
        heteronyms.votes(self.out, self.corpus)
        heteronyms.candidates(self.out)
        heteronyms.select(self.out, 0.5, KEEP, context={'lead', 'use'})
        result = self.read_out('relabel.json')
        self.assertEqual(result['relabel']['lead'], 'L EH1 D')
        self.assertEqual(result['relabel']['use'], 'Y UW1 Z')
        self.assertEqual({item['word']: item['reasons'] for item in result['rejected']}['lead'], ['a context word'])
        self.assertEqual(result['contradicted'], [])
        heteronyms.select(self.out, 0.5, names={'lead'})
        result = self.read_out('relabel.json')
        self.assertEqual(result['relabel'], {})
        self.assertEqual({item['word']: item['reasons'] for item in result['rejected']}['lead'], ['a name'])

    def test_books_dir(self):
        self.assertEqual(heteronyms.books_dir(), heteronyms.books_dir())
        self.assertTrue(heteronyms.books_dir().endswith(os.path.join('build', 'nn', 'books')))

    def test_committed(self):
        result = {'relabel': {'lead': 'L EH1 D', 'wound': 'W AW1 N D'},
                  'chosen': [{'word': 'lead', 'first': 'L IY1 D', 'votes': [1, 7], 'books': [1, 4], 'class': 'strong', 'kind': 'vowel'}],
                  'kept': [{'word': 'wound', 'phones': 'W AW1 N D', 'evidence': {'first': 'W UW1 N D'}, 'votes': None}]}
        committed = heteronyms.committed(result, 'phonon-0123456789ab')
        self.assertEqual(committed, {
            'description': heteronyms.DESCRIPTION, 'corpus': 'phonon-0123456789ab', 'relabel': result['relabel'],
            'evidence': {
                'lead': {'first': 'L IY1 D', 'votes': [1, 7], 'books': [1, 4], 'class': 'strong', 'corpus': 'phonon-0123456789ab',
                         'date': datetime.date.today().isoformat()},
                'wound': {'kept': {'first': 'W UW1 N D'}, 'votes': None}
            }
        })

    def test_main(self):
        self.write_measured()
        keep = os.path.join(self.directory, 'keep.json')
        with open(keep, 'w', encoding='utf-8') as file:
            json.dump(KEEP, file)
        written = os.path.join(self.directory, 'g2p-relabel.json')
        books = os.path.join(self.directory, 'books')
        os.makedirs(books)
        with open(os.path.join(books, 'pg1.txt'), 'w', encoding='utf-8') as file:
            file.write('The solo was long. A Solo spoke.\n')
        with unittest.mock.patch.object(tts, 'load_settings', return_value={'engine': {}}), \
                unittest.mock.patch.object(tts, 'corpus_dir', return_value=self.corpus) as corpus_dir, \
                unittest.mock.patch.object(readings, 'load_spec', return_value={}), \
                unittest.mock.patch.object(readings, 'candidate_words', return_value={}), \
                unittest.mock.patch.object(heteronyms, 'books_dir', return_value=books):

            # The selection, kept from the committed file and written as it
            with captured() as (_, stderr):
                self.assertEqual(heteronyms.main(['--out', self.out, '--keep', keep, '--write', written]), 0)
            corpus_dir.assert_called_with({'engine': {}})
            self.assertIn('votes: 8 measured words, 36 tokens', stderr.getvalue())
            self.assertIn('candidates: 5 words with a non-first majority, 4 significant', stderr.getvalue())
            self.assertIn('select: 1 relabeled of 4 significant candidates, 4 kept, 1 contradicted: lead L EH1 D', stderr.getvalue())
            with open(written, encoding='utf-8') as file:
                text = file.read()
            self.assertTrue(text.endswith('}\n'))
            committed = json.loads(text)
            self.assertEqual(committed['corpus'], 'phonon-0123456789ab')
            self.assertEqual(sorted(committed['relabel']), ['family', 'lead', 'record', 'solo', 'wound'])
            self.assertEqual(committed['evidence']['wound'], {'kept': {'first': 'W UW1 N D'}, 'votes': None})

            # The check - equal
            with captured() as (_, stderr):
                self.assertEqual(heteronyms.main(['--out', self.out, '--keep', keep, '--check', written]), 0)
            self.assertIn(f'check: the 5 relabeled words equal {written}\'s', stderr.getvalue())

            # The check - different (without the committed words kept)
            with captured() as (_, stderr):
                self.assertEqual(heteronyms.main(['--out', self.out, '--check', written]), 1)
            self.assertIn(f'check: the relabeled words differ from {written}\'s (corpus phonon-0123456789ab, theirs '
                          'phonon-0123456789ab): family None (was F EY1 M L IY0), record None (was R IH0 K AO1 R D), '
                          'solo None (was S OW1 L AH0), wound None (was W AW1 N D)', stderr.getvalue())

            # Neither written nor checked
            with captured():
                self.assertEqual(heteronyms.main(['--out', self.out]), 0)
            self.assertEqual(self.read_out('relabel.json')['relabel'], {'lead': 'L EH1 D'})

    def test_main_module(self):
        with unittest.mock.patch.dict(sys.modules), unittest.mock.patch.object(sys, 'argv', ['heteronyms', '--help']), \
                captured() as (stdout, _):
            sys.modules.pop('picovox_train.heteronyms')
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.heteronyms', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('relabeled words', stdout.getvalue())
