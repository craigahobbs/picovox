# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring, protected-access

import json
import os
import runpy
import sys
import tempfile
import unittest
import unittest.mock
import warnings

from picovox_train import data, frontend, readings

from .helpers import captured


# The test pronouncer's words
PHONES = {'i': ['AY1'], 'read': ['R IY1 D'], 'the': ['DH AH0'], 'use': ['Y UW1 S'], 'book': ['B UH1 K'], 'it': ['IH1 T']}


# The test pronouncer
def pronounce(word):
    return PHONES.get(word, ['K AE1 T'])[0].split()


# The test inventory
READINGS = {'read': [['R IY1 D'], ['R EH1 D']], 'use': [['Y UW1 S'], ['Y UW1 Z']], 'book': [['B UH1 K'], ['B UW1 K']]}


# Write an MFA label file - [(start, end, label)]
def write_lab(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(''.join(f'{start}\t{end}\t{label}\n' for start, end, label in rows))


# Write the word and phone labels of "I read the use book" - read as R EH1 D, use as Y UW1 Z, book as B OW1 K (no
# variant), it as IH1 T (not in the inventory)
def write_labels(phone_path):
    write_lab(readings.word_lab_path(phone_path), [(0.0, 0.1, 'i'), (0.1, 0.4, 'read'), (0.4, 0.5, 'the'), (0.5, 0.8, 'use'),
                                                   (0.8, 1.0, 'book'), (1.0, 1.2, 'it'), (1.2, 1.3, '')])
    write_lab(phone_path, [(0.0, 0.1, 'AY1'), (0.1, 0.2, 'R'), (0.2, 0.3, 'EH1'), (0.3, 0.4, 'D'), (0.4, 0.45, 'DH'),
                           (0.45, 0.5, 'AH0'), (0.5, 0.6, 'Y'), (0.6, 0.7, 'UW1'), (0.7, 0.8, 'Z'), (0.8, 0.85, 'B'),
                           (0.85, 0.9, 'OW1'), (0.9, 1.0, 'K'), (1.0, 1.1, 'IH1'), (1.1, 1.2, 'T'), (1.2, 1.3, 'sil'),
                           (1.3, 1.4, '')])


class TestReadings(unittest.TestCase):

    def test_base(self):
        self.assertEqual(readings.base('AH0'), 'AH')
        self.assertEqual(readings.base('K'), 'K')
        self.assertEqual(readings.unstressed('R IY1 D'), 'R IY D')

    def test_edit_ops(self):
        self.assertEqual(readings.edit_ops('R IY1 D'.split(), 'R IY1 D'.split()), [])
        self.assertEqual(readings.edit_ops('R IY1 D'.split(), 'R IY0 D'.split()), [('IY1', 'IY0')])
        self.assertEqual(readings.edit_ops('K AE1 T'.split(), 'K AE1 T S'.split()), [(None, 'S')])
        self.assertEqual(readings.edit_ops('K AE1 T S'.split(), 'K AE1 T'.split()), [('S', None)])
        self.assertEqual(readings.edit_ops('S K AE1 T'.split(), 'K AE1 T'.split()), [('S', None)])
        self.assertEqual(readings.edit_ops([], ['K']), [(None, 'K')])

    def test_consonant_change(self):
        self.assertFalse(readings.consonant_change('R IY1 D'.split(), 'R IY0 D'.split()))
        self.assertFalse(readings.consonant_change('R AH0 D'.split(), 'R IH0 D'.split()))
        self.assertFalse(readings.consonant_change('K AE1 T'.split(), 'K AE1 T S'.split()))
        self.assertFalse(readings.consonant_change('K AE1 T'.split(), 'K AE1 T AH0'.split()))
        self.assertFalse(readings.consonant_change('F AY1 ER0'.split(), 'F AY1 R'.split()))
        self.assertFalse(readings.consonant_change('R IY1 D'.split(), 'R EH1 D'.split()))
        self.assertTrue(readings.consonant_change('K L OW1 S'.split(), 'K L OW1 Z'.split()))

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
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'cmudict.dict')
            with open(path, 'w', encoding='utf-8') as file:
                file.write(';;; a comment\nread R IY1 D\nread(2) R EH1 D\nread(3) R IY1 D # a repeat\n\n'
                           'to T UW1\nto(2) T AH0\nthe DH AH0\nthe(2) DH IY1\nlead L IY1 D\nlead(2) L EH1 D\ncat K AE1 T\ncat(2) K AE0 T\n')
            variants = readings.read_variants(path)
            self.assertEqual(variants['read'], ['R IY1 D', 'R EH1 D'])
            self.assertEqual(variants['cat'], ['K AE1 T', 'K AE0 T'])
            self.assertEqual(readings.inventory(variants), {'read': [['R IY1 D'], ['R EH1 D']], 'lead': [['L IY1 D'], ['L EH1 D']]})
            with unittest.mock.patch.object(readings, 'DICTIONARY_PATH', path), \
                    unittest.mock.patch.object(readings.read_variants, '__defaults__', (path,)):
                self.assertEqual(sorted(readings.inventory()), ['lead', 'read'])

    def test_sz_only(self):
        self.assertTrue(readings.sz_only('Y UW1 S', 'Y UW1 Z'))
        self.assertTrue(readings.sz_only('Y UW1 S', 'Y UW0 Z'))
        self.assertFalse(readings.sz_only('Y UW1 S', 'Y UW1 S'))
        self.assertFalse(readings.sz_only('K L OW1 S', 'K L AW1 Z'))
        self.assertFalse(readings.sz_only('Y UW1 S', 'Y UW1 Z D'))

    def test_match_variant(self):
        variants = ['R IY1 D', 'R EH1 D', 'R IY0 D IY0', 'R IY1 D IY2']
        self.assertEqual(readings.match_variant('R EH1 D', variants), 1)
        self.assertEqual(readings.match_variant('R EH0 D', variants), 1)
        self.assertIsNone(readings.match_variant('R IY2 D IY1', variants))
        self.assertIsNone(readings.match_variant('K AE1 T', variants))

    def test_reading_of(self):
        word_readings = [['R IY1 D', 'R IY0 D IY0'], ['R EH1 D'], ['R IY2 D IY2']]
        self.assertEqual(readings.reading_of('R EH1 D', word_readings), 1)
        self.assertEqual(readings.reading_of('R EH0 D', word_readings), 1)
        self.assertIsNone(readings.reading_of('R IY1 D IY1', word_readings))
        self.assertIsNone(readings.reading_of('K AE1 T', word_readings))

    def test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'lab', 'phone', 'x.lab')
            write_lab(path, [(0.0, 0.1, 'K'), (0.1, 0.2, '')])
            with open(path, 'a', encoding='utf-8') as file:
                file.write('0.2\t0.3\n')
            self.assertEqual(readings.read_lab(path), [(0.0, 0.1, 'K')])
        self.assertEqual(readings.word_lab_path(os.path.join('a', 'lab', 'phone', 'b', 'phone', 'c.lab')),
                         os.path.join('a', 'lab', 'word', 'b', 'phone', 'c.lab'))

    def test_front_end_words(self):
        self.assertEqual(readings.front_end_words('I read it.', pronounce),
                         [('i', 'AY0'), ('read', 'R IY1 D'), ('it', 'IH0 T')])

    def test_utterance_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            phone_path = os.path.join(directory, 'lab', 'phone', 'u.lab')
            self.assertEqual(readings.utterance_readings('I read.', phone_path, READINGS, pronounce), ({}, []))
            write_labels(phone_path)
            overrides, tokens = readings.utterance_readings('I read the use, book it.', phone_path, READINGS, pronounce)
        self.assertEqual(overrides, {1: ['R', 'EH1', 'D']})
        self.assertEqual(tokens, [
            {'position': 1, 'word': 'read', 'reading': 1, 'frontEnd': 0, 'variant': 'R EH1 D', 'szOnly': False},
            {'position': 3, 'word': 'use', 'reading': 1, 'frontEnd': 0, 'variant': 'Y UW1 Z', 'szOnly': True}
        ])

    def test_token_words(self):
        boundaries = {'<bos>', '_', '<period>', '<comma>'}
        self.assertEqual(readings.token_words(['<bos>', 'AY1', '_', 'R', 'EH1', 'D', '<comma>', 'IH1', 'T', '<period>'], boundaries),
                         ['AY1', 'R EH1 D', 'IH1 T'])

    def test_count_disagreements(self):
        boundaries = {'<bos>', '_', '<period>'}
        aligned = [{'position': 1, 'word': 'read', 'reading': 1, 'variant': 'R EH1 D'},
                   {'position': 3, 'word': 'use', 'reading': 1, 'variant': 'Y UW1 Z'},
                   {'position': 4, 'word': 'book', 'reading': 0, 'variant': 'B UH1 K'},
                   {'position': 9, 'word': 'book', 'reading': 0, 'variant': 'B UH1 K'}]
        totals = {'tokens': 0, 'disagree': 0}
        tokens = '<bos> AY1 _ R EH1 D _ DH AH0 _ Y UW1 S _ B UW1 K <period>'.split()
        readings.count_disagreements(tokens, aligned, READINGS, boundaries, totals)
        self.assertEqual(totals, {'tokens': 3, 'disagree': 2, 'szOnly': 1})

    def test_attach_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            phone_path = os.path.join(directory, 'lab', 'phone', 'u.lab')
            write_labels(phone_path)
            utterance = {'labelFormat': 'mfa', 'text': 'I read the use, book it.', 'lab': phone_path}
            arctic = {'labelFormat': 'xwaves', 'text': 'I read.', 'lab': phone_path}
            with unittest.mock.patch.object(readings, 'inventory', return_value=READINGS), \
                    unittest.mock.patch.object(readings, 'text_to_phrases',
                                               lambda text, pronounce_word=None: frontend.text_to_phrases(text, pronounce)):
                counts = readings.attach_readings([(utterance, None), (arctic, None)])
        self.assertEqual(counts, {'utterances': 1, 'tokens': 2, 'overrides': 1})
        self.assertEqual(utterance['readings'], {1: ['R', 'EH1', 'D']})
        self.assertEqual(len(utterance['alignedReadings']), 2)
        self.assertNotIn('readings', arctic)

    def test_check_built(self):
        totals = {'tokens': 0, 'disagree': 0}
        readings.check_built({}, [], totals)
        self.assertEqual(totals, {'tokens': 0, 'disagree': 0})
        utterance = {'alignedReadings': [{'position': 0, 'word': 'read', 'reading': 1, 'variant': 'R EH1 D'}]}
        with unittest.mock.patch.dict(readings._INVENTORY_CACHE, clear=True), \
                unittest.mock.patch.object(readings, 'inventory', return_value=READINGS) as inventory:
            readings.check_built(utterance, ['<bos>', 'R', 'EH1', 'D', '<period>'], totals)
            readings.check_built(utterance, ['<bos>', 'R', 'IY1', 'D', '<period>'], totals)
        self.assertEqual(inventory.call_count, 1)
        self.assertEqual(totals, {'tokens': 2, 'disagree': 1})

    def test_check_command(self):
        with tempfile.TemporaryDirectory() as directory:
            label_dir = os.path.join(directory, 'lab', 'phone')
            build = os.path.join(directory, 'build')
            os.makedirs(build)
            write_labels(os.path.join(label_dir, 'dev-clean', '84', '121123', '84_121123_000008_000000.lab'))
            entries = [
                {'id': 'libritts-84/84_121123_000008_000000', 'speaker': 'libritts-84', 'sentence': '84_121123_000008_000000',
                 'text': 'I read the use, book it.', 'include': True},
                {'id': 'libritts-84/84_121123_000009_000000', 'speaker': 'libritts-84', 'sentence': '84_121123_000009_000000',
                 'text': 'I read.', 'include': True},
                {'id': 'libritts-84/84_121123_000010_000000', 'speaker': 'libritts-84', 'sentence': '84_121123_000010_000000',
                 'text': 'I read.', 'include': False},
                {'id': 'arctic-slt/arctic_a0001', 'speaker': 'arctic-slt', 'sentence': 'arctic_a0001', 'text': 'I read.',
                 'include': True}
            ]
            with open(os.path.join(build, 'index.json'), 'w', encoding='utf-8') as file:
                json.dump({'utterances': entries}, file)

            # The built tokens - read as the aligner heard it, use the front end's (S/Z only), book the other reading
            def write_alignment(read_phones):
                with open(os.path.join(build, 'alignment.jsonl'), 'w', encoding='utf-8') as file:
                    tokens = f'<bos> AY1 _ {read_phones} _ DH AH0 _ Y UW1 S <comma> B UW1 K _ IH1 T <period>'
                    for entry in entries:
                        file.write(json.dumps({'id': entry['id'], 'tokens': tokens, 'readings': {'1': 'R EH1 D'}}) + '\n')

            with unittest.mock.patch.object(data, 'LABEL_DIR', label_dir), \
                    unittest.mock.patch.object(readings, 'inventory', return_value=READINGS):
                write_alignment('R EH1 D')
                with captured() as (stdout, _):
                    self.assertEqual(readings.main(['check', '--out', build]), 0)
                self.assertEqual(stdout.getvalue(), f'readings check {build}: 1 heteronym tokens, 0 (0.00%) disagree with the aligner, '
                                                    '2 overrides; 1 tokens differ from the aligner only by S/Z (the front end\'s '
                                                    'reading kept)\n')
                write_alignment('R IY1 D')
                with captured() as (stdout, _):
                    self.assertEqual(readings.main(['check', '--out', build]), 1)
                self.assertIn('1 (100.00%) disagree', stdout.getvalue())
                self.assertIn(' - FAILED (over 1%)', stdout.getvalue())

    def test_main_module(self):
        with unittest.mock.patch.object(sys, 'argv', ['readings']), captured() as (_, stderr), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaises(SystemExit) as context:
                runpy.run_module('picovox_train.readings', run_name='__main__')
        self.assertEqual(context.exception.code, 2)
        self.assertIn('the following arguments are required', stderr.getvalue())
