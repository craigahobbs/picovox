# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import hashlib
import json
import os
import re
import runpy
import shutil
import sys
import tempfile
import types
import unittest
import unittest.mock

from picovox_train import contextlabels, corpora, g2p, readings, tts
from picovox_train.books import load_pins

from . import ctc_fixtures, data_fixtures, helpers
from .helpers import captured


# A fake spaCy (build/venv-train only): the tagger tags "read" VBD after "had" or "I", else VB, "lives" NNS, other words NN
class FakeToken:
    def __init__(self, idx, tag):
        self.idx = idx
        self.tag_ = tag


class FakeNlp:
    def __init__(self):
        self.calls = []

    def pipe(self, texts, batch_size):
        self.calls.append((list(texts), batch_size))
        return [self.tag(text) for text in texts]

    @staticmethod
    def tag(text):
        tokens = []
        previous = ''
        for match in re.finditer(r"[A-Za-z']+", text):
            word = match.group(0).lower()
            if word == 'read':
                tag = 'VBD' if previous in ('had', 'i') else 'VB'
            else:
                tag = 'NNS' if word == 'lives' else 'NN'
            tokens.append(FakeToken(match.start(), tag))
            previous = word
        return tokens


# The tiny candidate words
def tiny_words():
    return {
        'read': {'readings': [['R IY1 D'], ['R EH1 D']], 'family': 'own', 'curated': True,
                 'tags': {'VBD<!and|or|^': 1, 'VB<will|^': 0, 'VBN': 1}},
        'lives': {'readings': [['L IH1 V Z'], ['L AY1 V Z']], 'family': 'own', 'curated': True, 'tags': {'NNS': 1, 'VBZ': 0}},
        'record': {'readings': [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']], 'family': 'stress', 'curated': False, 'tags': {}},
        'lead': {'readings': [['L IY1 D'], ['L EH1 D']], 'family': 'own', 'curated': True, 'tags': {}},
        'bass': {'readings': [['B AE1 S'], ['B EY1 S']], 'family': 'own', 'curated': True, 'tags': {}},
        'excluded': {'readings': [['EH1 K S'], ['IH0 K S']], 'family': 'own', 'curated': True, 'tags': {}},
        'nodefault': {'readings': [['N OW1'], ['N AA1']], 'family': 'own', 'curated': True, 'tags': {}}
    }


class TestContextLabels(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def path(self, *parts):
        return os.path.join(self.directory, *parts)

    def write(self, path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text)

    def test_settings(self):
        settings = contextlabels.load_settings()
        self.assertEqual(settings['window'], contextlabels.WINDOW)
        self.assertIn('pg1342', contextlabels.HELD_OUT_BOOKS)
        self.assertIn('words', contextlabels.load_spec())

    def test_vowels(self):
        self.assertEqual(readings.variant_vowels('R IH0 K AO1 R D'), ['IH0', 'AO1'])
        self.assertEqual(readings.primary_vowel('R IH0 K AO1 R D'), 1)
        self.assertIsNone(readings.primary_vowel('S T'))

    def test_word_family(self):
        self.assertEqual(readings.word_family('record', [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']]), 'stress')
        self.assertEqual(readings.word_family('estimate', [['EH1 S T AH0 M AH0 T'], ['EH1 S T AH0 M EY2 T']]), 'ate')
        self.assertEqual(readings.word_family('estimates', [['EH1 S T AH0 M AH0 T S'], ['EH1 S T AH0 M EY2 T S']]), 'ate')
        self.assertEqual(readings.word_family('use', [['Y UW1 S'], ['Y UW1 Z']]), 'consonant')
        self.assertEqual(readings.word_family('lead', [['L IY1 D'], ['L EH1 D']]), 'vowel')
        # (-ate with both readings' full -ate vowels is no -ate pair)
        self.assertEqual(readings.word_family('abate', [['AH0 B EY2 T'], ['IH0 B EY2 T']]), 'vowel')

    def test_family_reading(self):
        stress = [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']]
        self.assertEqual(contextlabels.family_reading('stress', 'NN', stress), 0)
        self.assertEqual(contextlabels.family_reading('stress', 'VBD', stress), 1)
        self.assertIsNone(contextlabels.family_reading('stress', 'JJ', stress))
        # (a reading without a primary stress - unmapped)
        self.assertIsNone(contextlabels.family_reading('stress', 'NN', [['R EH2 K ER0 D'], ['R IH0 K AO1 R D']]))
        ate = [['EH1 S T AH0 M AH0 T'], ['EH1 S T AH0 M EY2 T']]
        self.assertEqual(contextlabels.family_reading('ate', 'VB', ate), 1)
        self.assertEqual(contextlabels.family_reading('ate', 'JJ', ate), 0)
        self.assertIsNone(contextlabels.family_reading('ate', 'RB', ate))
        self.assertIsNone(contextlabels.family_reading('ate', 'VB', [['EH1 S T AH0 M AH0 T']]))
        self.assertIsNone(contextlabels.family_reading('ate', 'NN', [['EH1 S T AH0 M EY2 T']]))
        use = [['Y UW1 S'], ['Y UW1 Z']]
        self.assertEqual(contextlabels.family_reading('consonant', 'VB', use), 1)
        self.assertEqual(contextlabels.family_reading('consonant', 'NNS', use), 0)
        self.assertIsNone(contextlabels.family_reading('consonant', 'JJ', use))
        self.assertEqual(contextlabels.family_reading('consonant', 'VB', [['Y UW1 Z'], ['Y UW1 S']]), 0)
        self.assertIsNone(contextlabels.family_reading('vowel', 'NN', use))

    def test_voicing_pair(self):
        self.assertTrue(readings.voicing_pair([['Y UW1 S'], ['Y UW1 Z']]))
        self.assertFalse(readings.voicing_pair([['S AH0 L EH1 K T'], ['Z AH0 L EH1 K T']]))
        self.assertFalse(readings.voicing_pair([['K L OW1 S'], ['K L OW1 Z IH0']]))
        self.assertFalse(readings.voicing_pair([['K L OW1 S'], ['K L OW1 T']]))

    @unittest.mock.patch.object(corpora, 'read_dictionary', lambda: helpers.DICTIONARY)
    def test_candidate_words(self):
        inventory = {
            'record': [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']], 'use': [['Y UW1 S'], ['Y UW1 Z']],
            'lead': [['L IY1 D'], ['L EH1 D']], 'select': [['S AH0 L EH1 K T'], ['Z AH0 L EH1 K T']],
            'present': [['P R EH1 Z AH0 N T'], ['P R IY0 Z EH1 N T']]
        }
        spec = {'exclude': ['present'], 'words': {
            'lead': {'tags': {'L EH1 D': ['NN'], 'L IY1 D': ['VB', 'VBP']}},
            'does': {'missing': ['D OW1 Z'], 'family': 'own'}
        }}
        words = contextlabels.candidate_words(spec, inventory)
        self.assertEqual(sorted(words), ['does', 'lead', 'record', 'use'])
        self.assertEqual(words['record']['family'], 'stress')
        self.assertEqual(words['use'], {'readings': [['Y UW1 S'], ['Y UW1 Z']], 'family': 'consonant', 'tags': {}, 'curated': False})
        self.assertEqual(words['lead'], {'readings': [['L IY1 D'], ['L EH1 D']], 'family': 'own', 'tags': {'NN': 1, 'VB': 0, 'VBP': 0},
                                         'curated': True})
        self.assertEqual(words['does']['readings'][-1], ['D OW1 Z'])
        self.assertTrue(words['does']['curated'])

        # The dictionary's inventory by default
        words = contextlabels.candidate_words({'exclude': [], 'words': {}})
        self.assertEqual(words['record']['family'], 'stress')

    def test_tag_reading(self):
        read = tiny_words()['read']
        self.assertEqual(contextlabels.tag_reading(read, 'VBD', 'read', 'had'), 1)
        self.assertIsNone(contextlabels.tag_reading(read, 'VBD', 'read', 'and'))
        self.assertIsNone(contextlabels.tag_reading(read, 'VBD', 'read', ','))
        self.assertEqual(contextlabels.tag_reading(read, 'VB', 'read', 'will'), 0)
        self.assertEqual(contextlabels.tag_reading(read, 'VB', 'read', '.'), 0)
        self.assertIsNone(contextlabels.tag_reading(read, 'VB', 'read', 'they'))
        self.assertEqual(contextlabels.tag_reading(read, 'VBN'), 1)
        self.assertIsNone(contextlabels.tag_reading(read, 'NN'))

        # Capitalized keys, and a family's mapping after the word's own (blocked by an own key of the tag)
        entry = {'readings': [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']], 'family': 'stress', 'tags': {'CapNNP': 1, 'VB<to': 0}}
        self.assertEqual(contextlabels.tag_reading(entry, 'NNP', 'Record'), 1)
        self.assertIsNone(contextlabels.tag_reading(entry, 'NNP', 'record'))
        self.assertIsNone(contextlabels.tag_reading(entry, 'VB', 'record', 'we'))
        self.assertEqual(contextlabels.tag_reading(entry, 'NN', 'record'), 0)
        self.assertEqual(contextlabels.tag_reading(entry, 'VBD'), 1)

    def test_context_items(self):
        self.assertEqual(contextlabels.previous_items('I read it, then read.'), {0: '', 2: 'i', 7: 'read', 11: ',', 16: 'then'})
        sequence = ['i', 'read', 'it', '.']
        self.assertEqual(contextlabels.context_window(sequence, 1), ['', '', 'i', 'read', 'it', '.', ''])
        self.assertEqual(contextlabels.paragraphs('One\ntwo.\n\n\n  Three  four.\n\n'), ['One two.', 'Three four.'])

    # The measured clips (readings.py's readings.jsonl) - a candidate token, a token without a measured reading, and a word
    # that isn't a candidate
    def write_readings(self, directory):
        def token(position, word, reading):
            return {'position': position, 'word': word, 'class': 'strong', 'frontEnd': 0, 'vote': reading, 'reading': reading}
        records = [
            {'id': 'c1', 'key': 'male', 'voice': 'Trevor', 'sex': 'male', 'book': 'pg1.txt', 'text': 'I had read it.', 'split': 'train',
             'tokens': [token(2, 'read', 1)]},
            {'id': 'c2', 'key': 'female', 'voice': 'Harper', 'sex': 'female', 'book': 'pg2.txt', 'text': 'Who lives here?',
             'split': 'val', 'tokens': [token(0, 'who', 0), token(1, 'lives', 0)]},
            {'id': 'c3', 'key': 'male', 'voice': 'Trevor', 'sex': 'male', 'book': 'pg1.txt', 'text': 'Nothing to read.', 'split': 'train',
             'tokens': [token(2, 'read', None)]}
        ]
        self.write(os.path.join(directory, 'readings', 'readings.jsonl'), ''.join(json.dumps(record) + '\n' for record in records))
        self.write(os.path.join(directory, 'readings', 'words.json'), json.dumps({'words': {}}))

    def test_aligned(self):
        settings = ctc_fixtures.tts_settings()
        out = self.path('out')
        with unittest.mock.patch.object(tts, 'TTS_DIR', self.path('tts')), \
                unittest.mock.patch.object(tts, 'load_settings', return_value=settings), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()):
            # The corpus's readings not measured
            with captured() as (stdout, stderr):
                self.assertEqual(contextlabels.main(['aligned', '--out', out]), 1)
            self.assertEqual(stdout.getvalue(), '')
            self.assertIn('the corpus\'s readings aren\'t measured', stderr.getvalue())
            self.assertFalse(os.path.exists(out))

            # The measured readings of the candidate words
            self.write_readings(tts.corpus_dir(settings))
            with captured() as (stdout, _):
                self.assertEqual(contextlabels.main(['aligned', '--out', out]), 0)
        self.assertEqual(stdout.getvalue(), f'aligned: 3 measured clips, 2 candidate tokens with a measured reading ({len(tiny_words())} '
                                            'candidate words)\n')
        with open(os.path.join(out, 'aligned.jsonl'), encoding='utf-8') as file:
            records = [json.loads(line) for line in file]
        self.assertEqual(records, [
            {'id': 'c1', 'book': 'pg1', 'voice': 'male', 'text': 'I had read it.',
             'tokens': [{'position': 2, 'offset': 6, 'word': 'read', 'reading': 1}]},
            {'id': 'c2', 'book': 'pg2', 'voice': 'female', 'text': 'Who lives here?',
             'tokens': [{'position': 1, 'offset': 4, 'word': 'lives', 'reading': 0}]}
        ])

    # The test books (pg1, pg2) and their pins (a pins file, patched in with the recipe's texts)
    def write_books(self):
        books_dir = self.path('books')
        self.write(os.path.join(books_dir, 'pg1.txt'), 'Header\n*** START OF THE BOOK ***\nI had read the record.\n\nNo words here.\n\n'
                                                       f'{"read " * 1100}\n\nThey will read it. The lead pipes.\n'
                                                       '*** END OF THE BOOK ***\nfooter\n')
        self.write(os.path.join(books_dir, 'pg2.txt'), 'Her lives were long. She read and rested.\n')
        pins = {}
        for name in ('pg1', 'pg2'):
            with open(os.path.join(books_dir, f'{name}.txt'), 'rb') as file:
                data = file.read()
            pins[name] = {'id': int(name[2:]), 'url': f'https://www.gutenberg.org/cache/epub/{name[2:]}/{name}.txt',
                          'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'pinned': '2026-10-15', 'archive': None}
        self.write(self.path('books.json'), json.dumps({'description': 'pins', 'texts': pins}))
        return books_dir

    # The recipe's texts and pins patched to the test books' (write_books)
    def patched_books(self, texts=('pg2', 'pg1')):
        return unittest.mock.patch.multiple(contextlabels, PINS_PATH=self.path('books.json'),
                                            SETTINGS={**contextlabels.SETTINGS, 'texts': list(texts)})

    def test_books(self):
        books_dir = self.write_books()
        with self.patched_books(), captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['books', '--books', books_dir]), 0)
        self.assertEqual(stdout.getvalue(), f'books: 2 pinned texts in {books_dir}\n')

        # A book that isn't its pinned text fails, the recipe's books directory by default
        self.write(os.path.join(books_dir, 'pg2.txt'), 'Revised.\n')
        with self.patched_books(), unittest.mock.patch.object(contextlabels, 'BOOKS_DIR', books_dir), captured() as (stdout, stderr):
            self.assertEqual(contextlabels.main(['books']), 1)
        self.assertEqual(stdout.getvalue(), '')
        self.assertTrue(stderr.getvalue().startswith('books: book pg2 (Project Gutenberg #2): the copy '))

    def test_book_text(self):
        books_dir = self.write_books()
        with self.patched_books():
            self.assertEqual(contextlabels.book_names(), ['pg1', 'pg2'])
            pins = load_pins(contextlabels.PINS_PATH)
        text = contextlabels.book_text(books_dir, 'pg1', pins)
        self.assertTrue(text.startswith('\nI had read the record.'))
        self.assertTrue(text.endswith('The lead pipes.\n'))
        self.assertEqual(contextlabels.book_text(books_dir, 'pg2', pins), 'Her lives were long. She read and rested.\n')

    def test_tag(self):
        out = self.path('out')
        self.write(os.path.join(out, 'aligned.jsonl'), json.dumps({'id': 'u1', 'text': 'I had read it.'}) + '\n' +
                   json.dumps({'id': 'u2', 'text': 'Who lives here?'}) + '\n')
        books_dir = self.write_books()
        nlp = FakeNlp()
        spacy = types.ModuleType('spacy')
        spacy.load = lambda name, disable: nlp
        with unittest.mock.patch.dict(sys.modules, {'spacy': spacy}), self.patched_books(), \
                unittest.mock.patch.object(contextlabels.multiprocessing, 'Pool', data_fixtures.FakePool), \
                unittest.mock.patch.object(contextlabels, 'TAG_CHUNK', 1), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()), captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['tag', '--out', out, '--books', books_dir, '-j', '2']), 0)
        self.assertEqual(stdout.getvalue(),
                         'tag: 2 aligned clips\ntag: pg1 2 paragraphs (2 in all)\ntag: pg2 1 paragraphs (3 in all)\n')
        # (the utterances in chunks, each tagged in batches of TAG_BATCH in one process)
        self.assertEqual(nlp.calls, [(['I had read it.'], 64), (['Who lives here?'], 64),
                                     (['I had read the record.', 'They will read it. The lead pipes.'], 64),
                                     (['Her lives were long. She read and rested.'], 64)])
        with open(os.path.join(out, 'aligned-tags.jsonl'), encoding='utf-8') as file:
            self.assertEqual([json.loads(line) for line in file], [{'id': 'u1', 'tags': {'6': 'VBD'}}, {'id': 'u2', 'tags': {'4': 'NNS'}}])
        with open(os.path.join(out, 'books.jsonl'), encoding='utf-8') as file:
            records = [json.loads(line) for line in file]
        self.assertEqual([(record['book'], record['text'], record['tags']) for record in records], [
            ('pg1', 'I had read the record.', {'6': 'VBD', '15': 'NN'}),
            ('pg1', 'They will read it. The lead pipes.', {'10': 'VB', '23': 'NN'}),
            ('pg2', 'Her lives were long. She read and rested.', {'4': 'NNS', '25': 'VB'})
        ])

        # The recipe's books directory by default; a book that isn't its pinned text fails before any tagging
        self.write(os.path.join(books_dir, 'pg1.txt'), 'Revised.\n')
        with unittest.mock.patch.dict(sys.modules, {'spacy': spacy}), self.patched_books(), \
                unittest.mock.patch.object(contextlabels, 'BOOKS_DIR', books_dir), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()), captured() as (stdout, stderr):
            self.assertEqual(contextlabels.main(['tag', '--out', out]), 1)
        self.assertEqual(stdout.getvalue(), '')
        self.assertTrue(stderr.getvalue().startswith('tag: book pg1 (Project Gutenberg #1): the copy '))

    def test_regression_windows(self):
        items = [{'word': 'read', 'text': 'I read what I read.', 'occurrence': 2}, {'word': 'lives', 'text': 'Their lives were long.'}]
        with helpers.json_files(contextlabels, {contextlabels.TEST_PATH: items}):
            windows = contextlabels.regression_windows()
        self.assertEqual(windows, {('read', 'what', 'i', 'read', '.', '', ''), ('', '', 'their', 'lives', 'were', 'long', '.')})
        self.assertTrue(contextlabels.regression_windows())

    def test_text_examples(self):
        examples = contextlabels.text_examples('I read it. Mr. Read.', [(2, 'read', 1), (99, 'read', 0), (0, 'read', 0)])
        self.assertEqual(examples, [{'window': ['', '', 'i', 'read', 'it', '.', 'mister'], 'word': 'read', 'reading': 1}])

    def test_measured_classes(self):
        with tempfile.TemporaryDirectory() as directory, \
                unittest.mock.patch.object(tts, 'load_settings', return_value=ctc_fixtures.tts_settings()), \
                unittest.mock.patch.object(tts, 'TTS_DIR', directory):
            self.assertEqual(contextlabels.measured_classes(), {})
            readings_dir = os.path.join(tts.corpus_dir(ctc_fixtures.tts_settings()), 'readings')
            os.makedirs(readings_dir)
            with open(os.path.join(readings_dir, 'summary.json'), 'w', encoding='utf-8') as file:
                json.dump({'classes': {'strong': True}}, file)
            self.assertEqual(contextlabels.measured_classes(), {'strong': True})

    def test_validate_mappings(self):
        words = tiny_words()
        aligned = [{'id': 'u1', 'text': 'I had read it, record it.', 'tokens': [
            {'offset': 6, 'word': 'read', 'reading': 1}, {'offset': 15, 'word': 'record', 'reading': 1},
            {'offset': 0, 'word': 'read', 'reading': 0}]},
                   {'id': 'u2', 'text': 'I read.', 'tokens': [{'offset': 2, 'word': 'read', 'reading': 0}]}]
        tags = {'u1': {'6': 'VBD', '15': 'VB'}, 'u2': {'2': 'VBD'}}
        by_word, by_family = contextlabels.validate_mappings(words, aligned, tags)
        self.assertEqual(by_word, {'read': {'agree': 1, 'total': 2}, 'record': {'agree': 1, 'total': 1}})
        self.assertEqual(by_family, {'stress': {'agree': 1, 'total': 1}})
        self.assertFalse(contextlabels.mapping_valid('read', by_word))
        self.assertFalse(contextlabels.mapping_valid('lives', by_word))
        self.assertTrue(contextlabels.mapping_valid('read', {'read': {'agree': 19, 'total': 20}}))
        self.assertFalse(contextlabels.mapping_valid('read', {'read': {'agree': 18, 'total': 20}}))

        # A family word of too few aligned tokens - its family's mapping (MIN_FAMILY_VALIDATE tokens at MIN_AGREEMENT); none
        # when MIN_FAMILY_VALIDATE is 0, and a word of enough tokens by its own alone
        families = {'stress': {'agree': 199, 'total': 200}}
        self.assertTrue(contextlabels.mapping_valid('record', by_word, 'stress', families))
        self.assertFalse(contextlabels.mapping_valid('record', by_word, 'stress', {'stress': {'agree': 189, 'total': 200}}))
        self.assertFalse(contextlabels.mapping_valid('record', by_word, 'stress', {'stress': {'agree': 199, 'total': 199}}))
        self.assertFalse(contextlabels.mapping_valid('record', by_word, 'ate', families))
        # A family word the corpus can't measure takes its family's mapping as it is; an own word doesn't
        self.assertTrue(contextlabels.mapping_valid('record', by_word, 'stress', {}, measurable=False))
        self.assertFalse(contextlabels.mapping_valid('record', by_word, 'own', {}, measurable=False))
        self.assertFalse(contextlabels.mapping_valid('read', {'read': {'agree': 18, 'total': 20}}, 'stress', {}, measurable=False))
        self.assertFalse(contextlabels.mapping_valid('record', {'record': {'agree': 0, 'total': 20}}, 'stress', families))
        with unittest.mock.patch.object(contextlabels, 'MIN_FAMILY_VALIDATE', 0):
            self.assertFalse(contextlabels.mapping_valid('record', by_word, 'stress', families))

    def test_default_readings(self):
        words = tiny_words()
        first = {'read': ['R', 'IY1', 'D'], 'lives': ['L', 'AY1', 'V', 'Z'], 'lead': ['L', 'EY1', 'D']}
        calls = []

        def load_data(relabel):
            calls.append(relabel)
            return {'first': first}
        config = {'stages': [{'relabel': 'train/data/g2p-relabel.json'}]}
        with unittest.mock.patch.object(g2p, 'load_data', load_data), unittest.mock.patch.object(g2p, 'load_config', lambda: config):
            defaults = contextlabels.default_readings(words)
            config['stages'][-1].pop('relabel')
            contextlabels.default_readings({})
        self.assertEqual(defaults['read'], 0)
        self.assertEqual(defaults['lives'], 1)
        self.assertIsNone(defaults['lead'])
        self.assertEqual(defaults['bass'], 0)
        self.assertTrue(calls[0].endswith(os.path.join('train', 'data', 'g2p-relabel.json')))
        self.assertIsNone(calls[1])

    def test_labels_unmeasured(self):
        # Words whose readings the corpus can't measure: a family word (only its family's rule maps it) needs
        # MIN_UNMEASURED_ALTERNATE examples of its less common readings; a seeded reading always counts as common
        out = self.path('out')
        self.write(os.path.join(out, 'aligned.jsonl'), '')
        self.write(os.path.join(out, 'aligned-tags.jsonl'), '')
        stress = {'family': 'stress', 'curated': False, 'tags': {}}
        words = {'record': {**stress, 'readings': [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']]},
                 'conduct': {**stress, 'readings': [['K AA1 N D AH0 K T'], ['K AA0 N D AH1 K T']]},
                 'object': {**stress, 'readings': [['AA1 B JH EH0 K T'], ['AH0 B JH EH1 K T']]}}
        books = []
        for word, nouns, verbs in (('record', 3, 3), ('conduct', 3, 2), ('object', 4, 0)):
            books.extend({'book': 'pg1', 'text': f'The {word}.', 'tags': {'4': 'NN'}} for _ in range(nouns))
            books.extend({'book': 'pg1', 'text': f'They {word} it.', 'tags': {'5': 'VB'}} for _ in range(verbs))
        self.write(os.path.join(out, 'books.jsonl'), ''.join(json.dumps(record) + '\n' for record in books))
        spec = {'exclude': [], 'collocations': {}, 'seeds': [{'word': 'object', 'phones': 'AH0 B JH EH1 K T', 'texts': ['I object.']}]}
        with unittest.mock.patch.object(contextlabels, 'load_spec', lambda: spec), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: words), \
                unittest.mock.patch.object(contextlabels, 'default_readings', lambda words: {word: 0 for word in words}), \
                unittest.mock.patch.object(contextlabels, 'measured_classes', lambda: {}), \
                unittest.mock.patch.object(contextlabels, 'regression_windows', set), \
                unittest.mock.patch.object(contextlabels, 'MIN_FAMILY_VALIDATE', 0), \
                unittest.mock.patch.object(contextlabels, 'MIN_ALTERNATE', 2), \
                unittest.mock.patch.object(contextlabels, 'MIN_ALTERNATE_SHARE', 0), \
                unittest.mock.patch.object(contextlabels, 'MIN_UNMEASURED_ALTERNATE', 3), \
                unittest.mock.patch.object(contextlabels, 'MAX_SILVER', 10), \
                captured():
            self.assertEqual(contextlabels.main(['labels', '--out', out]), 0)
        with open(os.path.join(out, 'labels.json'), encoding='utf-8') as file:
            labels = json.load(file)
        # (record - three verbs, kept; conduct - two, common but too few; object - one seeded verb, fewer than MIN_ALTERNATE)
        self.assertEqual(sorted(labels['words']), ['object', 'record'])
        self.assertEqual(labels['words']['record']['counts'], [3, 3])
        self.assertEqual(labels['words']['object']['counts'], [4, 1])

    def test_labels(self):
        out = self.path('out')
        aligned = []
        tags = []
        for number in range(12):
            reading = number % 2
            text = 'I had read it.' if reading else 'I will read it.'
            aligned.append({'id': f'a{number}', 'book': 'pg2' if number == 0 else 'pg1', 'voice': 'male', 'text': text,
                            'tokens': [{'offset': 6 if reading else 7, 'word': 'read', 'reading': reading}]})
            tags.append({'id': f'a{number}', 'tags': {'6': 'VBD'} if reading else {'7': 'VB'}})
        aligned.append({'id': 'b', 'book': 'pg1', 'voice': 'female', 'text': 'The lives.',
                        'tokens': [{'offset': 4, 'word': 'lives', 'reading': 1}, {'offset': 0, 'word': 'nodefault', 'reading': 0}]})
        tags.append({'id': 'b', 'tags': {}})
        # (a family word of too few tokens, validated by its family's mapping)
        aligned.append({'id': 'r', 'book': 'pg1', 'voice': 'female', 'text': 'They record it.',
                        'tokens': [{'offset': 5, 'word': 'record', 'reading': 1}]})
        tags.append({'id': 'r', 'tags': {'5': 'VB'}})
        # (a family word without a validated mapping - its family's has no tokens - not an aligned example)
        words = {**tiny_words(), 'estimate': {'readings': [['EH1 S T AH0 M AH0 T'], ['EH1 S T AH0 M EY2 T']], 'family': 'ate',
                                              'curated': False, 'tags': {}}}
        # (read - an inventory word here, its aligned tokens examples; lead - curated, its aligned token not an example)
        words['read'] = {**words['read'], 'curated': False}
        # (lead's noun tag maps to L IY1 D - "the lead pipes" is tagged a noun, but its collocation decides it: no silver)
        words['lead'] = {**words['lead'], 'tags': {'NN': 0}}
        aligned.append({'id': 'l', 'book': 'pg1', 'voice': 'male', 'text': 'They lead us.',
                        'tokens': [{'offset': 5, 'word': 'lead', 'reading': 0}]})
        tags.append({'id': 'l', 'tags': {}})
        aligned.append({'id': 'e', 'book': 'pg1', 'voice': 'male', 'text': 'An estimate.',
                        'tokens': [{'offset': 3, 'word': 'estimate', 'reading': 0}]})
        tags.append({'id': 'e', 'tags': {}})
        # (tokens a collocation decides - not aligned examples: one the corpus reads as the collocation does, one not)
        aligned.append({'id': 'c1', 'book': 'pg1', 'voice': 'male', 'text': 'The lead pipes.',
                        'tokens': [{'offset': 4, 'word': 'lead', 'reading': 1}]})
        aligned.append({'id': 'c2', 'book': 'pg1', 'voice': 'female', 'text': 'A lead pipe burst.',
                        'tokens': [{'offset': 2, 'word': 'lead', 'reading': 0}]})
        tags.extend([{'id': 'c1', 'tags': {}}, {'id': 'c2', 'tags': {}}])
        self.write(os.path.join(out, 'aligned.jsonl'), ''.join(json.dumps(record) + '\n' for record in aligned))
        self.write(os.path.join(out, 'aligned-tags.jsonl'), ''.join(json.dumps(record) + '\n' for record in tags))
        books = [
            {'book': 'pg1', 'text': 'He had read the lead pipes, the bass.', 'tags': {'7': 'VBD', '16': 'NN', '32': 'NN'}},
            {'book': 'pg2', 'text': 'He had read it. A lead balloon.', 'tags': {'7': 'VBD'}},
            {'book': 'pg1', 'text': 'They read.', 'tags': {}},
            {'book': 'pg1', 'text': 'We read. They lead the way. The bass.', 'tags': {}},
            {'book': 'pg1', 'text': 'They lead the way.', 'tags': {}},
            {'book': 'pg2', 'text': 'You read.', 'tags': {}}
        ]
        self.write(os.path.join(out, 'books.jsonl'), ''.join(json.dumps(record) + '\n' for record in books))
        spec = {'exclude': ['excluded'], 'collocations': {'lead': [['\\blead pipes?\\b', 'L EH1 D'], ['\\bwhite lead\\b', 'L EH1 D']]},
                'seeds': [{'word': 'lead', 'phones': 'L IY1 D', 'texts': ['They lead the way.', 'Lead on.']},
                          {'word': 'bass', 'phones': 'B EY1 S', 'texts': ['The bass singer.']}]}
        defaults = {word: 0 for word in words}
        defaults['nodefault'] = None
        held_out = {('', '', 'they', 'lead', 'the', 'way', '.')}
        with unittest.mock.patch.object(contextlabels, 'load_spec', lambda: spec), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: words), \
                unittest.mock.patch.object(contextlabels, 'default_readings', lambda words: defaults), \
                unittest.mock.patch.object(contextlabels, 'measured_classes', lambda: {'strong': True, 'weak': True}), \
                unittest.mock.patch.object(contextlabels, 'regression_windows', lambda: held_out), \
                unittest.mock.patch.object(contextlabels, 'MIN_VALIDATE', 3), \
                unittest.mock.patch.object(contextlabels, 'MIN_FAMILY_VALIDATE', 1), \
                unittest.mock.patch.object(contextlabels, 'MIN_ALTERNATE', 1), \
                unittest.mock.patch.object(contextlabels, 'MAX_SILVER', 1), \
                unittest.mock.patch.object(contextlabels, 'MAX_TRANSFER', 2), \
                unittest.mock.patch.object(contextlabels, 'HELD_OUT_BOOKS', {'pg2'}), \
                captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['labels', '--out', out]), 0)
        with open(os.path.join(out, 'labels.json'), encoding='utf-8') as file:
            labels = json.load(file)
        self.assertEqual(sorted(labels['words']), ['lead', 'read'])
        self.assertEqual(labels['words']['read'], {'readings': [['R IY1 D'], ['R EH1 D']], 'order': [0, 1], 'family': 'own',
                                                   'counts': [6, 7]})
        self.assertEqual(labels['words']['lead']['counts'], [1, 1])
        # (the curated words' mappings are hand-written - valid; estimate - its readings measured, too few - isn't)
        self.assertEqual(labels['valid'], ['bass', 'excluded', 'lead', 'lives', 'nodefault', 'read', 'record'])
        self.assertEqual(labels['validation'], {'words': {'read': {'agree': 12, 'total': 12}, 'record': {'agree': 1, 'total': 1}},
                                                'families': {'stress': {'agree': 1, 'total': 1}}})
        sources = sorted((example['word'], example['source'], example['split'], example['slot']) for example in labels['examples'])
        # (the aligned examples' split by their book - pg2 held out)
        self.assertEqual(sources.count(('read', 'aligned', 'heldout', 0)), 1)
        self.assertEqual(sources.count(('read', 'aligned', 'train', 1)), 6)
        self.assertEqual(sources.count(('read', 'silver', 'train', 1)), 1)
        self.assertIn(('lead', 'collocation', 'train', 1), sources)
        self.assertIn(('lead', 'seed', 'train', 0), sources)
        self.assertNotIn(('lead', 'collocation', 'heldout', 1), sources)
        self.assertFalse(any(word in ('lead', 'estimate') and source == 'aligned' for word, source, _, _ in sources))
        self.assertEqual({example['group'] for example in labels['examples'] if example['source'] == 'aligned'}, {'pg1', 'pg2'})
        self.assertEqual(sum(1 for example in labels['examples'] if example['word'] == 'lead' and example['source'] == 'seed'), 1)
        # The transfer windows - the ambiguous words' book windows that aren't examples or the regression set's (they lead
        # the way), not the held-out books', MAX_TRANSFER a word
        self.assertEqual(labels['transfer'], [['read', '.', 'they', 'lead', 'the', 'way', '.'], ['', '', 'we', 'read', '.', 'they', 'lead'],
                                              ['', '', 'they', 'read', '.', '', '']])
        lines = stdout.getvalue().split('\n')
        self.assertTrue(lines[0].startswith('labels: 2 ambiguous words (8 candidates, 7 with a validated mapping), '))
        self.assertTrue(lines[0].endswith('; 3 transfer windows'))
        self.assertEqual(lines[1], 'collocations: the corpus\'s readings disagree on 1 of 2 of their tokens')
        self.assertEqual(lines[2], 'families (aligned agreement): stress 1/1')

    def test_main_module(self):
        with unittest.mock.patch.dict(sys.modules), unittest.mock.patch.object(sys, 'argv', ['contextlabels', '--help']), \
                captured() as (stdout, _):
            sys.modules.pop('picovox_train.contextlabels')
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.contextlabels', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('The heteronym labels', stdout.getvalue())
