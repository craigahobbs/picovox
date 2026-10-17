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

from picovox_train import contextlabels, distill, g2p

from .helpers import captured


class FakePool:
    """multiprocessing.Pool in the test's process (so the workers' code is measured)"""

    def __init__(self, processes, initializer, initargs):
        self.processes = processes
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap(self, function, items, chunksize=1):  # pylint: disable=unused-argument
        return map(function, items)


# A fake spaCy (build/venv-pos only): the tagger tags "read" VBD after "had" or "I", else VB, "lives" NNS, other words NN
class FakeToken:
    def __init__(self, idx, tag):
        self.idx = idx
        self.tag_ = tag


class FakeNlp:
    def __init__(self):
        self.calls = []

    def pipe(self, texts, batch_size, n_process):
        self.calls.append((batch_size, n_process))
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


# Speakers held out (by the hash) and not
def speaker(held_out):
    return next(f'{number}' for number in range(1000) if contextlabels.held_out_speaker(f'{number}') == held_out)


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
        contextlabels.set_words(None)

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
        self.assertEqual(contextlabels.variant_vowels('R IH0 K AO1 R D'), ['IH0', 'AO1'])
        self.assertEqual(contextlabels.primary_vowel('R IH0 K AO1 R D'), 1)
        self.assertIsNone(contextlabels.primary_vowel('S T'))

    def test_word_family(self):
        self.assertEqual(contextlabels.word_family('record', [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']]), 'stress')
        self.assertEqual(contextlabels.word_family('estimate', [['EH1 S T AH0 M AH0 T'], ['EH1 S T AH0 M EY2 T']]), 'ate')
        self.assertEqual(contextlabels.word_family('estimates', [['EH1 S T AH0 M AH0 T S'], ['EH1 S T AH0 M EY2 T S']]), 'ate')
        self.assertEqual(contextlabels.word_family('use', [['Y UW1 S'], ['Y UW1 Z']]), 'consonant')
        self.assertEqual(contextlabels.word_family('lead', [['L IY1 D'], ['L EH1 D']]), 'vowel')
        # (-ate with both readings' full -ate vowels is no -ate pair)
        self.assertEqual(contextlabels.word_family('abate', [['AH0 B EY2 T'], ['IH0 B EY2 T']]), 'vowel')

    def test_family_reading(self):
        stress = [['R EH1 K ER0 D'], ['R IH0 K AO1 R D']]
        self.assertEqual(contextlabels.family_reading('stress', 'NN', stress), 0)
        self.assertEqual(contextlabels.family_reading('stress', 'VBD', stress), 1)
        self.assertIsNone(contextlabels.family_reading('stress', 'JJ', stress))
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
        self.assertTrue(contextlabels.voicing_pair([['Y UW1 S'], ['Y UW1 Z']]))
        self.assertFalse(contextlabels.voicing_pair([['S AH0 L EH1 K T'], ['Z AH0 L EH1 K T']]))
        self.assertFalse(contextlabels.voicing_pair([['K L OW1 S'], ['K L OW1 Z IH0']]))
        self.assertFalse(contextlabels.voicing_pair([['K L OW1 S'], ['K L OW1 T']]))

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

    def write_corpus(self):
        labels = self.path('lab', 'phone')
        audio = self.path('LibriTTS_R')
        utterances = [
            (speaker(False), 'I had read it.', [('I', [('AY1', 0.0, 0.1)]),
                                                ('had', [('HH', 0.1, 0.15), ('AE1', 0.15, 0.2), ('D', 0.2, 0.25)]),
                                                ('read', [('R', 0.25, 0.3), ('EH1', 0.3, 0.4), ('D', 0.4, 0.45)]),
                                                ('it', [('IH1', 0.45, 0.5), ('T', 0.5, 0.55)])]),
            (speaker(True), 'Who lives here?', [('who', [('HH', 0.0, 0.05), ('UW1', 0.05, 0.1)]),
                                                 ('lives', [('L', 0.1, 0.15), ('IH1', 0.15, 0.2), ('V', 0.2, 0.25), ('Z', 0.25, 0.3)]),
                                                 ('here', [('HH', 0.3, 0.35), ('IY1', 0.35, 0.4), ('R', 0.4, 0.45)])]),
            (speaker(False), 'Nothing to see.', [('nothing', [('N', 0.0, 0.1)]), ('to', [('T', 0.1, 0.2)]), ('see', [('S', 0.2, 0.3)])])
        ]
        for number, (speaker_id, text, words) in enumerate(utterances):
            utterance = f'{speaker_id}_1_{number:06d}'
            parts = ('train-clean-100', speaker_id, '1')
            word_lines = [f'{phones[0][1]}\t{phones[-1][2]}\t{word}' for word, phones in words] + ['0.9\t1.0\t']
            phone_lines = [f'{start}\t{end}\t{phone}' for _, phones in words for phone, start, end in phones] + ['0.9\t1.0\tsil']
            self.write(os.path.join(labels, *parts, utterance + '.lab'), '\n'.join(phone_lines) + '\n')
            self.write(self.path('lab', 'word', *parts, utterance + '.lab'), '\n'.join(word_lines) + '\n')
            self.write(os.path.join(audio, *parts, utterance + '.normalized.txt'), text + '\n')
        # (a label without its text, and another file)
        self.write(os.path.join(labels, *parts, 'missing.lab'), '0\t1\tsil\n')
        self.write(os.path.join(labels, *parts, 'notes.txt'), 'x\n')
        return labels, audio

    def test_aligned(self):
        labels, audio = self.write_corpus()
        out = self.path('out')
        with unittest.mock.patch.object(contextlabels, 'LABEL_DIR', labels), \
                unittest.mock.patch.object(contextlabels, 'LIBRITTS_DIR', audio), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()), \
                unittest.mock.patch.object(contextlabels.multiprocessing, 'Pool', FakePool), captured() as (stdout, _):
            self.assertEqual(len(contextlabels.libritts_utterances()), 3)
            self.assertEqual(contextlabels.main(['aligned', '--out', out, '-j', '1']), 0)
        self.assertEqual(stdout.getvalue(), f'aligned: 3 utterances, 2 candidate tokens with an aligned reading ({len(tiny_words())} '
                                            'candidate words)\n')
        with open(os.path.join(out, 'aligned.jsonl'), encoding='utf-8') as file:
            records = [json.loads(line) for line in file]
        self.assertEqual([record['tokens'] for record in records], [
            [{'position': 2, 'offset': 6, 'word': 'read', 'reading': 1}],
            [{'position': 1, 'offset': 4, 'word': 'lives', 'reading': 0}]
        ])
        self.assertEqual(records[0]['speaker'], speaker(False))
        self.assertEqual(records[0]['text'], 'I had read it.')

    def test_books(self):
        with unittest.mock.patch.object(distill, 'book_paths', lambda settings: [settings['books']] * len(settings['bookIds'])), \
                captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['books', '--books', self.directory]), 0)
            self.assertEqual(contextlabels.main(['books']), 0)
        count = len(contextlabels.SETTINGS['bookIds'])
        self.assertEqual(stdout.getvalue(), f'books: {count} in {self.directory}\nbooks: {count} in {contextlabels.BOOKS_DIR}\n')

    def write_books(self):
        books = self.path('books')
        self.write(os.path.join(books, 'pg1.txt'), 'Header\n*** START OF THE BOOK ***\nI had read the record.\n\nNo words here.\n\n'
                                                   f'{"read " * 1100}\n\nThey will read it. The lead pipes.\n'
                                                   '*** END OF THE BOOK ***\nfooter\n')
        self.write(os.path.join(books, 'pg2.txt'), 'Her lives were long. She read and rested.\n')
        return books

    def test_book_texts(self):
        books = self.write_books()
        with unittest.mock.patch.dict(contextlabels.SETTINGS, {'bookIds': [2, 1]}):
            texts = contextlabels.book_texts(books)
        self.assertEqual([book for book, _ in texts], ['pg1', 'pg2'])
        self.assertTrue(texts[0][1].startswith('\nI had read the record.'))
        self.assertTrue(texts[0][1].endswith('The lead pipes.\n'))
        self.assertEqual(texts[1][1], 'Her lives were long. She read and rested.\n')

    def test_tag(self):
        out = self.path('out')
        self.write(os.path.join(out, 'aligned.jsonl'), json.dumps({'id': 'u1', 'text': 'I had read it.'}) + '\n' +
                   json.dumps({'id': 'u2', 'text': 'Who lives here?'}) + '\n')
        books = self.write_books()
        nlp = FakeNlp()
        spacy = types.ModuleType('spacy')
        spacy.load = lambda name, disable: nlp
        with unittest.mock.patch.dict(sys.modules, {'spacy': spacy}), \
                unittest.mock.patch.dict(contextlabels.SETTINGS, {'bookIds': [1, 2]}), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()), captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['tag', '--out', out, '--books', books, '-j', '2']), 0)
        self.assertEqual(stdout.getvalue(),
                         'tag: 2 aligned utterances\ntag: pg1 2 paragraphs (2 in all)\ntag: pg2 1 paragraphs (3 in all)\n')
        self.assertEqual(nlp.calls[0], (64, 2))
        with open(os.path.join(out, 'aligned-tags.jsonl'), encoding='utf-8') as file:
            self.assertEqual([json.loads(line) for line in file], [{'id': 'u1', 'tags': {'6': 'VBD'}}, {'id': 'u2', 'tags': {'4': 'NNS'}}])
        with open(os.path.join(out, 'books.jsonl'), encoding='utf-8') as file:
            records = [json.loads(line) for line in file]
        self.assertEqual([(record['book'], record['text'], record['tags']) for record in records], [
            ('pg1', 'I had read the record.', {'6': 'VBD', '15': 'NN'}),
            ('pg1', 'They will read it. The lead pipes.', {'10': 'VB', '23': 'NN'}),
            ('pg2', 'Her lives were long. She read and rested.', {'4': 'NNS', '25': 'VB'})
        ])

        # The recipe's books directory by default
        with unittest.mock.patch.dict(sys.modules, {'spacy': spacy}), unittest.mock.patch.dict(contextlabels.SETTINGS, {'bookIds': []}), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()), \
                unittest.mock.patch.object(contextlabels, 'book_texts', lambda books_dir: [('pg9', books_dir)]), captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['tag', '--out', out]), 0)
        self.assertIn('tag: pg9 0 paragraphs', stdout.getvalue())

    def test_regression_windows(self):
        path = self.path('test.json')
        self.write(path, json.dumps([{'word': 'read', 'text': 'I read what I read.', 'occurrence': 2},
                                     {'word': 'lives', 'text': 'Their lives were long.'}]))
        with unittest.mock.patch.object(contextlabels, 'TEST_PATH', path):
            windows = contextlabels.regression_windows()
        self.assertEqual(windows, {('read', 'what', 'i', 'read', '.', '', ''), ('', '', 'their', 'lives', 'were', 'long', '.')})
        self.assertTrue(contextlabels.regression_windows())

    def test_held_out_speaker(self):
        number = int(hashlib.sha1(b'19').hexdigest(), 16) % 100
        self.assertEqual(contextlabels.held_out_speaker('19'), number < contextlabels.HELD_OUT_PERCENT)

    def test_text_examples(self):
        examples = contextlabels.text_examples('I read it. Mr. Read.', [(2, 'read', 1), (99, 'read', 0), (0, 'read', 0)])
        self.assertEqual(examples, [{'window': ['', '', 'i', 'read', 'it', '.', 'mister'], 'word': 'read', 'reading': 1}])

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

    def test_default_readings(self):
        words = tiny_words()
        first = {'read': ['R', 'IY1', 'D'], 'lives': ['L', 'AY1', 'V', 'Z'], 'lead': ['L', 'EY1', 'D']}
        calls = []

        def load_data(relabel):
            calls.append(relabel)
            return {'first': first}
        config = {'stages': [{'relabel': 'data/nn/g2p-relabel.json'}]}
        with unittest.mock.patch.object(g2p, 'load_data', load_data), unittest.mock.patch.object(g2p, 'load_config', lambda: config):
            defaults = contextlabels.default_readings(words)
            config['stages'][-1].pop('relabel')
            contextlabels.default_readings({})
        self.assertEqual(defaults['read'], 0)
        self.assertEqual(defaults['lives'], 1)
        self.assertIsNone(defaults['lead'])
        self.assertEqual(defaults['bass'], 0)
        self.assertTrue(calls[0].endswith(os.path.join('data', 'nn', 'g2p-relabel.json')))
        self.assertIsNone(calls[1])

    def test_labels(self):
        out = self.path('out')
        trained, held = speaker(False), speaker(True)
        aligned = []
        tags = []
        for number in range(12):
            reading = number % 2
            text = 'I had read it.' if reading else 'I will read it.'
            aligned.append({'id': f'a{number}', 'speaker': held if number == 0 else trained, 'text': text,
                            'tokens': [{'offset': 6 if reading else 7, 'word': 'read', 'reading': reading}]})
            tags.append({'id': f'a{number}', 'tags': {'6': 'VBD'} if reading else {'7': 'VB'}})
        aligned.append({'id': 'b', 'speaker': trained, 'text': 'The lives.', 'tokens': [{'offset': 4, 'word': 'lives', 'reading': 1},
                                                                                       {'offset': 0, 'word': 'nodefault', 'reading': 0}]})
        tags.append({'id': 'b', 'tags': {}})
        self.write(os.path.join(out, 'aligned.jsonl'), ''.join(json.dumps(record) + '\n' for record in aligned))
        self.write(os.path.join(out, 'aligned-tags.jsonl'), ''.join(json.dumps(record) + '\n' for record in tags))
        books = [
            {'book': 'pg1', 'text': 'He had read the lead pipes, the bass.', 'tags': {'7': 'VBD', '16': 'NN', '32': 'NN'}},
            {'book': 'pg2', 'text': 'He had read it. A lead balloon.', 'tags': {'7': 'VBD'}},
            {'book': 'pg1', 'text': 'They read.', 'tags': {}}
        ]
        self.write(os.path.join(out, 'books.jsonl'), ''.join(json.dumps(record) + '\n' for record in books))
        spec = {'exclude': ['excluded'], 'collocations': {'lead': [['\\blead pipes?\\b', 'L EH1 D'], ['\\bwhite lead\\b', 'L EH1 D']]},
                'seeds': [{'word': 'lead', 'phones': 'L IY1 D', 'texts': ['They lead the way.', 'Lead on.']},
                          {'word': 'bass', 'phones': 'B EY1 S', 'texts': ['The bass singer.']}]}
        defaults = {word: 0 for word in tiny_words()}
        defaults['nodefault'] = None
        held_out = {('', '', 'they', 'lead', 'the', 'way', '.')}
        with unittest.mock.patch.object(contextlabels, 'load_spec', lambda: spec), \
                unittest.mock.patch.object(contextlabels, 'candidate_words', lambda spec: tiny_words()), \
                unittest.mock.patch.object(contextlabels, 'default_readings', lambda words: defaults), \
                unittest.mock.patch.object(contextlabels, 'regression_windows', lambda: held_out), \
                unittest.mock.patch.object(contextlabels, 'MIN_VALIDATE', 3), \
                unittest.mock.patch.object(contextlabels, 'MIN_ALTERNATE', 1), \
                unittest.mock.patch.object(contextlabels, 'MAX_SILVER', 1), \
                unittest.mock.patch.object(contextlabels, 'HELD_OUT_BOOKS', {'pg2'}), \
                captured() as (stdout, _):
            self.assertEqual(contextlabels.main(['labels', '--out', out]), 0)
        with open(os.path.join(out, 'labels.json'), encoding='utf-8') as file:
            labels = json.load(file)
        self.assertEqual(sorted(labels['words']), ['lead', 'read'])
        self.assertEqual(labels['words']['read'], {'readings': [['R IY1 D'], ['R EH1 D']], 'order': [0, 1], 'family': 'own',
                                                   'counts': [6, 7]})
        self.assertEqual(labels['words']['lead']['counts'], [1, 1])
        self.assertEqual(labels['valid'], ['read'])
        self.assertEqual(labels['validation']['words'], {'read': {'agree': 12, 'total': 12}})
        sources = sorted((example['word'], example['source'], example['split'], example['slot']) for example in labels['examples'])
        self.assertEqual(sources.count(('read', 'aligned', 'heldout', 0)), 1)
        self.assertEqual(sources.count(('read', 'aligned', 'train', 1)), 6)
        self.assertEqual(sources.count(('read', 'silver', 'train', 1)), 1)
        self.assertIn(('lead', 'collocation', 'train', 1), sources)
        self.assertIn(('lead', 'seed', 'train', 0), sources)
        self.assertNotIn(('lead', 'collocation', 'heldout', 1), sources)
        self.assertEqual(sum(1 for example in labels['examples'] if example['word'] == 'lead' and example['source'] == 'seed'), 1)
        lines = stdout.getvalue().split('\n')
        self.assertTrue(lines[0].startswith('labels: 2 ambiguous words (7 candidates, 1 with a validated mapping), '))
        self.assertEqual(lines[1], 'families (aligned agreement): ')

    def test_main_module(self):
        with unittest.mock.patch.dict(sys.modules), unittest.mock.patch.object(sys, 'argv', ['contextlabels', '--help']), \
                captured() as (stdout, _):
            sys.modules.pop('picovox_train.contextlabels')
            with self.assertRaises(SystemExit) as raised:
                runpy.run_module('picovox_train.contextlabels', run_name='__main__')
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('The heteronym labels', stdout.getvalue())
