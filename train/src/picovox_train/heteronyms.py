# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The G2P's relabeled words (train/data/g2p-relabel.json, g2p.py's focus stage) - dictionary words whose first
pronunciation isn't what the training speech's voices say, from the readings measured in the corpus (measure.py -
the CTC vote or the formants, the classes their controls pass) (python is PYTHONPATH=train/src build/venv-train/bin/python -B):

    python -m picovox_train.heteronyms --out DIR [--keep train/data/g2p-relabel.json] [--check train/data/g2p-relabel.json]
    python -m picovox_train.heteronyms --out DIR --keep train/data/g2p-relabel.json --write train/data/g2p-relabel.json

1. votes - for every measured word, its tokens' readings by reading, and the books and voices of each (votes.json)
2. candidates - words whose majority reading isn't the first, significant: the Wilson 95% lower bound of majority /
   (majority + first) above one half, the majority from MIN_BOOKS books or more (the voices share one front end, so a
   book - its own usage and names - is the independent unit), with the kind of difference (candidates.json)
3. select - the relabeled words: significant candidates, not function words, not the context labels' candidate words
   (the context network decides their readings), not names (words the books capitalize mid-sentence more often than not
   - books.book_capitals; a name's reading is the engine's own G2P's guess), whose difference is lexical (a stressed
   vowel's quality, or the primary stress moving with a quality change - not reduction or stress alone), with the
   majority's share at least --share (relabel.json); with --keep, a committed word the corpus can't judge (its readings
   unmeasured, its class's method unused, no significant majority either way, or a context word) is kept with its
   evidence, and one whose first reading is a significant majority is dropped (contradicted)
"""

import argparse
import datetime
import json
import math
import os
import sys

from . import read_json
from . import books, readings
from .readings import FUNCTION_WORDS, VOWELS, base, edit_ops, kind


# The least books of a significant majority
MIN_BOOKS = 3

#
# 1. The votes
#

# The corpus's measured readings (measure.py's) - its readings.jsonl records and its words.json
def read_measured(directory):
    with open(os.path.join(directory, 'readings', 'readings.jsonl'), encoding='utf-8') as file:
        records = [json.loads(line) for line in file]
    return records, read_json(os.path.join(directory, 'readings', 'words.json'))


def votes(out, directory):
    """votes.json - per measured word: its readings' first variants, class, and per reading the votes, books, and voices"""
    records, words = read_measured(directory)
    counts = {}
    for record in records:
        for token in record['tokens']:
            entry = counts.setdefault(token['word'], {'variants': [reading[0] for reading in words['words'][token['word']]],
                                                      'class': token['class'], 'tokens': 0, 'abstained': 0})
            entry.setdefault('votes', [0] * len(entry['variants']))
            entry.setdefault('books', [[] for _ in entry['variants']])
            entry.setdefault('voices', [[] for _ in entry['variants']])
            entry['tokens'] += 1
            if token['reading'] is None:
                entry['abstained'] += 1
                continue
            entry['votes'][token['reading']] += 1
            for name, value in (('books', record['book']), ('voices', record['key'])):
                if value not in entry[name][token['reading']]:
                    entry[name][token['reading']].append(value)
    with open(os.path.join(out, 'votes.json'), 'w', encoding='utf-8') as file:
        json.dump({'corpus': os.path.basename(directory), 'words': counts}, file)
    return f'{len(counts)} measured words, {sum(entry["tokens"] for entry in counts.values())} tokens'


#
# 2. The candidates
#

# The Wilson score interval's lower bound
def wilson_lower(successes, total, z=1.96):
    if total == 0:
        return 0.0
    share = successes / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    return (centre - spread) / (1 + z * z / total)


# A word's significant majority against its first reading - the majority's index, or 0 when the first is a significant
# majority against the best other, or None
def significant(entry):
    best = max(range(1, len(entry['votes'])), key=lambda index: (entry['votes'][index], -index))
    majority, first = entry['votes'][best], entry['votes'][0]
    if wilson_lower(majority, majority + first) > 0.5 and len(entry['books'][best]) >= MIN_BOOKS:
        return best
    if wilson_lower(first, majority + first) > 0.5 and len(entry['books'][0]) >= MIN_BOOKS:
        return 0
    return None


def candidates(out):
    """candidates.json - every word with a non-first majority"""
    with open(os.path.join(out, 'votes.json'), encoding='utf-8') as file:
        counts = json.load(file)['words']
    items = []
    for word, entry in counts.items():
        best = max(range(len(entry['votes'])), key=lambda index, entry=entry: (entry['votes'][index], -index))
        if best == 0:
            continue
        lower = wilson_lower(entry['votes'][best], entry['votes'][best] + entry['votes'][0])
        items.append({'word': word, 'first': entry['variants'][0], 'majority': entry['variants'][best], 'index': best,
                      'class': entry['class'], 'votes': entry['votes'], 'books': [len(books) for books in entry['books']],
                      'voices': [len(voices) for voices in entry['voices']], 'lower': lower,
                      'significant': significant(entry) == best,
                      'kind': kind(entry['variants'][0].split(), entry['variants'][best].split())})
    items.sort(key=lambda item: -sum(item['votes']))
    with open(os.path.join(out, 'candidates.json'), 'w', encoding='utf-8') as file:
        json.dump(items, file, indent=1)
    return f'{len(items)} words with a non-first majority, {sum(1 for item in items if item["significant"])} significant'


#
# 3. The selection
#

# Whether two variants' difference is more than vowel reduction - with the vowels paired by syllable (or by the edit
# alignment when the syllable counts differ): some pair changes quality with both stressed (lead, wind), or the primary
# stress moves while a vowel's quality changes too (content, record)
def lexical(first, majority):
    first_vowels = [phone for phone in first.split() if base(phone) in VOWELS]
    majority_vowels = [phone for phone in majority.split() if base(phone) in VOWELS]
    if len(first_vowels) == len(majority_vowels):
        pairs = [(x, y) for x, y in zip(first_vowels, majority_vowels) if x != y]
    else:
        pairs = [(x, y) for x, y in edit_ops(first.split(), majority.split()) if x and y and base(x) in VOWELS and base(y) in VOWELS]
    if any(x[-1] in '12' and y[-1] in '12' and base(x) != base(y) for x, y in pairs):
        return True
    primary_moves = any(x[-1] == '1' and y[-1] != '1' for x, y in pairs) and any(x[-1] != '1' and y[-1] == '1' for x, y in pairs)
    return primary_moves and any(base(x) != base(y) for x, y in pairs)


def select(out, share, keep=None, context=(), names=()):
    """relabel.json - {relabel: {word: phones}, chosen, rejected, kept, contradicted}; context, the context labels'
    candidate words - not relabeled (the context network decides their readings), a committed one kept; names, the
    words the books capitalize mid-sentence more often than not - not relabeled"""
    with open(os.path.join(out, 'candidates.json'), encoding='utf-8') as file:
        items = json.load(file)
    with open(os.path.join(out, 'votes.json'), encoding='utf-8') as file:
        counts = json.load(file)['words']
    chosen, rejected = [], []
    for item in items:
        if not item['significant']:
            continue
        majority, first = item['votes'][item['index']], item['votes'][0]
        reasons = []
        if item['word'] in FUNCTION_WORDS:
            reasons.append('function word')
        if item['word'] in context:
            reasons.append('a context word')
        if item['word'] in names:
            reasons.append('a name')
        if item['kind'] not in ('vowel', 'consonant'):
            reasons.append(f'kind {item["kind"]}')
        elif item['kind'] == 'vowel' and not lexical(item['first'], item['majority']):
            reasons.append('reduction only')
        if majority / (majority + first) < share:
            reasons.append(f'share {majority / (majority + first):.2f}')
        (rejected if reasons else chosen).append(dict(item, reasons=reasons))

    # The committed words - kept when the corpus can't judge them, dropped when it contradicts them
    kept, contradicted = [], []
    relabel = {item['word']: item['majority'] for item in chosen}
    for word, phones in (keep or {}).get('relabel', {}).items():
        if word in relabel:
            continue
        entry = counts.get(word)
        verdict = significant(entry) if entry and len(entry['votes']) > 1 and word not in context else None
        if verdict == 0:
            contradicted.append({'word': word, 'phones': phones, 'votes': entry['votes']})
        elif verdict is None or entry['variants'][verdict] != phones:
            kept.append({'word': word, 'phones': phones, 'evidence': keep['evidence'].get(word),
                         'votes': entry['votes'] if entry else None})
            relabel[word] = phones
    with open(os.path.join(out, 'relabel.json'), 'w', encoding='utf-8') as file:
        json.dump({'relabel': relabel, 'chosen': chosen, 'rejected': rejected, 'kept': kept, 'contradicted': contradicted}, file,
                  indent=1)
    return f'{len(chosen)} relabeled of {sum(item["significant"] for item in items)} significant candidates, {len(kept)} kept, ' + \
        f'{len(contradicted)} contradicted: ' + ', '.join(f'{item["word"]} {item["majority"]}' for item in chosen)


DESCRIPTION = (
    'The G2P\'s relabeled words (picovox_train/g2p.py relabel, the focus stage) - dictionary words whose first pronunciation '
    'isn\'t what the training speech\'s voices say (picovox_train/heteronyms.py): the significant majority reading of the '
    'corpus\'s measured tokens (picovox_train/measure.py - the CTC vote or the formants; the Wilson 95% lower bound over the '
    'first above one half, from 3+ books), a lexical difference (not vowel reduction or stress alone, not a function word); '
    'or, where the corpus can\'t judge a word (kept), its earlier evidence. evidence: first (the first variant), votes and '
    'books per reading, class, corpus, date - or kept, the earlier evidence')


# The committed file of a selection - {description, corpus, relabel, evidence}
def committed(result, corpus):
    today = datetime.date.today().isoformat()
    evidence = {item['word']: {'first': item['first'], 'votes': item['votes'], 'books': item['books'], 'class': item['class'],
                               'corpus': corpus, 'date': today} for item in result['chosen']}
    # (a kept word's earlier evidence once - kept again, its kept evidence)
    for item in result['kept']:
        earlier = item['evidence']['kept'] if item['evidence'] and 'kept' in item['evidence'] else item['evidence']
        evidence[item['word']] = {'kept': earlier, 'votes': item['votes']}
    return {'description': DESCRIPTION, 'corpus': corpus, 'relabel': result['relabel'], 'evidence': evidence}


# The committed file's text - its relabels a line each, and each word's evidence on one line
def committed_text(value):
    def entries(items):
        return ',\n'.join(f'        {json.dumps(key, ensure_ascii=False)}: {json.dumps(item, ensure_ascii=False)}'
                           for key, item in items.items())
    return (f'{{\n    "description": {json.dumps(value["description"])},\n    "corpus": {json.dumps(value["corpus"])},\n'
            f'    "relabel": {{\n{entries(value["relabel"])}\n    }},\n    "evidence": {{\n{entries(value["evidence"])}\n    }}\n}}\n')


# The pinned books' directory (the context labels' - books.py)
def books_dir():
    from . import contextlabels  # pylint: disable=import-outside-toplevel
    return contextlabels.BOOKS_DIR


def main(argv=None):
    from . import tts  # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(prog='python -m picovox_train.heteronyms',
                                     description='The G2P\'s relabeled words from the training speech\'s readings')
    parser.add_argument('--out', required=True, help='the output directory')
    parser.add_argument('--share', type=float, default=0.5, help='the least majority share (0.5: a majority)')
    parser.add_argument('--keep', help='the committed relabel file - its words the corpus can\'t judge are kept')
    parser.add_argument('--check', help='a relabel file (train/data/g2p-relabel.json) the result must equal (exit 1 if not)')
    parser.add_argument('--write', help='write the result as the committed relabel file')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    directory = tts.corpus_dir(tts.load_settings())
    keep = None
    if args.keep:
        with open(args.keep, encoding='utf-8') as file:
            keep = json.load(file)
    print(f'votes: {votes(args.out, directory)}', file=sys.stderr, flush=True)
    print(f'candidates: {candidates(args.out)}', file=sys.stderr, flush=True)
    context = set(readings.candidate_words(readings.load_spec()))
    with open(os.path.join(args.out, 'candidates.json'), encoding='utf-8') as file:
        capitals = books.book_capitals(books_dir(), [item['word'] for item in json.load(file)])
    names = {word for word, (capital, lower) in capitals.items() if capital > lower}
    print(f'select: {select(args.out, args.share, keep, context, names)}', file=sys.stderr, flush=True)
    with open(os.path.join(args.out, 'relabel.json'), encoding='utf-8') as file:
        result = json.load(file)
    if args.write:
        with open(args.write, 'w', encoding='utf-8') as file:
            file.write(committed_text(committed(result, os.path.basename(directory))))
    if args.check:
        with open(args.check, encoding='utf-8') as file:
            expected = json.load(file)
        relabel = result['relabel']
        if relabel != expected['relabel']:
            print(f'check: the relabeled words differ from {args.check}\'s (corpus {os.path.basename(directory)}, theirs '
                  f'{expected.get("corpus")}): ' +
                  ', '.join(f'{word} {relabel.get(word)} (was {expected["relabel"].get(word)})'
                            for word in sorted(set(relabel) | set(expected['relabel']))
                            if relabel.get(word) != expected['relabel'].get(word)),
                  file=sys.stderr)
            return 1
        print(f'check: the {len(relabel)} relabeled words equal {args.check}\'s', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
