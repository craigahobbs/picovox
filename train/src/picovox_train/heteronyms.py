# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The G2P's relabeled words (data/nn/g2p-relabel.json, g2p.py's focus stage) - dictionary words whose first
pronunciation isn't what readers say, from the LibriTTS-R readers' usage of the dictionary's variants (python is
PYTHONPATH=train/src build/venv-nn/bin/python -B):

    python -m picovox_train.heteronyms --out build/nn/heteronyms [-j 6]                     # every step (resumable)
    python -m picovox_train.heteronyms --out DIR --check data/nn/g2p-relabel.json           # and the same words?

1. readers - for every word with 2+ pronunciations, which variant the readers used, by the MFA word and phone alignments
   (data/corpora/libritts_label/lab/{word,phone}/{train-clean-100,dev-clean}): each aligned token's phones are matched
   to the variants exactly (with stress), else ignoring stress when that leaves one variant (readers.json, tokens.jsonl)
2. candidates - words whose readers' majority variant isn't the first, significant (the Wilson 95% lower bound of
   majority / (majority + first) above one half, used by 3+ speakers), with the kind of difference (candidates.json)
3. acoustic - a listening-free check of the aligner's choices: for each significant candidate differing in a vowel's
   quality, the vowel's F1 and F2 (the middle third, Lobanov-normalized per speaker) classified by the nearest pooled
   reference vowel - the first's, the majority's, or schwa; for a voicing difference, the fricative's voiced share
   (acoustic.json)
4. select - the relabeled words: significant candidates, not function words, whose difference is lexical (a stressed
   vowel's quality, or the primary stress moving with a quality change - not reduction or stress alone), with the
   aligner's majority share at least --share, and where the acoustic check measured 5+ tokens, its majority share at
   least --share (at 0.5, a simple majority) and schwa no commoner than the majority's vowel unless that vowel is AH
   (relabel.json - data/nn/g2p-relabel.json's format)
"""

import argparse
import collections
import json
import math
import multiprocessing
import os
import random
import re
import sys

import numpy

from .frontend import ROOT


LABELS = os.path.join(ROOT, 'data', 'corpora', 'libritts_label', 'lab')
AUDIO = os.path.join(ROOT, 'data', 'corpora', 'libritts_r', 'LibriTTS_R')
SUBSETS = ('train-clean-100', 'dev-clean')
VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
REDUCED = {'AH', 'IH', 'ER', 'IY', 'UW'}
VOICING = {('DH', 'TH'), ('TH', 'DH'), ('Z', 'S'), ('S', 'Z'), ('ZH', 'SH'), ('SH', 'ZH'), ('V', 'F'), ('F', 'V')}

# Closed-class words - their reduced forms are contextual (stress, phrase position), so they keep the dictionary's first
FUNCTION_WORDS = set("""
a an the this that these those some any each every no
i me my mine myself you your yours yourself yourselves he him his himself she her hers herself it its itself we us our ours
ourselves they them their theirs themselves one who whom whose which what that's
of to in on at by for from with within without into onto upon over under about above below across after against along
among amongst around before behind beneath beside besides between beyond but despite down during except inside like
near off out outside past per since than through throughout till toward towards underneath unless until up via whereas
re and or nor so yet if because although though while whether as either neither both then there here
am is are was were be been being have has had having do does did done shall should will would can could may might must
ought let's i'll you'll he'll she'll we'll they'll it'll i'd you'd he'd she'd we'd they'd i've you've we've they've
i'm you're we're they're he's she's it's isn't aren't wasn't weren't don't doesn't didn't hasn't haven't hadn't won't
wouldn't can't cannot couldn't shouldn't mustn't there's here's what's who's
""".split())


# A phone without its stress
def base(phone):
    return re.sub(r'\d', '', phone)


# An MFA label file's intervals - [(start, end, label)]
def read_lab(path):
    rows = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 3 and fields[2]:
                rows.append((float(fields[0]), float(fields[1]), fields[2]))
    return rows


# The dictionary - word to its pronunciations (every entry)
def read_dictionary():
    pronunciations = collections.defaultdict(list)
    with open(os.path.join(ROOT, 'data', 'cmudict.dict'), encoding='utf-8') as file:
        for line in file:
            content = line.split('#')[0].strip()
            if content and not content.startswith(';;;'):
                word, *phones = content.split()
                pronunciations[re.sub(r'\(\d+\)$', '', word)].append(phones)
    return pronunciations


# The edit alignment of two phone sequences (a stress-only substitution costs half) - the cost table
def edit_costs(first, second):
    def substitution(one, two):
        return 0 if one == two else (0.5 if base(one) == base(two) else 1)
    cost = [[0.0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for row in range(len(first) + 1):
        cost[row][0] = row
    for column in range(len(second) + 1):
        cost[0][column] = column
    for row in range(1, len(first) + 1):
        for column in range(1, len(second) + 1):
            cost[row][column] = min(cost[row - 1][column] + 1, cost[row][column - 1] + 1,
                                    cost[row - 1][column - 1] + substitution(first[row - 1], second[column - 1]))
    return cost, substitution


# The differing (first phone or None, second phone or None) pairs of the edit alignment
def edit_ops(first, second):
    cost, substitution = edit_costs(first, second)
    ops, row, column = [], len(first), len(second)
    while row > 0 or column > 0:
        if row > 0 and column > 0:
            sub = substitution(first[row - 1], second[column - 1])
            if cost[row][column] == cost[row - 1][column - 1] + sub:
                if sub:
                    ops.append((first[row - 1], second[column - 1]))
                row, column = row - 1, column - 1
                continue
        if row > 0 and cost[row][column] == cost[row - 1][column] + 1:
            ops.append((first[row - 1], None))
            row -= 1
        else:
            ops.append((None, second[column - 1]))
            column -= 1
    return ops[::-1]


#
# 1. The readers' usage
#

# One utterance's tokens of multi-pronunciation words - [(word, phones, previous word, next word, start, end, phone
# intervals, label path)]
def utterance_tokens(task):
    word_path, multi = task
    phone_path = word_path.replace(os.sep + 'word' + os.sep, os.sep + 'phone' + os.sep, 1)
    if not os.path.exists(phone_path):
        return []
    words = read_lab(word_path)
    phones = read_lab(phone_path)
    tokens = []
    for index, (start, end, word) in enumerate(words):
        if word in multi:
            inside = [(s, e, p) for s, e, p in phones if s >= start - 1e-4 and e <= end + 1e-4 and p not in ('sp', 'sil', 'spn')]
            previous = words[index - 1][2] if index > 0 else '<s>'
            following = words[index + 1][2] if index + 1 < len(words) else '</s>'
            tokens.append((word, ' '.join(p for _, _, p in inside), previous, following, start, end, inside,
                           os.path.relpath(word_path, LABELS)))
    return tokens


def readers(out, jobs):
    """readers.json (per word: variants, exact and stress-only matches, ambiguous, other, speakers) and tokens.jsonl"""
    multi = {word: variants for word, variants in read_dictionary().items() if len(variants) >= 2}
    paths = []
    for subset in SUBSETS:
        for dirpath, _, names in os.walk(os.path.join(LABELS, 'word', subset)):
            paths.extend(os.path.join(dirpath, name) for name in names if name.endswith('.lab'))
    paths.sort()
    multi_set = set(multi)
    with multiprocessing.Pool(jobs) as pool:
        results = pool.map(utterance_tokens, [(path, multi_set) for path in paths], chunksize=64)
    counts = {}
    with open(os.path.join(out, 'tokens.jsonl'), 'w', encoding='utf-8') as tokens_file:
        for result in results:
            for word, phones, previous, following, start, end, inside, path in result:
                variants = multi[word]
                entry = counts.setdefault(word, {'variants': [' '.join(v) for v in variants], 'exact': [0] * len(variants),
                                                 'stressOnly': [0] * len(variants), 'ambiguous': 0,
                                                 'other': collections.Counter(), 'speakers': [collections.Counter() for _ in variants]})
                observed = phones.split()
                exact = [index for index, variant in enumerate(variants) if variant == observed]
                loose = [index for index, variant in enumerate(variants) if [base(p) for p in variant] == [base(p) for p in observed]]
                choice = None
                if exact:
                    choice = exact[0]
                    entry['exact'][choice] += 1
                elif len(loose) == 1:
                    choice = loose[0]
                    entry['stressOnly'][choice] += 1
                elif loose:
                    entry['ambiguous'] += 1
                else:
                    entry['other'][phones] += 1
                if choice is not None:
                    entry['speakers'][choice][path.split(os.sep)[2]] += 1
                tokens_file.write(json.dumps({'word': word, 'phones': phones, 'choice': choice, 'prev': previous, 'next': following,
                                              'start': start, 'end': end, 'segments': inside, 'path': path}) + '\n')
    for entry in counts.values():
        entry['otherTotal'] = sum(entry['other'].values())
        entry['other'] = dict(entry['other'].most_common(5))
        entry['speakerCounts'] = [len(speakers) for speakers in entry['speakers']]
        del entry['speakers']
    with open(os.path.join(out, 'readers.json'), 'w', encoding='utf-8') as file:
        json.dump(counts, file)
    return f'{len(paths)} utterances, {len(multi)} multi-pronunciation words, {len(counts)} seen'


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


# The kind of difference: 'stress' (stress digits only), 'reduced' (only unstressed reduced vowels swap),
# 'consonant-drop' (only consonants inserted or deleted), 'syllable' (a vowel inserted, deleted, or traded for a
# consonant), 'vowel' (a vowel's quality changes), else 'consonant'
def kind(first, other):
    ops = edit_ops(first, other)
    if all(x is not None and y is not None and base(x) == base(y) for x, y in ops):
        return 'stress'

    def vowel(phone):
        return phone is not None and base(phone) in VOWELS

    def unstressed(phone):
        return phone is not None and base(phone) in REDUCED and phone[-1] == '0'
    if all(x is not None and y is not None and unstressed(x) and unstressed(y) for x, y in ops):
        return 'reduced'
    if all((x is None or not vowel(x)) and (y is None or not vowel(y)) and (x is None or y is None) for x, y in ops):
        return 'consonant-drop'
    if any((x is None and vowel(y)) or (y is None and vowel(x)) or (x and y and vowel(x) != vowel(y)) for x, y in ops) and \
            all(not (vowel(x) and vowel(y)) or base(x) == base(y) for x, y in ops):
        return 'syllable'
    if any(vowel(x) and vowel(y) and base(x) != base(y) for x, y in ops) or any(vowel(x) != vowel(y) for x, y in ops if x and y):
        return 'vowel'
    return 'consonant'


def candidates(out):
    """candidates.json - every word with a non-first majority"""
    with open(os.path.join(out, 'readers.json'), encoding='utf-8') as file:
        counts = json.load(file)
    items = []
    for word, entry in counts.items():
        matched = [exact + loose for exact, loose in zip(entry['exact'], entry['stressOnly'])]
        best = max(range(len(matched)), key=lambda index, matched=matched: (matched[index], -index))
        if best == 0:
            continue
        lower = wilson_lower(matched[best], matched[best] + matched[0])
        variants = [variant.split() for variant in entry['variants']]
        items.append({'word': word, 'first': entry['variants'][0], 'majority': entry['variants'][best], 'index': best,
                      'counts': matched, 'variants': entry['variants'], 'speakers': entry['speakerCounts'],
                      'other': entry['otherTotal'], 'lower': lower,
                      'significant': lower > 0.5 and entry['speakerCounts'][best] >= 3,
                      'kind': kind(variants[0], variants[best])})
    items.sort(key=lambda item: -sum(item['counts']))
    with open(os.path.join(out, 'candidates.json'), 'w', encoding='utf-8') as file:
        json.dump(items, file, indent=1)
    return f'{len(items)} words with a non-first majority, {sum(1 for item in items if item["significant"])} significant'


#
# 3. The acoustic check
#

# The differing position of two variants - (index in the first, index in the majority, 'vowel' or 'voicing') or None
def difference(first, majority):
    cost, substitution = edit_costs(first, majority)
    row, column, pairs = len(first), len(majority), []
    while row > 0 and column > 0:
        if cost[row][column] == cost[row - 1][column - 1] + substitution(first[row - 1], majority[column - 1]):
            pairs.append((row - 1, column - 1))
            row, column = row - 1, column - 1
        elif cost[row][column] == cost[row - 1][column] + 1:
            row -= 1
        else:
            column -= 1
    for row, column in reversed(pairs):
        if base(first[row]) != base(majority[column]) and base(first[row]) in VOWELS and base(majority[column]) in VOWELS:
            return row, column, 'vowel'
    for row, column in reversed(pairs):
        if (base(first[row]), base(majority[column])) in VOICING:
            return row, column, 'voicing'
    return None


# Measure an utterance - its target tokens' formants or voiced share, and its other vowels' formants (the references)
def measure(task):
    import parselmouth  # pylint: disable=import-outside-toplevel
    rel, sex, wanted = task
    subset, speaker, chapter, name = rel.split(os.sep)
    wav = os.path.join(AUDIO, subset, speaker, chapter, name[:-4] + '.wav')
    if not os.path.exists(wav):
        return None
    sound = parselmouth.Sound(wav)  # pylint: disable=c-extension-no-member # (a compiled module pylint can't inspect)
    formant = sound.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5500 if sex == 'F' else 5000)
    pitch = None
    phones = read_lab(os.path.join(LABELS, 'phone', rel))

    def formants(start, end):
        times = numpy.linspace(start + (end - start) / 3, end - (end - start) / 3, 5)
        first = numpy.nanmedian([formant.get_value_at_time(1, time) for time in times])
        second = numpy.nanmedian([formant.get_value_at_time(2, time) for time in times])
        return (float(first), float(second)) if numpy.isfinite(first) and numpy.isfinite(second) else None

    vowels = []
    targets = []
    for word_start, word_end, word in read_lab(os.path.join(LABELS, 'word', rel)):
        inside = [(s, e, p) for s, e, p in phones if s >= word_start - 1e-4 and e <= word_end + 1e-4]
        key = (round(word_start, 3), word)
        if key in wanted:
            label_index, difference_kind = wanted[key]
            if label_index < len(inside):
                start, end, phone = inside[label_index]
                if difference_kind == 'vowel' and end - start >= 0.04:
                    value = formants(start, end)
                    if value:
                        targets.append({'word': word, 'start': word_start, 'phone': phone, 'f': value})
                elif difference_kind == 'voicing' and end - start >= 0.03:
                    if pitch is None:
                        pitch = sound.to_pitch_ac(time_step=0.005, pitch_floor=75, pitch_ceiling=500)
                    voiced = [pitch.get_value_at_time(time) for time in numpy.arange(start + 0.005, end - 0.004, 0.005)]
                    share = float(numpy.mean([numpy.isfinite(v) and v > 0 for v in voiced])) if len(voiced) else None
                    targets.append({'word': word, 'start': word_start, 'phone': phone, 'voiced': share})
        else:
            for start, end, phone in inside:
                if base(phone) in VOWELS and end - start >= 0.05:
                    value = formants(start, end)
                    if value:
                        vowels.append((phone, value))
    return {'rel': rel, 'speaker': speaker, 'vowels': vowels, 'targets': targets}


def acoustic(out, jobs, per_word):
    """acoustic.json - per significant candidate, the aligner's and the acoustic check's votes"""
    rng = random.Random(1)
    with open(os.path.join(out, 'candidates.json'), encoding='utf-8') as file:
        significant = [item for item in json.load(file) if item['significant']]
    sexes = {}
    with open(os.path.join(AUDIO, 'speakers.tsv'), encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 2:
                sexes[fields[0]] = fields[1]
    plan = {}
    for item in significant:
        first, majority = item['first'].split(), item['majority'].split()
        found = difference(first, majority)
        if found:
            plan[item['word']] = {'first': first, 'majority': majority, 'firstIndex': found[0], 'majorityIndex': found[1],
                                  'kind': found[2]}

    # The tokens to measure - labeled with the first or the majority variant, up to per_word a word
    by_word = collections.defaultdict(list)
    with open(os.path.join(out, 'tokens.jsonl'), encoding='utf-8') as file:
        for line in file:
            token = json.loads(line)
            entry = plan.get(token['word'])
            if entry is not None:
                phones = [base(phone) for phone in token['phones'].split()]
                if phones == [base(phone) for phone in entry['first']]:
                    by_word[token['word']].append((token, entry['firstIndex'], 'first'))
                elif phones == [base(phone) for phone in entry['majority']]:
                    by_word[token['word']].append((token, entry['majorityIndex'], 'majority'))
    wanted = collections.defaultdict(dict)
    labels = {}
    for word, tokens in by_word.items():
        rng.shuffle(tokens)
        for token, index, label in tokens[:per_word]:
            rel = os.path.relpath(token['path'], 'word')
            wanted[rel][(round(token['start'], 3), word)] = (index, plan[word]['kind'])
            labels[(rel, round(token['start'], 3), word)] = label

    # Reference utterances - 12 more a speaker
    by_speaker = collections.defaultdict(list)
    for subset in SUBSETS:
        for dirpath, _, names in os.walk(os.path.join(LABELS, 'word', subset)):
            for name in names:
                rel = os.path.relpath(os.path.join(dirpath, name), os.path.join(LABELS, 'word'))
                by_speaker[rel.split(os.sep)[1]].append(rel)
    for paths in by_speaker.values():
        paths.sort()
        rng.shuffle(paths)
        for rel in paths[:12]:
            wanted.setdefault(rel, {})
    tasks = [(rel, sexes.get(rel.split(os.sep)[1], 'M'), items) for rel, items in sorted(wanted.items())]
    with multiprocessing.Pool(jobs) as pool:
        results = [result for result in pool.imap_unordered(measure, tasks, chunksize=8) if result]

    # Lobanov normalization per speaker, and the pooled reference vowels
    speaker_vowels = collections.defaultdict(list)
    for result in results:
        speaker_vowels[result['speaker']].extend(result['vowels'])
    norms = {}
    for speaker, vowels in speaker_vowels.items():
        values = numpy.array([value for _, value in vowels])
        if len(values) >= 30:
            norms[speaker] = (values.mean(0), values.std(0))
    references = collections.defaultdict(list)
    for speaker, vowels in speaker_vowels.items():
        if speaker in norms:
            mean, std = norms[speaker]
            for phone, value in vowels:
                references[phone].append((numpy.array(value) - mean) / std)
    reference_means = {phone: numpy.mean(values, 0) for phone, values in references.items() if len(values) >= 20}

    # Classify the target tokens
    report = {}
    for result in results:
        if result['speaker'] not in norms:
            continue
        mean, std = norms[result['speaker']]
        for target in result['targets']:
            word = target['word']
            entry = plan[word]
            row = report.setdefault(word, {'kind': entry['kind'], 'first': ' '.join(entry['first']),
                                           'majority': ' '.join(entry['majority']), 'tokens': 0,
                                           'aligner': collections.Counter(), 'acoustic': collections.Counter(), 'agree': 0})
            aligner = labels[(result['rel'], round(target['start'], 3), word)]
            if entry['kind'] == 'vowel':
                options = {'first': entry['first'][entry['firstIndex']], 'majority': entry['majority'][entry['majorityIndex']],
                           'schwa': 'AH0'}
                options = {name: phone if phone in reference_means else re.sub(r'\d', '1', phone) for name, phone in options.items()}
                normalized = (numpy.array(target['f']) - mean) / std
                distances = {name: float(numpy.linalg.norm(normalized - reference_means[phone])) for name, phone in options.items()
                             if phone in reference_means}
                if options['schwa'] in (options['first'], options['majority']):
                    distances.pop('schwa', None)
                vote = min(distances, key=distances.get)
            else:
                if target['voiced'] is None:
                    continue
                voiced_first = base(entry['first'][entry['firstIndex']]) in ('DH', 'Z', 'ZH', 'V')
                vote = 'first' if (target['voiced'] >= 0.5) == voiced_first else 'majority'
            row['tokens'] += 1
            row['aligner'][aligner] += 1
            row['acoustic'][vote] += 1
            row['agree'] += aligner == vote
    for row in report.values():
        row['aligner'] = dict(row['aligner'])
        row['acoustic'] = dict(row['acoustic'])
    with open(os.path.join(out, 'acoustic.json'), 'w', encoding='utf-8') as file:
        json.dump({'references': {phone: [float(x) for x in value] for phone, value in reference_means.items()}, 'words': report},
                  file, indent=1)
    return f'{len(plan)} words checked, {sum(row["tokens"] for row in report.values())} tokens classified'


#
# 4. The selection
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


def select(out, share):
    """relabel.json - {relabel: {word: phones}, chosen, rejected}"""
    with open(os.path.join(out, 'candidates.json'), encoding='utf-8') as file:
        significant = [item for item in json.load(file) if item['significant']]
    with open(os.path.join(out, 'acoustic.json'), encoding='utf-8') as file:
        checks = json.load(file)['words']
    chosen, rejected = [], []
    for item in significant:
        majority, first = item['counts'][item['index']], item['counts'][0]
        reasons = []
        if item['word'] in FUNCTION_WORDS:
            reasons.append('function word')
        if item['kind'] not in ('vowel', 'consonant'):
            reasons.append(f'kind {item["kind"]}')
        elif item['kind'] == 'vowel' and not lexical(item['first'], item['majority']):
            reasons.append('reduction only')
        if majority / (majority + first) < share:
            reasons.append(f'aligner share {majority / (majority + first):.2f}')
        check = checks.get(item['word'])
        if check and check['acoustic'].get('majority', 0) + check['acoustic'].get('first', 0) >= 5:
            votes_majority, votes_first, votes_schwa = (check['acoustic'].get(key, 0) for key in ('majority', 'first', 'schwa'))
            if (votes_majority / (votes_majority + votes_first) < share) if share > 0.5 else votes_majority <= votes_first:
                reasons.append(f'acoustic share {votes_majority / (votes_majority + votes_first):.2f}')
            if check['kind'] == 'vowel' and votes_schwa > votes_majority:
                new_vowels = {base(phone) for phone in item['majority'].split()} - {base(phone) for phone in item['first'].split()}
                if 'AH' not in new_vowels:
                    reasons.append('acoustic schwa')
            item['acoustic'] = check['acoustic']
        (rejected if reasons else chosen).append(dict(item, reasons=reasons))
    with open(os.path.join(out, 'relabel.json'), 'w', encoding='utf-8') as file:
        json.dump({'relabel': {item['word']: item['majority'] for item in chosen}, 'chosen': chosen, 'rejected': rejected}, file,
                  indent=1)
    return f'{len(chosen)} relabeled of {len(significant)} significant candidates: ' + \
        ', '.join(f'{item["word"]} {item["majority"]}' for item in chosen)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.heteronyms',
                                     description='The G2P\'s relabeled words from readers\' usage')
    parser.add_argument('--out', required=True, help='the output directory')
    parser.add_argument('-j', '--jobs', type=int, default=6)
    parser.add_argument('--per-word', type=int, default=240, help='the acoustic check\'s tokens a word')
    parser.add_argument('--share', type=float, default=0.5, help='the least majority share, aligner and acoustic (0.5: a majority)')
    parser.add_argument('--check', help='a relabel file (data/nn/g2p-relabel.json) the result must equal (exit 1 if not)')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    steps = [('readers', 'readers.json', lambda: readers(args.out, args.jobs)),
             ('candidates', 'candidates.json', lambda: candidates(args.out)),
             ('acoustic', 'acoustic.json', lambda: acoustic(args.out, args.jobs, args.per_word)),
             ('select', 'relabel.json', lambda: select(args.out, args.share))]
    for name, output, step in steps:
        if os.path.exists(os.path.join(args.out, output)):
            print(f'{name}: done before', file=sys.stderr, flush=True)
        else:
            print(f'{name}: {step()}', file=sys.stderr, flush=True)
    if args.check:
        with open(os.path.join(args.out, 'relabel.json'), encoding='utf-8') as file:
            relabel = json.load(file)['relabel']
        with open(args.check, encoding='utf-8') as file:
            expected = json.load(file)['relabel']
        if relabel != expected:
            print(f'check: the relabeled words differ from {args.check}: ' +
                  ', '.join(f'{word} {relabel.get(word)} (was {expected.get(word)})' for word in sorted(set(relabel) | set(expected))
                            if relabel.get(word) != expected.get(word)), file=sys.stderr)
            return 1
        print(f'check: the {len(relabel)} relabeled words equal {args.check}\'s', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
