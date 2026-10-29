# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Heteronym readings - the pronouncing dictionary's variants of a word clustered into readings, how two readings are told
apart in the audio (pair_class - measure.py measures them in the training speech), the words the corpus measures and
covers (the inventory, and the context labels' candidate words), and the corpus's measured readings (read_readings, and
the overrides the voice's tokens take - tts.py). The phone helpers (edit alignment, kinds of difference) are shared with
heteronyms.py and contextlabels.py.

Readings: two variants are different readings when they differ lexically (a stressed vowel's quality; the primary
stress moving with a quality change), by a consonant (close S/Z), by the primary stress moving alone (the noun/verb
pairs: insult, conduct), or by a secondary-stressed full vowel against a reduced one (the -ate words); otherwise
(reduction, stress) they are forms of one reading. The inventory is every dictionary word with 2+ readings, except the
closed-class (function) words, whose reduced forms are contextual - the front end keeps the G2P's form for those.

"""

import json
import os
import re

from . import ROOT, corpora, read_json
from .frontend import BOUNDARIES
from .frontend import FUNCTION_WORDS as FRONT_END_FUNCTION_WORDS


VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
REDUCED = {'AH', 'IH', 'ER', 'IY', 'UW'}
VOICING = {('DH', 'TH'), ('TH', 'DH'), ('Z', 'S'), ('S', 'Z'), ('ZH', 'SH'), ('SH', 'ZH'), ('V', 'F'), ('F', 'V')}

# Closed-class words (heteronyms.py's relabeling skips them) - their reduced forms are contextual (stress, phrase
# position), so they keep the dictionary's first
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


# The kind of difference: 'stress' (stress digits only), 'reduced' (only unstressed reduced vowels swap),
# 'consonant-drop' (only consonants inserted or deleted), 'syllable' (a vowel inserted, deleted, or traded for a
# consonant), 'vowel' (a vowel's quality changes), else 'consonant'
def kind(first, other):
    ops = edit_ops(first, other)
    if all(x is not None and y is not None and base(x) == base(y) for x, y in ops):
        return 'stress'

    def vowel(phone):
        return phone is not None and base(phone) in VOWELS

    def reduced(phone):
        return phone is not None and base(phone) in REDUCED and phone[-1] == '0'
    if all(x is not None and y is not None and reduced(x) and reduced(y) for x, y in ops):
        return 'reduced'
    if all((x is None or not vowel(x)) and (y is None or not vowel(y)) and (x is None or y is None) for x, y in ops):
        return 'consonant-drop'
    if any((x is None and vowel(y)) or (y is None and vowel(x)) or (x and y and vowel(x) != vowel(y)) for x, y in ops) and \
            all(not (vowel(x) and vowel(y)) or base(x) == base(y) for x, y in ops):
        return 'syllable'
    if any(vowel(x) and vowel(y) and base(x) != base(y) for x, y in ops) or any(vowel(x) != vowel(y) for x, y in ops if x and y):
        return 'vowel'
    return 'consonant'


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


# Closed-class words (heteronyms.py's) and the function words - their reduced forms are contextual (stress, phrase
# position), so they keep the G2P's form
CLOSED_CLASS = FUNCTION_WORDS | FRONT_END_FUNCTION_WORDS


# Phones (a string) without stress digits
def unstressed(phones):
    return ' '.join(base(phone) for phone in phones.split())


# The vowel pairs that differ between two variants - by syllable, or by the edit alignment when the counts differ
def vowel_pairs(first, second):
    first_vowels = [phone for phone in first.split() if base(phone) in VOWELS]
    second_vowels = [phone for phone in second.split() if base(phone) in VOWELS]
    if len(first_vowels) == len(second_vowels):
        return [(left, right) for left, right in zip(first_vowels, second_vowels) if left != right]
    return [(left, right) for left, right in edit_ops(first.split(), second.split())
            if left and right and base(left) in VOWELS and base(right) in VOWELS]


# Whether two variants are different readings (see the module's description)
def distinct(first, second):
    pairs = vowel_pairs(first, second)
    if any(left[-1] in '12' and right[-1] in '12' and base(left) != base(right) for left, right in pairs):
        return True
    primary_moves = any(left[-1] == '1' and right[-1] != '1' for left, right in pairs) and \
        any(left[-1] != '1' and right[-1] == '1' for left, right in pairs)
    if primary_moves or kind(first.split(), second.split()) == 'consonant':
        return True
    return any({left[-1], right[-1]} == {'2', '0'} and base(left) != base(right) for left, right in pairs)


# The readings of a word's variants (dictionary order) - lists of variants; each variant joins the first reading whose
# first variant it is not distinct from
def readings_of(variants):
    clusters = []
    for variant in variants:
        home = next((cluster for cluster in clusters if not distinct(cluster[0], variant)), None)
        if home is None:
            clusters.append([variant])
        else:
            home.append(variant)
    return clusters


# Every dictionary word's variants (phone strings, dictionary order, without repeats)
def read_variants():
    variants = {}
    for word, pronunciations in corpora.read_dictionary().items():
        variants[word] = []
        for phones in pronunciations:
            if ' '.join(phones) not in variants[word]:
                variants[word].append(' '.join(phones))
    return variants


# The heteronym inventory - {word: readings} for every dictionary word with 2+ readings, closed-class words excepted
def inventory(variants=None):
    variants = read_variants() if variants is None else variants
    result = {}
    for word, word_variants in variants.items():
        if len(word_variants) >= 2 and word not in CLOSED_CLASS:
            readings = readings_of(word_variants)
            if len(readings) >= 2:
                result[word] = readings
    return result


# Whether two pronunciations (strings) differ only by S against Z (and stress) - the stem-final voicing pairs (use,
# close, excuse), which context.py's gate leaves aside
def sz_only(first, second):
    ops = edit_ops(first.split(), second.split())
    voicing = [left is not None and right is not None and {base(left), base(right)} == {'S', 'Z'} for left, right in ops]
    stress = [left is not None and right is not None and base(left) == base(right) for left, right in ops]
    return any(voicing) and all(flag or same for flag, same in zip(voicing, stress))


# The reading of phones (a string) - the reading with that variant, else the one reading with it ignoring stress; or None
def reading_of(phones, readings):
    for index, reading in enumerate(readings):
        if phones in reading:
            return index
    loose = [index for index, reading in enumerate(readings) if unstressed(phones) in {unstressed(variant) for variant in reading}]
    return loose[0] if len(loose) == 1 else None


# The words' phones (strings) of a token sequence - split at the word boundaries (default the front end's) and phrase
# ends, after <bos>
def token_words(tokens, boundaries=None):
    boundaries = BOUNDARIES if boundaries is None else boundaries
    words, current = [], []
    for token in tokens[1:]:
        if token in boundaries:
            words.append(' '.join(current))
            current = []
        else:
            current.append(token)
    return words


#
# How a reading is measured, and the corpus's coverage
#

# The recognizer tokens of a phone that tell its quality - its strict tokens (ctc.py) without the r-colored ones
def quality_tokens(phone):
    from . import ctc  # pylint: disable=import-outside-toplevel
    return {token for token in ctc.STRICT_TOKENS[base(phone)] if not (len(token) > 1 and token.endswith('ɹ'))}


# The classes in order of preference (a word's class is its readings' best)
CLASSES = ('strong', 'voicing', 'weak', 'merger', 'stress')


# How two readings' first variants (strings) are told apart in the audio - 'strong' (a full vowel's quality or a
# consonant the recognizer hears apart: the CTC vote), 'voicing' (S against Z: the CTC vote with S and Z strict),
# 'weak' (a reduced vowel against a full one, or a phone added or dropped: the formants), 'merger' (the vowels the
# recognizer merges: the formants), or 'stress' (stress alone - not measured)
def pair_class(first, other):
    from . import ctc  # pylint: disable=import-outside-toplevel
    if unstressed(first) == unstressed(other):
        return 'stress'
    if ctc.merged(first) == ctc.merged(other):
        return 'merger'
    classes = set()
    for left, right in edit_ops(first.split(), other.split()):
        if left is None or right is None:
            classes.add('weak')
        elif base(left) != base(right):
            left_vowel, right_vowel = base(left) in VOWELS, base(right) in VOWELS
            if left_vowel and right_vowel:
                if left[-1] == '0' or right[-1] == '0':
                    classes.add('weak')
                else:
                    classes.add('merger' if quality_tokens(left) & quality_tokens(right) else 'strong')
            elif not left_vowel and not right_vowel:
                classes.add('voicing' if {base(left), base(right)} == {'S', 'Z'} else 'strong')
            else:
                classes.add('weak')
    return next((name for name in CLASSES if name in classes), 'stress')


# A word's class - the best of its readings' classes against the first
def word_class(readings):
    classes = {pair_class(readings[0][0], reading[0]) for reading in readings[1:]}
    return next(name for name in CLASSES if name in classes)


# The context labels' curated spec (contextlabels.py) - train/data/context-labels.json
SPEC_PATH = os.path.join(ROOT, 'train', 'data', 'context-labels.json')


# The curated spec
def load_spec(path=SPEC_PATH):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# A variant's vowels
def variant_vowels(variant):
    return [phone for phone in variant.split() if phone[-1].isdigit()]


# The index of a variant's primary-stressed vowel (None if none)
def primary_vowel(variant):
    return next((index for index, phone in enumerate(variant_vowels(variant)) if phone.endswith('1')), None)


# A word's family by its first two readings - 'stress' (the primary stress moves: noun/verb pairs), 'ate' (a final -ate
# full vowel against a reduced one), 'consonant' (S/Z), or 'vowel'
def word_family(word, readings):
    first, second = readings[0][0], readings[1][0]
    first_vowels, second_vowels = variant_vowels(first), variant_vowels(second)
    if len(first_vowels) == len(second_vowels) >= 2 and primary_vowel(first) != primary_vowel(second):
        return 'stress'
    if re.search(r'ates?$', word) and (first_vowels[-1:] + second_vowels[-1:]).count('EY2') == 1:
        return 'ate'
    if kind(first.split(), second.split()) == 'consonant':
        return 'consonant'
    return 'vowel'


# Whether a family word's readings are a stem-final voicing pair (use, close, excuse, uses - not an accent's S/Z or CH/SH)
def voicing_pair(readings):
    first, second = readings[0][0].split(), readings[1][0].split()
    ops = edit_ops(first, second)
    if len(ops) != 1 or None in ops[0] or {ops[0][0], ops[0][1]} != {'S', 'Z'}:
        return False
    position = next(index for index, (left, right) in enumerate(zip(first, second)) if left != right)
    return position >= len(first) - 3


# The candidate words and their readings (dictionary order) - the inventory's family words (the noun/verb stress pairs,
# the -ate words, the S/Z words) and the spec's words (with the readings the dictionary lacks) - {word: {readings, family,
# tags, curated}}; tags maps a tag ("Cap" + tag for a capitalized token) to a reading for a word's own mapping,
# before its family's
def candidate_words(spec, heteronyms=None):
    heteronyms = inventory() if heteronyms is None else heteronyms
    variants = read_variants()
    words = {}
    for word, readings in heteronyms.items():
        family = word_family(word, readings)
        if family == 'consonant' and not voicing_pair(readings):
            family = 'vowel'
        if family != 'vowel' and word not in spec['exclude']:
            words[word] = {'readings': readings, 'family': family, 'tags': {}, 'curated': False}
    for word, entry in spec['words'].items():
        readings = [list(reading) for reading in heteronyms.get(word) or readings_of(variants[word])]
        readings.extend([variant] for variant in entry.get('missing', []))
        tags = {}
        for variant, tag_list in entry.get('tags', {}).items():
            reading = reading_of(variant, readings)
            for tag in tag_list:
                tags[tag] = reading
        words[word] = {'readings': readings, 'family': entry.get('family', 'own'), 'tags': tags,
                       'curated': True}
    return words


# The words whose readings the corpus measures - {word: readings}: the inventory's, and the context labels' candidate
# words (with the readings the dictionary lacks)
def measured_words():
    words = inventory()
    for word, entry in candidate_words(load_spec(), words).items():
        words[word] = entry['readings']
    return words


# The corpus's coverage quotas - {word: sentences}: the gate's words (coverage "gate"), the context labels' candidate
# words ("context"), and the other measured words ("other"); a word whose readings can't be measured, none
def coverage_quotas(coverage, gate_words):
    words = measured_words()
    context = set(candidate_words(load_spec(), inventory()))
    quotas = {}
    for word, readings in words.items():
        if word_class(readings) != 'stress':
            quotas[word] = coverage['gate'] if word in gate_words else (coverage['context'] if word in context else coverage['other'])
    return quotas


# The corpus's measured clips - readings.jsonl's records (None if it isn't measured)
def read_readings(directory):
    path = os.path.join(directory, 'readings', 'readings.jsonl')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as file:
        return [json.loads(line) for line in file]


# The classes whose measured readings the voice's tokens take - the CTC vote's (the formants judge the G2P's evidence only)
OVERRIDE_CLASSES = ('strong', 'voicing')


# The measured readings that differ from the front end's (the voice's tokens take them) - {clip id: {position: phones}}
def overrides(directory):
    records = read_readings(directory)
    if records is None:
        return {}
    words = read_json(os.path.join(directory, 'readings', 'words.json'))['words']
    result = {}
    for record in records:
        for token in record['tokens']:
            if token['class'] in OVERRIDE_CLASSES and token['reading'] is not None and token['reading'] != token['frontEnd']:
                result.setdefault(record['id'], {})[str(token['position'])] = words[token['word']][token['reading']][0]
    return result
