# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Heteronym readings - the pronouncing dictionary's variants of a word clustered into readings, and the readings the
LibriTTS-R speakers used, by the Montreal Forced Aligner's word and phone labels (LibriTTSLabel). The voice training
data (multi.py build) takes a heteronym token's reading from the alignment where the aligner chose a dictionary
variant of another reading than the front end's - the speaker's reading, heard in the audio - unless the two differ only by
S against Z (the aligner's word-final voicing choice is unreliable: use, close, excuse), where the front end's is kept.

    # a build's heteronym tokens against the aligner
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.readings check --out DIR

Readings: two variants are different readings when they differ lexically (a stressed vowel's quality; the primary
stress moving with a quality change), by a consonant (close S/Z), by the primary stress moving alone (the noun/verb
pairs: insult, conduct), or by a secondary-stressed full vowel against a reduced one (the -ate words); otherwise
(reduction, stress) they are forms of one reading. The inventory is every dictionary word with 2+ readings, except the
closed-class (function) words, whose reduced forms are contextual - the front end keeps the G2P's form for those, and
for the variants of the front end's own reading (the voice learns reduction from the audio).

A token's variant is matched as readings.py matches it everywhere (the review's readers.py rule): the phones inside the
word's interval equal to a variant, else equal ignoring stress to exactly one variant; otherwise none.
"""

import argparse
import difflib
import json
import os
import re
import sys

from . import data
from .frontend import FUNCTION_WORDS, ROOT, text_to_phrases


DICTIONARY_PATH = os.path.join(ROOT, 'data', 'cmudict.dict')

# The share of a build's heteronym tokens that may still disagree with the aligner (the make train gate)
MAX_DISAGREE = 0.01

VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
REDUCED = {'AH', 'IH', 'ER', 'IY', 'UW'}

# Closed-class words - articles, pronouns and possessives, prepositions, conjunctions, auxiliaries and modals, and their
# contractions: their reduced forms are contextual (stress, phrase position), so they keep the G2P's form
CLOSED_CLASS = set("""
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
""".split()) | FUNCTION_WORDS


# A phone without its stress digit
def base(phone):
    return re.sub(r'\d', '', phone)


# Phones (a string) without stress digits
def unstressed(phones):
    return ' '.join(base(phone) for phone in phones.split())


# The Levenshtein alignment of two phone lists (a substitution of the same phone with another stress costs one half) -
# the (first's phone or None, second's phone or None) pairs that differ
def edit_ops(first, second):
    def substitution(left, right):
        return 0 if left == right else (0.5 if base(left) == base(right) else 1)
    rows, columns = len(first) + 1, len(second) + 1
    cost = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        cost[row][0] = row
    for column in range(columns):
        cost[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            cost[row][column] = min(cost[row - 1][column] + 1, cost[row][column - 1] + 1,
                                    cost[row - 1][column - 1] + substitution(first[row - 1], second[column - 1]))
    ops, row, column = [], len(first), len(second)
    while row > 0 or column > 0:
        if row > 0 and column > 0:
            cost_sub = substitution(first[row - 1], second[column - 1])
            if cost[row][column] == cost[row - 1][column - 1] + cost_sub:
                if cost_sub:
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


# Whether a difference (edit_ops) is a consonant substitution - not stress alone, not reduced vowels swapped, not consonants
# only inserted or deleted, no syllable gained or lost, and no vowel's quality changed
def consonant_change(first, second):
    ops = edit_ops(first, second)

    def vowel(phone):
        return phone is not None and base(phone) in VOWELS

    def reduced(phone):
        return phone is not None and base(phone) in REDUCED and phone[-1] == '0'
    if all(left is not None and right is not None and base(left) == base(right) for left, right in ops):
        return False
    if all(reduced(left) and reduced(right) for left, right in ops):
        return False
    if all(not vowel(left) and not vowel(right) and (left is None or right is None) for left, right in ops):
        return False
    syllable = any((left is None and vowel(right)) or (right is None and vowel(left)) or
                   (left and right and vowel(left) != vowel(right)) for left, right in ops)
    if syllable and all(not (vowel(left) and vowel(right)) or base(left) == base(right) for left, right in ops):
        return False
    return not (any(vowel(left) and vowel(right) and base(left) != base(right) for left, right in ops) or
                any(vowel(left) != vowel(right) for left, right in ops if left and right))


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
    if primary_moves or consonant_change(first.split(), second.split()):
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
def read_variants(path=DICTIONARY_PATH):
    variants = {}
    with open(path, encoding='utf-8') as file:
        for line in file:
            content = line.split('#')[0].strip()
            if content and not content.startswith(';;;'):
                word, *phones = content.split()
                word = re.sub(r'\(\d+\)$', '', word)
                phones = ' '.join(phones)
                if phones not in variants.setdefault(word, []):
                    variants[word].append(phones)
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


# Whether two pronunciations (strings) differ only by S against Z (and stress) - the aligner can't judge those
# (word-final voicing: use, close, excuse), so the front end's reading is kept
def sz_only(first, second):
    ops = edit_ops(first.split(), second.split())
    voicing = [left is not None and right is not None and {base(left), base(right)} == {'S', 'Z'} for left, right in ops]
    stress = [left is not None and right is not None and base(left) == base(right) for left, right in ops]
    return any(voicing) and all(flag or same for flag, same in zip(voicing, stress))


# The index of the variant observed phones (a string) match - exactly, else ignoring stress when that leaves one; or None
def match_variant(phones, variants):
    if phones in variants:
        return variants.index(phones)
    loose = [index for index, variant in enumerate(variants) if unstressed(variant) == unstressed(phones)]
    return loose[0] if len(loose) == 1 else None


# The reading of phones (a string) - the reading with that variant, else the one reading with it ignoring stress; or None
def reading_of(phones, readings):
    for index, reading in enumerate(readings):
        if phones in reading:
            return index
    loose = [index for index, reading in enumerate(readings) if unstressed(phones) in {unstressed(variant) for variant in reading}]
    return loose[0] if len(loose) == 1 else None


#
# The aligned readings of an utterance
#

# An MFA label file's labeled intervals - [(start, end, label)] (unlabeled silences dropped)
def read_lab(path):
    rows = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 3 and fields[2]:
                rows.append((float(fields[0]), float(fields[1]), fields[2]))
    return rows


# The word label path of a phone label path (lab/phone/... -> lab/word/...)
def word_lab_path(phone_lab_path):
    return phone_lab_path.replace(os.sep + 'phone' + os.sep, os.sep + 'word' + os.sep, 1)


# The front end's words of a text, with their phones as strings - [(word, phones)] in order (the readings' positions)
def front_end_words(text, pronounce_word=None):
    words = []
    for phrase in text_to_phrases(text, pronounce_word):
        for word in phrase['words']:
            phones = ' '.join(phone['ph'] if phone['stress'] is None else f'{phone["ph"]}{phone["stress"]}' for phone in word['phones'])
            words.append((word['text'], phones))
    return words


# The aligned heteronym tokens of an utterance - its text, its MFA phone label path, and the inventory - (overrides, tokens):
# overrides {position: phones list} where the aligner's reading differs from the front end's (the speaker's variant), and
# tokens [{position, word, reading (the aligner's), frontEnd (the front end's reading, None if none)}] for every inventory
# word the aligner matched to a dictionary variant
def utterance_readings(text, phone_lab, readings, pronounce_word=None):
    word_lab = word_lab_path(phone_lab)
    if not os.path.exists(word_lab) or not os.path.exists(phone_lab):
        return {}, []
    words = front_end_words(text, pronounce_word)
    labels = read_lab(word_lab)
    phones = [row for row in read_lab(phone_lab) if row[2] not in ('sp', 'sil', 'spn')]
    matcher = difflib.SequenceMatcher(None, [word for word, _ in words], [label for _, _, label in labels], autojunk=False)
    overrides, tokens = {}, []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            position, label_index = block.a + offset, block.b + offset
            word, front_phones = words[position]
            word_readings = readings.get(word)
            if word_readings is None:
                continue
            start, end, _ = labels[label_index]
            observed = ' '.join(phone for phone_start, phone_end, phone in phones
                                if phone_start >= start - 1e-4 and phone_end <= end + 1e-4)
            variants = [variant for reading in word_readings for variant in reading]
            choice = match_variant(observed, variants)
            if choice is None:
                continue
            variant = variants[choice]
            reading = reading_of(variant, word_readings)
            front_reading = reading_of(front_phones, word_readings)
            voicing = reading != front_reading and sz_only(variant, front_phones)
            tokens.append({'position': position, 'word': word, 'reading': reading, 'frontEnd': front_reading, 'variant': variant,
                           'szOnly': voicing})
            if reading != front_reading and not voicing:
                overrides[position] = variant.split()
    return overrides, tokens


# The words' phones (strings) of a token sequence - split at the word boundaries and phrase ends, after <bos>
def token_words(tokens, boundaries):
    words, current = [], []
    for token in tokens[1:]:
        if token in boundaries:
            words.append(' '.join(current))
            current = []
        else:
            current.append(token)
    return words


# Count a built utterance's heteronym tokens against the aligner - its tokens (names), the aligned tokens
# (utterance_readings), the inventory, and the totals to update ({tokens, disagree})
def count_disagreements(tokens, aligned, readings, boundaries, totals):
    words = token_words(tokens, boundaries)
    for token in aligned:
        position = token['position']
        built = words[position] if position < len(words) else None
        if built is not None and reading_of(built, readings[token['word']]) != token['reading'] and sz_only(token['variant'], built):
            totals['szOnly'] = totals.get('szOnly', 0) + 1
            continue
        totals['tokens'] += 1
        if built is None or reading_of(built, readings[token['word']]) != token['reading']:
            totals['disagree'] += 1


# The heteronym readings of a voice data build's LibriTTS-R tasks ((utterance, ...) tuples - the utterance dicts get
# 'readings', the overrides tokenize applies, and 'alignedReadings', the aligned tokens) - the counts {utterances,
# tokens, overrides}. The front end's own readings come from the pronouncer already loaded (frontend.py's G2P,
# with the context network).
def attach_readings(tasks):
    inventory_readings = inventory()
    counts = {'utterances': 0, 'tokens': 0, 'overrides': 0}
    for task in tasks:
        utterance = task[0]
        if utterance.get('labelFormat') != 'mfa':
            continue
        overrides, tokens = utterance_readings(utterance['text'], utterance['lab'], inventory_readings)
        utterance['readings'] = overrides
        utterance['alignedReadings'] = tokens
        counts['utterances'] += 1
        counts['tokens'] += len(tokens)
        counts['overrides'] += len(overrides)
    return counts


# Count a built utterance's heteronym tokens against the aligner (its utterance's alignedReadings) - totals {tokens,
# disagree}
def check_built(utterance, tokens, totals):
    from .frontend import BOUNDARIES  # pylint: disable=import-outside-toplevel
    aligned = utterance.get('alignedReadings')
    if aligned:
        # (built once - setdefault would evaluate inventory(), about 1.7 s, on every call)
        if 'inventory' not in _INVENTORY_CACHE:
            _INVENTORY_CACHE['inventory'] = inventory()
        count_disagreements(tokens, aligned, _INVENTORY_CACHE['inventory'], BOUNDARIES, totals)


_INVENTORY_CACHE = {}


# The check command - a build's included LibriTTS-R heteronym tokens against the aligner (alignment.jsonl's tokens and
# readings, re-aligned); exits 1 above MAX_DISAGREE
def check_command(args):
    from .frontend import BOUNDARIES  # pylint: disable=import-outside-toplevel
    with open(os.path.join(args.out, 'index.json'), encoding='utf-8') as file:
        index = json.load(file)
    included = {entry['id']: entry for entry in index['utterances'] if entry['include'] and entry['speaker'].startswith('libritts')}
    readings = inventory()
    totals = {'tokens': 0, 'disagree': 0, 'overrides': 0, 'szOnly': 0}
    with open(os.path.join(args.out, 'alignment.jsonl'), encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            entry = included.get(record['id'])
            if entry is None:
                continue
            totals['overrides'] += len(record.get('readings', {}))
            sentence = entry['sentence']
            speaker, chapter = sentence.split('_')[:2]
            lab = next((path for path in (os.path.join(data.LABEL_DIR, subset, speaker, chapter, sentence + '.lab')
                                          for subset in data.SUBSETS) if os.path.exists(path)), None)
            if lab is None:
                continue
            _, aligned = utterance_readings(entry['text'], lab, readings, lambda word: [])
            count_disagreements(record['tokens'].split(), aligned, readings, BOUNDARIES, totals)
    share = totals['disagree'] / max(1, totals['tokens'])
    failed = share > MAX_DISAGREE
    print(f'readings check {args.out}: {totals["tokens"]} heteronym tokens, {totals["disagree"]} ({100 * share:.2f}%) disagree with '
          f'the aligner, {totals["overrides"]} overrides; {totals["szOnly"]} tokens differ from the aligner only by S/Z (the front '
          f'end\'s reading kept){" - FAILED (over 1%)" if failed else ""}')
    return 1 if failed else 0


# The command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m picovox_train.readings', description='Heteronym readings from the aligned audio')
    commands = parser.add_subparsers(dest='command', required=True)
    check_parser = commands.add_parser('check', help='a build\'s heteronym tokens against the aligner (exit 1 over 1%%)')
    check_parser.add_argument('--out', required=True, help='the picovox_train.multi build directory')
    check_parser.set_defaults(func=check_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
