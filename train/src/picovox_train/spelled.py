# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The spelled head's labels - whether a word written in capitals is spelled (its letters' names - "AI", "TOC", "FBI") or
read as a word ("NOT", "NASA"). The front end keeps a word's capitals (lib/lexicon.js textToPhrases) and asks the G2P's
spelled head - a tanh layer and two logits on the encoder's final states (context.py, lib/g2p.js) - of each word in
capitals; a spelled word is its letters' names, each pronounced by the network. No word list at runtime: the network
learns which letter strings are words. The recipe is train/data/train.json's "context" "g2p" "spelled" settings.

The head's two logits are a word's two places: beside a word in capitals that is read (a heading, a shout - "LEAVE US
ALONE"), and alone among lowercase words ("the US government"). The labels below are the first's - a common word is
read. The second's are the same, but for the words of train/data/g2p-capitals.json ("alone" - common words that, alone
in capitals, are acronyms: "US", "AD", "UN" - by Wikipedia's articles against the books): spelled. The file's "spelled"
words - Wiktionary's words in capitals pronounced as their letters ("UCLA") - are spelled in both places, but a common
word of two or three letters ("PIE").

The labels, of the letter strings of two letters or more (g2p.py's data - the dictionary and the teacher-labeled words -
and the 128 books' word counts, train/data/g2p-words.json's: not the listeners' texts' - their counts are of the
lowercased words, and Wikipedia writes its acronyms in capitals, so by its counts "AI" and "GPS" are common words):

- a word of two or three letters - every string of them is labeled, so the head learns the table: read when it is a
  common word (its count in the books at least "shortCounts" - "NOT", "US"), unless the dictionary's first pronunciation
  spells it ("TV"); every other string spelled ("TOC", "AI" - a rare dictionary word in capitals is far more often an
  acronym)
- a longer word - spelled when the dictionary's first pronunciation spells it ("HTML"), or another of its pronunciations
  does and it isn't a common word ("NOAA"); every other known word read ("NASA", "STOP"). The monitoring set's words
  (g2p.py - held out of the G2P's training) are held out here too: the share of them the head finds spelled is its
  measure on words it hasn't seen
- "synthetic" strings of four or five consonants, spelled ("HTTP" - a longer acronym the dictionary lacks)

The head's batches draw each group by BATCH_SHARES.
"""

import collections
import itertools
import os
import random
import re

import numpy

from . import ROOT, g2p, read_json
from .frontend import LETTER_WORDS


CAPITALS_PATH = os.path.join(ROOT, 'train', 'data', 'g2p-capitals.json')


# The letters, and the consonants of the synthetic strings
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
CONSONANTS = 'bcdfghjklmnpqrstvwxz'
LETTERS_REGEX = re.compile(r'^[a-z]{2,}$')
STRESS_REGEX = re.compile(r'\d')

# The longest "short" word - every string up to it is labeled
SHORT = 3

# The synthetic strings' seed
SYNTHETIC_SEED = 20261110

# A batch's share of each group: the spelled short strings (nearly the whole table) and the short words; the
# dictionary's spelled words, short and long (few - so the head keeps them); the synthetic strings; and the long words,
# by their frequency (count ** "alpha") and uniformly
BATCH_SHARES = {'shortSpelled': 0.3125, 'shortDictionary': 0.0625, 'longSpelled': 0.0625, 'synthetic': 0.0625,
                'shortWords': 0.1875, 'longWordsFrequent': 0.1875, 'longWords': 0.125}


# A pronunciation without its stress
def unstressed(phones):
    return [STRESS_REGEX.sub('', phone) for phone in phones]


# The letters' names (without stress) - each letter's word's target (the front end's LETTER_WORDS: "a" is "ay")
def letter_names(first):
    return {letter: unstressed(first[LETTER_WORDS.get(letter, letter)]) for letter in ALPHABET}


# A word's spelling - its letters' names
def spelling(word, names):
    return [phone for letter in word for phone in names[letter]]


def spelled_labels(data, settings):
    """The spelled head's labels of g2p.py's data - {labels ({word: spelled} - beside a word in capitals), alone (the
    words read there but spelled alone - train/data/g2p-capitals.json's), heldOut (the monitoring set's long words - read,
    not trained), counts ({word: its count in the books}), groups ({group: words} - the batches'), alpha (the
    settings')}"""
    names = letter_names(data['first'])
    words_file = read_json(g2p.WORDS_PATH)
    counts = collections.Counter(words_file['weightCounts'])
    counts.update(words_file['evalCounts'])
    short_counts = {int(length): int(count) for length, count in settings['shortCounts'].items()}
    common = short_counts[SHORT]

    # The dictionary's spelled words - by their first pronunciation, or another
    first_spelled = set()
    other_spelled = set()
    for word, variants in data['all'].items():
        if LETTERS_REGEX.match(word):
            word_spelling = spelling(word, names)
            if unstressed(data['first'][word]) == word_spelling:
                first_spelled.add(word)
            elif any(unstressed(variant) == word_spelling for variant in variants):
                other_spelled.add(word)

    # The short strings - every one
    labels = {}
    for length in range(2, SHORT + 1):
        for letters in itertools.product(ALPHABET, repeat=length):
            word = ''.join(letters)
            labels[word] = word in first_spelled or counts[word] < short_counts[length]

    # The long words, and the synthetic strings
    known = set(data['first']) | set(data['labels']) | set(counts)
    monitor = set(data['monitor'])
    held_out = []
    for word in sorted(known):
        if LETTERS_REGEX.match(word) and SHORT < len(word) <= g2p.MAX_WORD:
            spelled = word in first_spelled or (word in other_spelled and counts[word] < common)
            if word in monitor and not spelled:
                held_out.append(word)
            else:
                labels[word] = spelled
    rng = random.Random(SYNTHETIC_SEED)
    synthetic = set()
    while len(synthetic) < int(settings['synthetic']):
        word = ''.join(rng.choice(CONSONANTS) for _ in range(rng.choice((SHORT + 1, SHORT + 1, SHORT + 2))))
        if word not in known:
            synthetic.add(word)
    labels.update((word, True) for word in synthetic)

    # The words Wiktionary spells - but a common short word
    capitals = read_json(CAPITALS_PATH)
    labels.update((word, True) for word in capitals['spelled'] if len(word) > SHORT or labels[word])

    # The batches' groups
    dictionary_spelled = first_spelled | other_spelled
    groups = collections.defaultdict(list)
    for word, spelled in sorted(labels.items()):
        short = len(word) <= SHORT
        if spelled and word in synthetic:
            groups['synthetic'].append(word)
        elif spelled:
            groups['shortSpelled' if short else 'longSpelled'].append(word)
            if short and word in dictionary_spelled:
                groups['shortDictionary'].append(word)
        else:
            groups['shortWords' if short else 'longWords'].append(word)
    alone = sorted(word for word in capitals['alone'] if labels.get(word) is False)
    return {'labels': labels, 'alone': alone, 'heldOut': held_out, 'counts': dict(counts), 'groups': dict(groups),
            'alpha': float(settings['alpha'])}


# The batches' sampler of the labels - sample(rng, size) gives (words, their labels - each [alone, beside a word in
# capitals] as 1.0 and 0.0): each group's share of the batch (BATCH_SHARES), drawn uniformly - the frequent long words in
# proportion to count ** alpha
def batch_sampler(spelled):
    groups, labels, counts, alone = spelled['groups'], spelled['labels'], spelled['counts'], set(spelled['alone'])
    weights = numpy.array([counts.get(word, 0) ** spelled['alpha'] for word in groups['longWords']])
    probabilities = {'longWordsFrequent': weights / weights.sum()}

    def sample(rng, size):
        words = []
        for name, share in BATCH_SHARES.items():
            group = groups['longWords' if name == 'longWordsFrequent' else name]
            chosen = rng.choice(len(group), size=round(share * size), p=probabilities.get(name))
            words.extend(group[index] for index in chosen)
        return words, [[1.0 if labels[word] or word in alone else 0.0, 1.0 if labels[word] else 0.0] for word in words]
    return sample


def spelled_errors(spelled, decisions):
    """The head's errors - decisions(words) gives whether each is spelled, [alone, beside a word in capitals]:
    {spelledMissed (the spelled strings found read), wordsSpelled (the words found spelled, the most frequent first),
    heldOutSpelled (the held-out words found spelled) - beside a word in capitals; aloneMissed (the words spelled alone
    found read there), aloneDiffers (the other strings found another way alone); words, spelled, heldOut (their
    counts)}"""
    labels, counts, alone = spelled['labels'], spelled['counts'], set(spelled['alone'])
    words = sorted(labels)
    found = dict(zip(words, decisions(words)))
    held_out = spelled['heldOut']
    held_out_found = decisions(held_out) if held_out else []
    return {
        'spelledMissed': [word for word in words if labels[word] and not found[word][1]],
        'wordsSpelled': sorted((word for word in words if not labels[word] and found[word][1]),
                               key=lambda word: (-counts.get(word, 0), word)),
        'heldOutSpelled': [word for word, (_, beside) in zip(held_out, held_out_found) if beside],
        'aloneMissed': [word for word in sorted(alone) if not found[word][0]],
        'aloneDiffers': [word for word in words if word not in alone and found[word][0] != labels[word]],
        'words': sum(1 for word in words if not labels[word]), 'spelled': sum(1 for word in words if labels[word]),
        'heldOut': len(held_out)
    }
