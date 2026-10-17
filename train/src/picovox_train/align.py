# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Reconcile a corpus's phone labels (times) with our front end's tokens for the same text, giving each token a
duration - data.py's alignment.

Our tokens come from the text (frontend.py - our lexicon's pronunciations); the labels are the corpus's own
(ARCTIC: Festival's lexicon and postlexical rules, aligned by EHMM; LibriTTS: the Montreal Forced Aligner's
dictionary). They mostly agree; where they don't, a weighted edit-distance alignment pairs them:

- a phone paired with a phone (the same or a substitute) takes the label's duration
- a pause pairs with a boundary token (_ or a phrase end) at no cost; a boundary with no pause gets duration 0
- an extra label phone (not in our pronunciation), or our phone missing from the labels, joins the nearest paired
  phone in the same word span (preferring the previous one) into a block, and the block's time is split among our
  phones in proportion to their typical (mean) durations
- the leading silence goes to <bos>, the trailing silence to the last phrase end
"""

import functools

from .frontend import BOS, BOUNDARIES, VOWELS


# Alignment costs - a substitution of a similar phone costs less than a deletion plus an insertion
COST_NEAR = 0.2           # AH and the reduced AH (AX) - stress or reduction differences of the same vowel
COST_VOWEL = 0.6          # a vowel for another vowel
COST_CONSONANT = 0.7      # a consonant for another consonant
COST_CROSS = 1.3          # a vowel for a consonant
COST_INDEL = 1.0          # a phone in only one of the sequences
COST_PAUSE_INSIDE = 1.5   # a pause that falls inside a word (not at a boundary token)
INFINITY = float('inf')

# Voicing pairs and other close consonants - a cheaper substitution
CLOSE_CONSONANTS = set(frozenset(pair) for pair in [
    ('P', 'B'), ('T', 'D'), ('K', 'G'), ('F', 'V'), ('TH', 'DH'), ('S', 'Z'), ('SH', 'ZH'), ('CH', 'JH'),
    ('M', 'N'), ('N', 'NG'), ('T', 'DH'), ('D', 'DH'), ('S', 'SH'), ('Z', 'ZH')
])
COST_CLOSE = 0.4

PAUSE = 'PAU'


# The comparison key of one of our phone tokens - its ARPAbet base, with the reduced AH (AH0) as AX
def token_key(token):
    base = token.rstrip('012')
    if base == 'AH' and token.endswith('0'):
        return 'AX'
    return base


# The comparison key of a label phone - ARCTIC/Festival lower-case (ax the reduced AH, pau silence) or MFA ARPAbet
# with stress (AH0 the reduced AH, sil/sp/empty silence, spn spoken noise); None for an unknown label
def label_key(label):
    label = label.strip()
    if label in ('', 'pau', 'sil', 'sp', 'ssil', 'h#'):
        return PAUSE
    if label == 'spn':
        return 'SPN'
    upper = label.upper()
    if upper in ('AX', 'AH0'):
        return 'AX'
    return upper.rstrip('012')


# Is a comparison key a vowel?
def is_vowel_key(key):
    return key == 'AX' or key in VOWELS


# The substitution cost of our phone key for a label phone key (memoized - a few thousand pairs, looked up millions of times)
@functools.cache
def substitution_cost(ours, label):
    if ours == label:
        return 0.0
    if label == 'SPN':
        return COST_CONSONANT
    if {ours, label} == {'AH', 'AX'}:
        return COST_NEAR
    our_vowel, label_vowel = is_vowel_key(ours), is_vowel_key(label)
    if our_vowel and label_vowel:
        return COST_VOWEL
    if not our_vowel and not label_vowel:
        return COST_CLOSE if frozenset((ours, label)) in CLOSE_CONSONANTS else COST_CONSONANT
    return COST_CROSS


# Align our token names with label keys (pauses merged) - the path [(token index or None, label index or None)]
def align_sequences(tokens, labels):
    count_tokens, count_labels = len(tokens), len(labels)
    is_boundary = [token in BOUNDARIES for token in tokens]
    keys = [None if boundary else token_key(token) for token, boundary in zip(tokens, is_boundary)]
    cost = [[INFINITY] * (count_labels + 1) for _ in range(count_tokens + 1)]
    back = [[None] * (count_labels + 1) for _ in range(count_tokens + 1)]
    cost[0][0] = 0.0
    for i_token in range(count_tokens + 1):
        for i_label in range(count_labels + 1):
            # (every cell is reachable - a token or a label alone is always a finite step)
            here = cost[i_token][i_label]
            # Pair the token and the label
            if i_token < count_tokens and i_label < count_labels:
                label = labels[i_label]
                if is_boundary[i_token]:
                    step = 0.0 if label == PAUSE else INFINITY
                else:
                    step = INFINITY if label == PAUSE else substitution_cost(keys[i_token], label)
                if here + step < cost[i_token + 1][i_label + 1]:
                    cost[i_token + 1][i_label + 1] = here + step
                    back[i_token + 1][i_label + 1] = (i_token, i_label)
            # The token alone (a boundary without a pause is free)
            if i_token < count_tokens:
                step = 0.0 if is_boundary[i_token] else COST_INDEL
                if here + step < cost[i_token + 1][i_label]:
                    cost[i_token + 1][i_label] = here + step
                    back[i_token + 1][i_label] = (i_token, None)
            # The label alone (a pause inside a word costs more)
            if i_label < count_labels:
                step = COST_PAUSE_INSIDE if labels[i_label] == PAUSE else COST_INDEL
                if here + step < cost[i_token][i_label + 1]:
                    cost[i_token][i_label + 1] = here + step
                    back[i_token][i_label + 1] = (None, i_label)
    path = []
    i_token, i_label = count_tokens, count_labels
    while (i_token, i_label) != (0, 0):
        previous_token, previous_label = back[i_token][i_label]
        path.append((previous_token if previous_token is not None else None,
                     previous_label if previous_label is not None else None))
        i_token = previous_token if previous_token is not None else i_token
        i_label = previous_label if previous_label is not None else i_label
    path.reverse()
    return path, cost[count_tokens][count_labels]


# Merge adjacent pauses - [(key, seconds)] from [(label, seconds)]
def label_segments(labels):
    segments = []
    for label, seconds in labels:
        key = label_key(label)
        if key == PAUSE and segments and segments[-1][0] == PAUSE:
            segments[-1] = (PAUSE, segments[-1][1] + seconds)
        else:
            segments.append((key, seconds))
    return segments


# Assign each token a duration in seconds from the labels [(label, seconds)] - returns (seconds per token, report);
# mean_seconds maps a token key to its typical duration (for splitting blocks)
def assign_durations(tokens, labels, mean_seconds):
    segments = label_segments(labels)
    durations = [0.0] * len(tokens)

    # The leading and trailing silences
    lead = segments.pop(0)[1] if segments and segments[0][0] == PAUSE else 0.0
    trail = segments.pop()[1] if segments and segments[-1][0] == PAUSE else 0.0
    if tokens[0] != BOS or tokens[-1] not in BOUNDARIES or tokens[-1] == BOS:
        raise ValueError('tokens must start with <bos> and end with a phrase end')
    durations[0] = lead
    durations[-1] = trail

    # Align the middle
    middle = tokens[1:-1]
    keys = [key for key, _ in segments]
    path, cost = align_sequences(middle, keys)

    # Count the alignment's operations
    report = {'phones': 0, 'labelPhones': 0, 'exact': 0, 'near': 0, 'substituted': 0, 'deleted': 0, 'inserted': 0,
              'pauseAtBoundary': 0, 'pauseInsideWord': 0, 'cost': round(cost, 3), 'edits': [],
              'status': [None] * len(tokens)}
    status = report['status']
    last_token = 0
    for i_token, i_label in path:
        token = middle[i_token] if i_token is not None else None
        if i_token is not None:
            last_token = 1 + i_token
        label = keys[i_label] if i_label is not None else None
        if token is not None and token in BOUNDARIES:
            if label is not None:
                report['pauseAtBoundary'] += 1
            continue
        if label == PAUSE:
            report['pauseInsideWord'] += 1
            continue
        if token is not None:
            report['phones'] += 1
        if label is not None:
            report['labelPhones'] += 1
        if token is not None and label is not None:
            ours = token_key(token)
            if ours == label:
                report['exact'] += 1
                status[1 + i_token] = 'exact'
            elif {ours, label} == {'AH', 'AX'}:
                report['near'] += 1
                status[1 + i_token] = 'near'
            else:
                report['substituted'] += 1
                report['edits'].append(f'{ours}>{label}')
                status[1 + i_token] = 'substituted'
        elif token is not None:
            report['deleted'] += 1
            report['edits'].append(f'{token_key(token)}>-')
            status[1 + i_token] = 'deleted'
        else:
            report['inserted'] += 1
            report['edits'].append(f'->{label}')
            # An inserted label phone counts against the token before it (the first phone if none)
            index = last_token if last_token > 0 else 1
            status[index] = (status[index] or '') + '+inserted'

    # Blocks - each phone pair is an anchor; the unpaired phones and labels join the nearest anchor in the same span
    # between boundary tokens (the previous one preferred)
    anchors = []          # [{'tokens': [middle index], 'seconds': total}]
    span_of_anchor = []   # the span index of each anchor
    unattached = []       # [(path position, span, token index or None, seconds)]
    span = 0
    for i_token, i_label in path:
        token = middle[i_token] if i_token is not None else None
        seconds = segments[i_label][1] if i_label is not None else 0.0
        if token is not None and token in BOUNDARIES:
            durations[1 + i_token] += seconds
            span += 1
        elif token is not None and i_label is not None:
            anchors.append({'tokens': [i_token], 'seconds': seconds})
            span_of_anchor.append(span)
        else:
            unattached.append((len(anchors), span, i_token, seconds))
    if not anchors:
        raise ValueError('no paired phones')
    for anchor_count, item_span, i_token, seconds in unattached:
        previous, following = anchor_count - 1, anchor_count
        if previous >= 0 and span_of_anchor[previous] == item_span:
            target = previous
        elif following < len(anchors) and span_of_anchor[following] == item_span:
            target = following
        else:
            target = previous if previous >= 0 else following
        if i_token is not None:
            anchors[target]['tokens'].append(i_token)
        anchors[target]['seconds'] += seconds
    for anchor in anchors:
        members = sorted(anchor['tokens'])
        weights = [mean_seconds.get(token_key(middle[index]), 0.08) for index in members]
        total_weight = sum(weights)
        for index, weight in zip(members, weights):
            durations[1 + index] += anchor['seconds'] * weight / total_weight
    return durations, report


# Quantize token durations (seconds) to frames so the boundaries round and the total is preserved; phone tokens get
# at least one frame (taken from the longest neighbor phone)
def durations_to_frames(seconds, frame_seconds, tokens):
    frames = []
    elapsed = 0.0
    previous_boundary = 0
    for duration in seconds:
        elapsed += duration
        boundary = int(round(elapsed / frame_seconds))
        frames.append(boundary - previous_boundary)
        previous_boundary = boundary
    for index, token in enumerate(tokens):
        if token in BOUNDARIES or frames[index] > 0:
            continue
        neighbors = [neighbor for neighbor in (index - 1, index + 1)
                     if 0 <= neighbor < len(tokens) and tokens[neighbor] not in BOUNDARIES and frames[neighbor] > 1]
        if not neighbors:
            neighbors = [neighbor for neighbor in range(len(tokens)) if tokens[neighbor] not in BOUNDARIES and frames[neighbor] > 1]
        if neighbors:
            donor = max(neighbors, key=lambda neighbor: frames[neighbor])
            frames[donor] -= 1
            frames[index] += 1
    return frames


# Pause refinement - the minimum silence (seconds) taken as a pause at a phrase end and at a plain word boundary
# (closures of stops on both sides of a word boundary make long silences, so a plain boundary needs a longer one),
# and the least each neighboring phone keeps (a stop keeps its closure and burst)
MIN_PAUSE_PHRASE = 0.08
MIN_PAUSE_WORD = 0.15
KEEP_STOP = 0.05
KEEP_OTHER = 0.03
STOPS = set(['P', 'B', 'T', 'D', 'K', 'G', 'CH', 'JH'])


# Move silence that the labels gave to the phones next to a boundary token into the boundary token (aligners often fold
# pauses into neighboring phones) - seconds per token (starting at the audio's start), the silent runs [(start, end)]
# in seconds; returns (refined seconds, count of pauses changed, seconds moved)
def refine_pauses(tokens, seconds, silent_runs):
    times = [0.0]
    for duration in seconds:
        times.append(times[-1] + duration)
    count = len(tokens)
    changed, moved = 0, 0.0

    # Keep - the least a phone token keeps
    def keep(index):
        duration = times[index + 1] - times[index]
        return min(duration, KEEP_STOP if token_key(tokens[index]) in STOPS else KEEP_OTHER)

    for index, token in enumerate(tokens):
        if token not in BOUNDARIES:
            continue
        start, end = times[index], times[index + 1]
        minimum = MIN_PAUSE_WORD if token == '_' else MIN_PAUSE_PHRASE
        candidates = [(run_start, run_end) for run_start, run_end in silent_runs
                      if run_end - run_start >= minimum and run_end >= start - 0.02 and run_start <= end + 0.02]
        if not candidates:
            continue
        run_start, run_end = max(candidates, key=lambda run: run[1] - run[0])
        new_start = start if index == 0 else min(start, max(run_start, times[index - 1] + keep(index - 1)))
        new_end = end if index == count - 1 else max(end, min(run_end, times[index + 2] - keep(index + 1)))
        if new_start < start - 1e-6 or new_end > end + 1e-6:
            moved += (start - new_start) + (new_end - end)
            changed += 1
            times[index], times[index + 1] = new_start, new_end
    return [times[index + 1] - times[index] for index in range(count)], changed, moved
