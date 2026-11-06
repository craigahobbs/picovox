# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
A clip's token durations from its CTC alignment (ctc.ctc_labels: our phone tokens in order, each with its times, and a
pause before and after them) - data.py's alignment: the leading pause goes to <bos>, each phone takes its label's time,
the boundary tokens between phones none, and the trailing pause goes to the last phrase end; then silences the
alignment gave to the phones next to a boundary token move into it (refine_pauses), and the durations are quantized to
frames (durations_to_frames).
"""

from .frontend import BOUNDARIES


# The comparison key of one of our phone tokens - its ARPAbet base, with the reduced AH (AH0) as AX
def token_key(token):
    base = token.rstrip('012')
    if base == 'AH' and token.endswith('0'):
        return 'AX'
    return base


# Each phone key's mean label duration over labels [(label, start, end)] - {key: seconds}
def mean_durations(labels):
    totals = {}
    for label, start, end in labels:
        if label != 'pau':
            total, count = totals.get(token_key(label), (0.0, 0))
            totals[token_key(label)] = (total + (end - start), count + 1)
    return {key: total / count for key, (total, count) in totals.items()}


# Token durations in seconds (from the audio's start) from a clip's labels [(label, start, end)] - mean_seconds, each
# phone key's mean duration (mean_durations): a phone's time is multiplied and divided by it, the float rounding the
# training data has always had (a phone's boundary falls on a half frame, where an ulp's difference moves a frame)
def label_durations(tokens, labels, mean_seconds):
    seconds = [0.0] * len(tokens)
    if labels[0][0] == 'pau':
        seconds[0] = labels[0][2] - labels[0][1]
    if labels[-1][0] == 'pau':
        seconds[-1] = labels[-1][2] - labels[-1][1]
    phones = iter(label for label in labels if label[0] != 'pau')
    for index, token in enumerate(tokens):
        if token not in BOUNDARIES:
            _, start, end = next(phones)
            weight = mean_seconds[token_key(token)]
            seconds[index] = (end - start) * weight / weight
    return seconds


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
# in seconds; returns the refined seconds
def refine_pauses(tokens, seconds, silent_runs):
    times = [0.0]
    for duration in seconds:
        times.append(times[-1] + duration)
    count = len(tokens)

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
            times[index], times[index + 1] = new_start, new_end
    return [times[index + 1] - times[index] for index in range(count)]
