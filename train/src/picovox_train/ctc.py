# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Our speech measurement - a wav2vec2 phoneme recognizer's CTC posteriors (facebook/wav2vec2-lv-60-espeak-cv-ft, espeak
IPA, 20 ms frames), for the training speech's alignment and checks (tts.py) and the readings of its heteronym tokens
(readings.py, g2paudit.py):

- Alignment - our tokens (frontend.py) force-aligned to the posteriors: each of our phones is a position whose
  posterior is the sum over the IPA tokens it may be heard as (IPA_TOKENS - reductions, flaps, devoicing, the
  recognizer's r-colored and syllabic merged tokens); a Viterbi pass over the CTC topology gives each phone a run of
  frames; the blank frames between two runs are split at the midpoint; the first and last phones extend to the speech's
  energy onset and offset; then align.py turns the labels into token durations and moves silences into boundary tokens.
  Checked against CMU ARCTIC's EHMM labels (300 utterances of slt and bdl - the former arctic-check, in git history):
  phone onsets within 20 ms 68%, within 40 ms 93%, mean absolute error 18 ms, bias 0.5 ms.
- Checks - the recognizer's greedy phone string against the sentence: the phone error rate (a garbled or hallucinated
  clip) and the missing tail (a clip cut short).
- The candidate vote - the CTC likelihood of each candidate pronunciation of a word, with its neighbors' phones, on the
  word's segment (each phone a strict class of recognizer tokens - STRICT_TOKENS; S and Z each only their own where the
  candidates differ by them); a vote when one is VOTE_MARGIN more likely than every other.
"""

import json
import math
import os
import re

from . import ROOT
from .wav import read_wav


# (the recognizer's download, with the listener models' - downloads/listeners/, kept by make clean)
HF_HOME = os.path.join(ROOT, 'downloads', 'listeners', 'hf')


# A phone string without stress
def unstressed(phones):
    return re.sub(r'\d', '', phones)


# The phoneme recognizer - wav2vec2 large (LibriLight 60k h pretraining) fine-tuned on CommonVoice to espeak IPA
# phonemes (Xu, Baevski, and Auli, 2021; Apache 2.0); its frames are 20 ms
CTC_REPO = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
CTC_SAMPLE_RATE = 16000
CTC_FRAME_SECONDS = 0.02

# Our ARPAbet phones (comparison keys, align.py - AX the reduced AH0) to the recognizer's IPA tokens: the usual en-us
# espeak rendering first, then accepted variants (reductions, flaps and glottal stops, devoicing, and the recognizer's
# r-colored and syllabic merged tokens, which both phones of the pair may claim); a phone's posterior is the sum over
# its tokens
IPA_TOKENS = {
    'AA': ['ɑː', 'ɑ', 'ɑːɹ', 'a', 'aː', 'ɔ', 'ɔː', 'ɒ'],
    'AE': ['æ', 'a', 'ɛ', 'æː', 'eə'],
    'AH': ['ʌ', 'ə', 'ɐ', 'ɑ', 'ɜ'],
    'AX': ['ə', 'ɐ', 'ɪ', 'ᵻ', 'ʌ', 'əl', 'ɚ', 'ɛ'],
    'AO': ['ɔː', 'ɔ', 'ɑː', 'ɑ', 'ɒ', 'oː', 'o', 'ɔːɹ', 'oːɹ'],
    'AW': ['aʊ', 'a', 'ʊ', 'au'],
    'AY': ['aɪ', 'a', 'aɪɚ', 'aɪə', 'ai', 'ɑ', 'ʌ'],
    'EH': ['ɛ', 'e', 'ɛɹ', 'eː', 'æ', 'ɪ', 'eə'],
    'ER': ['ɜː', 'ɚ', 'ɜ', 'ə', 'ɐ', 'ɹ', 'ər'],
    'EY': ['eɪ', 'e', 'eː', 'ɛ', 'ɛɪ'],
    'IH': ['ɪ', 'ᵻ', 'i', 'ə', 'ɪɹ', 'iə', 'ɛ'],
    'IY': ['iː', 'i', 'ɪ', 'iə', 'ɪɹ'],
    'OW': ['oʊ', 'o', 'oː', 'əʊ', 'ɔ', 'oːɹ', 'ʊ'],
    'OY': ['ɔɪ', 'oɪ', 'ɔ'],
    'UH': ['ʊ', 'u', 'ʊɹ', 'ə', 'uː'],
    'UW': ['uː', 'u', 'ʊ', 'ju', 'ʉ'],
    'B': ['b', 'β', 'p'],
    'CH': ['tʃ', 'ʃ', 't'],
    'D': ['d', 'ɾ', 't'],
    'DH': ['ð', 'd', 'θ', 'z'],
    'F': ['f', 'v'],
    'G': ['ɡ', 'k'],
    'HH': ['h'],
    'JH': ['dʒ', 'ʒ', 'd'],
    'K': ['k', 'kh', 'kʰ', 'ɡ'],
    'L': ['l', 'ɫ', 'əl', 'l̩'],
    'M': ['m'],
    'N': ['n', 'n̩', 'ŋ', 'm'],
    'NG': ['ŋ', 'n'],
    'P': ['p', 'ph', 'pʰ', 'b'],
    'R': ['ɹ', 'r', 'ɚ', 'ɑːɹ', 'ɔːɹ', 'oːɹ', 'ɛɹ', 'ɪɹ', 'ʊɹ'],
    'S': ['s', 'z'],
    'SH': ['ʃ'],
    'T': ['t', 'ɾ', 'ʔ', 'th', 'tʰ', 'd'],
    'TH': ['θ', 'ð', 't'],
    'V': ['v', 'f'],
    'W': ['w'],
    'Y': ['j', 'i'],
    'Z': ['z', 's'],
    'ZH': ['ʒ', 'ʃ']
}

# Alignment settings (chosen against the ARCTIC labels by the former arctic-check - see git history): the CTC frames'
# time offset, how the blank frames between two phones' runs are split ('mid' - at the midpoint, the best of the
# splits tried), and how far the first phone's start and the last phone's end may extend to the speech's energy onset
# and offset
CTC_SHIFT = -0.005
CTC_SPLIT = 'mid'
EDGE_EXTEND = 0.15


# Load the phoneme recognizer - (model, vocab {token: id}, device)
def load_ctc(device=None):
    os.environ.setdefault('HF_HOME', HF_HOME)
    import torch  # pylint: disable=import-outside-toplevel
    # (huggingface_hub and transformers are build/venv-train's)
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel,import-error
    from transformers import Wav2Vec2ForCTC  # pylint: disable=import-outside-toplevel,import-error
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    directory = snapshot_download(CTC_REPO, allow_patterns=['*.json', 'pytorch_model.bin', 'README.md'])
    with open(os.path.join(directory, 'vocab.json'), encoding='utf-8') as file:
        vocab = json.load(file)
    model = Wav2Vec2ForCTC.from_pretrained(directory).to(device).eval()
    return model, vocab, device


# The CTC log posteriors of 16 kHz clips (float32 arrays), batched - a list of [frames, vocab] arrays
def ctc_log_posteriors(model, device, samples_list, max_batch_seconds=80.0):
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    order = sorted(range(len(samples_list)), key=lambda index: len(samples_list[index]))
    results = [None] * len(samples_list)
    batch = []

    def run(indices):
        longest = max(len(samples_list[index]) for index in indices)
        inputs = np.zeros((len(indices), longest), dtype=np.float32)
        mask = np.zeros((len(indices), longest), dtype=np.int64)
        for row, index in enumerate(indices):
            samples = samples_list[index].astype(np.float32)
            inputs[row, :len(samples)] = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
            mask[row, :len(samples)] = 1
        with torch.no_grad():
            logits = model(torch.from_numpy(inputs).to(device), attention_mask=torch.from_numpy(mask).to(device)).logits
            log_posteriors = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            lengths = model._get_feat_extract_output_lengths(torch.tensor([len(samples_list[index]) for index in indices]))  # pylint: disable=protected-access
        for row, index in enumerate(indices):
            results[index] = log_posteriors[row, :int(lengths[row])]

    for index in order:
        if batch and (len(batch) + 1) * len(samples_list[index]) > max_batch_seconds * CTC_SAMPLE_RATE:
            run(batch)
            batch = []
        batch.append(index)
    if batch:
        run(batch)
    return results


# Viterbi CTC alignment of a sequence of distinct positions - log posteriors [frames, positions] and the blank's
# [frames]; returns each position's run of frames [(first, last)], or None if there are too few frames
def ctc_viterbi(position_scores, blank_scores):
    import numpy as np  # pylint: disable=import-outside-toplevel
    frames, count = position_scores.shape
    if frames < count:
        return None
    states = 2 * count + 1
    emit = np.empty((frames, states), dtype=np.float64)
    emit[:, 0::2] = blank_scores[:, None]
    emit[:, 1::2] = position_scores
    score = np.full(states, -np.inf)
    score[0], score[1] = emit[0, 0], emit[0, 1]
    back = np.zeros((frames, states), dtype=np.int8)
    for frame in range(1, frames):
        stay = score
        step = np.concatenate(([-np.inf], score[:-1]))
        skip = np.concatenate(([-np.inf, -np.inf], score[:-2]))
        skip[0::2] = -np.inf          # only a phone state may skip the blank before it
        best = np.maximum(np.maximum(stay, step), skip)
        back[frame] = np.where(best == stay, 0, np.where(best == step, 1, 2))
        score = best + emit[frame]
    state = states - 1 if score[states - 1] >= score[states - 2] else states - 2
    path = [0] * frames
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    runs = [None] * count
    for frame, state in enumerate(path):
        if state % 2 == 1:
            position = state // 2
            runs[position] = (runs[position][0] if runs[position] else frame, frame)
    return runs


# The phone tokens of a token sequence - [(token index, comparison key)]
def phone_positions(tokens):
    from . import align  # pylint: disable=import-outside-toplevel
    from .frontend import BOUNDARIES  # pylint: disable=import-outside-toplevel
    return [(index, align.token_key(token)) for index, token in enumerate(tokens) if token not in BOUNDARIES]


# Align our tokens with a clip's CTC log posteriors - (labels [(label, start, end)] in seconds with 'pau' for the
# edge silences, per-phone confidences, alignment score), or None; samples are 16 kHz
def ctc_labels(tokens, log_posteriors, vocab, samples):
    import numpy as np  # pylint: disable=import-outside-toplevel
    from .data import silent_runs  # pylint: disable=import-outside-toplevel
    positions = phone_positions(tokens)
    columns = []
    for _, key in positions:
        ids = [vocab[token] for token in IPA_TOKENS[key] if token in vocab]
        columns.append(np.logaddexp.reduce(log_posteriors[:, ids], axis=1))
    position_scores = np.stack(columns, axis=1)
    blank_scores = log_posteriors[:, vocab['<pad>']]
    runs = ctc_viterbi(position_scores, blank_scores)
    if runs is None:
        return None
    frames = len(log_posteriors)
    seconds = len(samples) / CTC_SAMPLE_RATE

    # Phone boundaries - each phone's run, the blank frames between two runs split at their midpoint (CTC_SPLIT)
    starts, ends = [], []
    for index, (first, _) in enumerate(runs):
        start = first if index == 0 else (runs[index - 1][1] + 1 + first) / 2
        starts.append(start)
        if index > 0:
            ends.append(start)
    ends.append(runs[-1][1] + 1)
    starts = [min(seconds, max(0.0, frame * CTC_FRAME_SECONDS + CTC_SHIFT)) for frame in starts]
    ends = [min(seconds, max(0.0, frame * CTC_FRAME_SECONDS + CTC_SHIFT)) for frame in ends]

    # The edges - extend the first and last phones to the speech's energy onset and offset (CTC runs are peaky)
    silences = silent_runs(samples.astype(np.float64), CTC_SAMPLE_RATE)
    onset = silences[0][1] if silences and silences[0][0] == 0 else 0.0
    trailing = [run for run in silences if run[1] >= seconds - 0.011]
    offset = trailing[-1][0] if trailing else seconds
    if onset < starts[0]:
        starts[0] = max(onset, starts[0] - EDGE_EXTEND)
    if offset > ends[-1]:
        ends[-1] = min(offset, ends[-1] + EDGE_EXTEND)
    for index in range(1, len(starts)):
        starts[index] = max(starts[index], starts[index - 1] + 0.005)
        ends[index - 1] = starts[index]
    ends[-1] = max(ends[-1], starts[-1] + 0.005)

    labels = []
    if starts[0] > 0:
        labels.append(('pau', 0.0, starts[0]))
    for (index, _), start, end in zip(positions, starts, ends):
        labels.append((tokens[index], start, end))
    if ends[-1] < seconds:
        labels.append(('pau', ends[-1], seconds))

    # Confidences - each phone's best posterior over its run; the path's mean log posterior per frame
    confidences = [float(np.exp(position_scores[first:last + 1, index].max())) for index, (first, last) in enumerate(runs)]
    path_score = float(sum(position_scores[first:last + 1, index].sum() for index, (first, last) in enumerate(runs)))
    run_frames = set(frame for first, last in runs for frame in range(first, last + 1))
    path_score += float(sum(blank_scores[frame] for frame in range(frames) if frame not in run_frames))
    return labels, confidences, path_score / frames


# The ARPAbet keys each recognizer token may stand for (IPA_TOKENS inverted)
TOKEN_KEYS = {}
for _key, _tokens in IPA_TOKENS.items():
    for _token in _tokens:
        TOKEN_KEYS.setdefault(_token, set()).add(_key)


# The recognizer's greedy string against a sentence's phones - (phone error rate, tail): the error rate is
# (substitutions + deletions + insertions) / phones, a recognized token matching a phone if it may stand for it; the
# tail is the count of the sentence's last phones missing after the last recognized token (a clip cut short)
def greedy_phone_error(greedy, tokens):
    expected = [key for _, key in phone_positions(tokens)]
    heard = greedy.split()
    count, heard_count = len(expected), len(heard)
    table = [list(range(heard_count + 1))]
    for row, key in enumerate(expected, 1):
        current = [row] + [0] * heard_count
        previous = table[-1]
        for column, token in enumerate(heard, 1):
            cost = 0 if key in TOKEN_KEYS.get(token, ()) else 1
            current[column] = min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + cost)
        table.append(current)
    matched = min(range(count + 1), key=lambda row: (table[row][heard_count] + (count - row), row))
    return table[count][heard_count] / max(1, count), count - matched


# Token durations (seconds, from the audio's start) of tokens from labels [(label, start, end)] - data.py's
# alignment and pause refinement; returns (seconds, report)
def label_durations(tokens, labels, samples, rate, mean_seconds):
    from . import align  # pylint: disable=import-outside-toplevel
    from .data import silent_runs  # pylint: disable=import-outside-toplevel
    seconds, report = align.assign_durations(tokens, [(label, end - start) for label, start, end in labels], mean_seconds)
    seconds, report['pausesRefined'], report['pauseSecondsMoved'] = align.refine_pauses(tokens, seconds, silent_runs(samples, rate))
    return seconds, report


# Read a 16 kHz mono clip (float64)
def read_16k(path):
    import numpy as np  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    samples, rate = read_wav(path)
    samples = samples.mean(axis=1)
    if rate != CTC_SAMPLE_RATE:
        samples = soxr.resample(samples, rate, CTC_SAMPLE_RATE, quality='VHQ')
    return np.ascontiguousarray(samples)


# The recognizer's greedy (best path) phoneme string, tokens separated by spaces
def greedy_decode(log_posteriors, vocab):
    names = {index: token for token, index in vocab.items()}
    best = log_posteriors.argmax(axis=1)
    tokens = []
    previous = None
    for index in best:
        if index != previous and names[int(index)] not in ('<pad>', '<s>', '</s>', '<unk>'):
            tokens.append(names[int(index)])
        previous = index
    return ' '.join(tokens)


# First-pass F0 of a clip (Praat, 60-700 Hz) - the voiced values (a worker task)
def first_pass_f0(path):
    import numpy as np  # pylint: disable=import-outside-toplevel
    import parselmouth  # pylint: disable=import-outside-toplevel, import-error
    sound = parselmouth.Sound(path)
    f0 = sound.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=700).selected_array['frequency']
    return f0[f0 > 0].astype(np.float32)


#
# The candidate vote
#

# The recognizer's IPA tokens each phone may be heard as (IPA_TOKENS, without the tokens of another vowel's
# quality - the candidates differ there); an unstressed vowel may be reduced (REDUCTIONS)
STRICT_TOKENS = {
    'AA': ['ɑː', 'ɑ', 'ɑːɹ', 'a', 'aː', 'ɒ'],
    'AE': ['æ', 'a', 'æː', 'eə'],
    'AH': ['ʌ', 'ə', 'ɐ'],
    'AO': ['ɔː', 'ɔ', 'ɒ', 'ɔːɹ', 'oːɹ', 'ɑː', 'ɑ'],
    'AW': ['aʊ', 'au'],
    'AY': ['aɪ', 'aɪɚ', 'aɪə', 'ai'],
    'EH': ['ɛ', 'e', 'ɛɹ', 'eː', 'eə'],
    'ER': ['ɜː', 'ɚ', 'ɜ'],
    'EY': ['eɪ', 'e', 'eː', 'ɛɪ'],
    'IH': ['ɪ', 'ᵻ', 'ɪɹ'],
    'IY': ['iː', 'i', 'iə', 'ɪɹ'],
    'OW': ['oʊ', 'o', 'oː', 'əʊ', 'oːɹ'],
    'OY': ['ɔɪ', 'oɪ'],
    'UH': ['ʊ', 'ʊɹ'],
    'UW': ['uː', 'u', 'ʉ'],
    'B': ['b', 'β'],
    'CH': ['tʃ'],
    'D': ['d', 'ɾ'],
    'DH': ['ð'],
    'F': ['f'],
    'G': ['ɡ'],
    'HH': ['h'],
    'JH': ['dʒ'],
    'K': ['k', 'kh', 'kʰ'],
    'L': ['l', 'ɫ', 'əl', 'l̩'],
    'M': ['m'],
    'N': ['n', 'n̩'],
    'NG': ['ŋ'],
    'P': ['p', 'ph', 'pʰ'],
    'R': ['ɹ', 'r', 'ɚ', 'ɑːɹ', 'ɔːɹ', 'oːɹ', 'ɛɹ', 'ɪɹ', 'ʊɹ'],
    'S': ['s', 'z'],
    'SH': ['ʃ'],
    'T': ['t', 'ɾ', 'ʔ', 'th', 'tʰ'],
    'TH': ['θ'],
    'V': ['v'],
    'W': ['w'],
    'Y': ['j'],
    'Z': ['z', 's'],
    'ZH': ['ʒ']
}
REDUCTIONS = ['ə', 'ɐ', 'ɪ', 'ᵻ']

# The strict tokens of S and Z where two candidates differ by them (STRICT_TOKENS's accept both - devoicing and
# voicing in context - which would make a voicing difference unmeasurable)
VOICING_TOKENS = {'S': ['s'], 'Z': ['z']}

# A token's vote - the candidate at least VOTE_MARGIN more likely (log) than every other
VOTE_MARGIN = 2.0


# The recognizer token ids of an aligned phone (stress digit and all) - voicing, its S or Z only
def phone_token_ids(phone, vocab, voicing=False):
    key = unstressed(phone)
    tokens = (VOICING_TOKENS[key] if voicing and key in VOICING_TOKENS else STRICT_TOKENS[key]) + (REDUCTIONS if phone[-1:] == '0' else [])
    return [vocab[token] for token in dict.fromkeys(tokens) if token in vocab]


# The CTC log likelihood of a phone sequence (each phone a class of recognizer tokens, never merged with its neighbor)
# given a segment's log posteriors [frames, vocab] - voicing, the positions whose S or Z is strict
def ctc_log_likelihood(log_posteriors, phones, vocab, voicing=()):
    import numpy  # pylint: disable=import-outside-toplevel
    frames = len(log_posteriors)
    count = len(phones)
    if frames < count:
        return -math.inf
    blank = log_posteriors[:, vocab['<pad>']]
    emit = numpy.empty((frames, 2 * count + 1))
    emit[:, 0::2] = blank[:, None]
    for index, phone in enumerate(phones):
        emit[:, 2 * index + 1] = numpy.logaddexp.reduce(log_posteriors[:, phone_token_ids(phone, vocab, index in voicing)], axis=1)
    score = numpy.full(2 * count + 1, -numpy.inf)
    score[0], score[1] = emit[0, 0], emit[0, 1]
    for frame in range(1, frames):
        step = numpy.concatenate(([-numpy.inf], score[:-1]))
        skip = numpy.concatenate(([-numpy.inf, -numpy.inf], score[:-2]))
        skip[0::2] = -numpy.inf
        score = numpy.logaddexp(numpy.logaddexp(score, step), skip) + emit[frame]
    return float(numpy.logaddexp(score[-1], score[-2]))


# The vowels with an r-colored token (STRICT_TOKENS) - ɑːɹ, ɔːɹ, oːɹ, ɛɹ, ɪɹ, ʊɹ
R_COLORED = {'AA', 'AO', 'OW', 'EH', 'IH', 'IY', 'UH'}


# A candidate's log likelihood in its context - the better of its phones and (when it has a vowel and R the recognizer
# may hear as one r-colored token, which a run of one frame can't split) the phones with those Rs dropped; voicing, the
# candidate's positions whose S or Z is strict
def candidate_log_likelihood(log_posteriors, before, phones, after, vocab, voicing=()):
    sequence = before + phones + after
    strict = {len(before) + position for position in voicing}
    kept = [index for index, phone in enumerate(sequence) if not (phone == 'R' and index and unstressed(sequence[index - 1]) in R_COLORED)]
    score = ctc_log_likelihood(log_posteriors, sequence, vocab, strict)
    if len(kept) == len(sequence):
        return score
    merged_r = [sequence[index] for index in kept]
    merged_strict = {position for position, index in enumerate(kept) if index in strict}
    return max(score, ctc_log_likelihood(log_posteriors, merged_r, vocab, merged_strict))


# A token's vote among candidate scores - the best's index when it's VOTE_MARGIN above the rest, else None
def vote(scores):
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    return order[0] if scores[order[0]] - scores[order[1]] >= VOTE_MARGIN else None


# A segment of a clip's audio at 16 kHz
def read_segment(path, start, end):
    import soxr  # pylint: disable=import-outside-toplevel
    samples, rate = read_wav(path)
    return soxr.resample(samples[int(start * rate):int(end * rate)].mean(axis=1), rate, CTC_SAMPLE_RATE, quality='VHQ')


# A pronunciation with the mergers' vowels merged (AA and AO; AE, EH, and EY before R; IH and IY before R - the
# recognizer's one ɪɹ token), the reduced vowels (AH0, IH0) one schwa, and without stress
def merged(phones):
    items = phones.split()
    bases = unstressed(phones).split()

    # (a vowel before R)
    def before_r(index):
        return index + 1 < len(bases) and bases[index + 1] == 'R'
    return ' '.join('A*' if phone in ('AA', 'AO') else
                    ('E*' if phone in ('AE', 'EH', 'EY') and before_r(index) else
                     ('I*' if phone in ('IH', 'IY') and before_r(index) else
                      ('@' if items[index] in ('AH0', 'IH0') else phone)))
                    for index, phone in enumerate(bases))
