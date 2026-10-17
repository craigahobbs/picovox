# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Emotional speech for the style modifiers - CREMA-D (Cao et al., 2014): 91 actors (48 male, 43 female, ages 20-74)
each reading 12 fixed sentences as angry, disgusted, fearful, happy, neutral, and sad (the sentence IEO also at low,
medium, and high intensity), 7,442 clips, 16 kHz WAV, with crowd-sourced ratings of every clip's emotion and intensity
(audio alone, video alone, and both - about 10 ratings each).

    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.cremad fetch            # data/corpora/crema_d/ (650 MB, about 7 minutes)
    PYTHONPATH=train/src build/venv-listeners/bin/python -B -m picovox_train.cremad align     # CTC phone alignments -> build/nn/cremad/alignments.jsonl (GPU, 2 min)
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.cremad build            # build/nn/data/cremad/ and data/nn/cremad.json (1 minute)
    PYTHONPATH=train/src build/venv-nn/bin/python -B -m picovox_train.cremad build --out DIR  # elsewhere, its summary (cremad.json) beside it

The align command needs transformers, so it runs in build/venv-listeners (no new packages); build needs Praat
(parselmouth), so it runs in build/venv-nn.

Alignment - CREMA-D has no phone labels. Our tokens (frontend.py) are force-aligned to the CTC posteriors of a
wav2vec2 phoneme recognizer (facebook/wav2vec2-lv-60-espeak-cv-ft, espeak IPA, 20 ms frames): each of our phones is a
position whose posterior is the sum over the IPA tokens it may be heard as (IPA_TOKENS - reductions, flaps, devoicing,
the recognizer's r-colored and syllabic merged tokens); a Viterbi pass over the CTC topology gives each phone a run of
frames; the blank frames between two runs are split at the midpoint; the first and last phones extend to the speech's
energy onset and offset; then align.py turns the labels into token durations and moves silences into boundary
tokens, exactly as for ARCTIC. Checked against ARCTIC's EHMM labels (300 utterances of slt and bdl - the former
arctic-check, in git history):
phone onsets within 20 ms 68%, within 40 ms 93%, mean absolute error 18 ms, bias 0.5 ms. The data's tokens are the
front end's at build time: a clip's alignment is reused when its phones without stress (all the alignment sees) are
the same, else build asks for align again.

Checks per clip (build) - the recognizer's greedy phone string against the sentence: a clip reading another of the 12
sentences is relabeled (two), a clip cut short (its final phones missing) or garbled (greedy phone error above 0.6)
is excluded, as is one with more than 0.5% of its samples clipped at full scale.

The data (build/nn/data/cremad/) - data.py's format (ArcticData(directory) loads it), one speaker per actor
('cremad-<actor>'), with per-utterance style fields: emotion and emotionId (index of index.json "emotions"), intensity
and intensityId ("intensities"), the crowd's audio-only voiceVote,
voiceAgreement, and voiceLevel, alignment quality (alignScore, phoneConfidence, heard, phoneError), and recording
measures (clipped, peakDb, levelDb, noiseFloorDb). The split is by actor and sentence (all six emotions together),
about 5% validation and 5% test. data/nn/cremad.json is the committed summary.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

from . import ROOT, SOURCE_DIR


CORPORA_DIR = os.path.join(ROOT, 'data', 'corpora')
CREMAD_DIR = os.path.join(CORPORA_DIR, 'crema_d')
WAV_DIR = os.path.join(CREMAD_DIR, 'AudioWAV')
USER_AGENT = 'voice-synth-corpora/1.0 (+https://github.com/craigahobbs/voice-synth; research)'

# The official repository (git LFS) - the media host serves the LFS objects by path
REPO = 'CheyneyComputerScience/CREMA-D'
REPO_BRANCH = 'master'
TREE_URL = f'https://api.github.com/repos/{REPO}/git/trees/{REPO_BRANCH}?recursive=1'
MEDIA_URL = f'https://media.githubusercontent.com/media/{REPO}/{REPO_BRANCH}/{{path}}'
RAW_URL = f'https://raw.githubusercontent.com/{REPO}/{REPO_BRANCH}/{{path}}'
METADATA_FILES = ['LICENSE.txt', 'README.md', 'SentenceFilenames.csv', 'VideoDemographics.csv',
                  'processedResults/summaryTable.csv', 'processedResults/tabulatedVotes.csv',
                  'finishedEmoResponses.csv', 'finishedResponses.csv']

LICENSE = ('Open Database License 1.0 (https://opendatacommons.org/licenses/odbl/1-0/) for the database; any rights '
           'in its individual contents under the Database Contents License 1.0 '
           '(https://opendatacommons.org/licenses/dbcl/1-0/) - the repository\'s LICENSE.txt. Cite Cao, Cooper, '
           'Keutmann, Gur, Nenkova, and Verma, "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", IEEE '
           'Transactions on Affective Computing 5(4), 2014. A model trained on it is a "Produced Work" under the ODbL '
           '(attribution notice; share-alike applies only to derived databases, such as build/nn/data/cremad/ if it '
           'were redistributed)')

# The sentences (the file name's code) and the emotions
SENTENCES = {
    'IEO': 'It\'s eleven o\'clock.',
    'TIE': 'That is exactly what happened.',
    'IOM': 'I\'m on my way to the meeting.',
    'IWW': 'I wonder what this is about.',
    'TAI': 'The airplane is almost full.',
    'MTI': 'Maybe tomorrow it will be cold.',
    'IWL': 'I would like a new alarm clock.',
    'ITH': 'I think I have a doctor\'s appointment.',
    'DFA': 'Don\'t forget a jacket.',
    'ITS': 'I think I\'ve seen this before.',
    'TSI': 'The surface is slick.',
    'WSI': 'We\'ll stop in a couple of minutes.'
}
EMOTIONS = {'ANG': 'angry', 'DIS': 'disgusted', 'FEA': 'fearful', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
INTENSITIES = {'LO': 'low', 'MD': 'medium', 'HI': 'high', 'XX': 'unspecified'}


# Print a progress line to stderr
def log(message):
    print(message, file=sys.stderr, flush=True)


# Download a URL to a path (atomically, via path + '.part'), unless the path exists; retries on transient errors
def download(url, path, attempts=5):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if content.startswith(b'version https://git-lfs'):
                raise IOError(f'{url}: an LFS pointer, not the object')
            with open(path + '.part', 'wb') as file:
                file.write(content)
            os.replace(path + '.part', path)
            return len(content)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts - 1 or getattr(error, 'code', None) == 404:
                raise
            log(f'  retry {url}: {error}')
            time.sleep(2 ** attempt)
    return 0


# Fetch the CREMA-D audio (WAV) and metadata into data/corpora/crema_d/
def fetch(args):
    os.makedirs(CREMAD_DIR, exist_ok=True)
    tree_path = os.path.join(CREMAD_DIR, 'tree.json')
    download(TREE_URL, tree_path)
    with open(tree_path, encoding='utf-8') as file:
        tree = json.load(file)
    wavs = sorted(item['path'] for item in tree['tree'] if item['type'] == 'blob' and item['path'].startswith('AudioWAV/'))
    for path in METADATA_FILES:
        download(RAW_URL.format(path=path), os.path.join(CREMAD_DIR, path))
    tasks = [(MEDIA_URL.format(path=path), os.path.join(CREMAD_DIR, path)) for path in wavs]
    start, total, done = time.time(), 0, 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for size in pool.map(lambda task: download(*task), tasks):
            total += size
            done += 1
            if done % 500 == 0:
                log(f'  {done}/{len(tasks)} files, {total / 1e6:.0f} MB downloaded, {time.time() - start:.0f} s')
    present = sum(1 for path in wavs if os.path.exists(os.path.join(CREMAD_DIR, path)))
    log(f'crema-d: {present}/{len(wavs)} WAV files present; downloaded {total / 1e6:.0f} MB in {time.time() - start:.0f} s')
    write_readme()
    return 0 if present == len(wavs) else 1


# Directory size in bytes
def directory_size(path):
    return sum(os.path.getsize(os.path.join(dirpath, name)) for dirpath, _, names in os.walk(path) for name in names)


README_START = '<!-- crema-d: written by nn/cremad.py readme -->'
README_END = '<!-- crema-d end -->'


# Write (or replace) CREMA-D's section of data/corpora/README.md (corpora.py rewrites the file without it - fetch
# writes it again after it; the marker keeps its first writer's name, so an existing section is found)
def write_readme():
    path = os.path.join(CORPORA_DIR, 'README.md')
    text = open(path, encoding='utf-8').read() if os.path.exists(path) else '# Speech corpora\n'
    if README_START in text:
        text = text[:text.index(README_START)].rstrip('\n') + '\n' + text[text.index(README_END) + len(README_END):].lstrip('\n')
    size = directory_size(CREMAD_DIR) / 1e6 if os.path.isdir(CREMAD_DIR) else 0
    section = '\n'.join([
        README_START,
        '## crema-d',
        '',
        f'- Source: https://github.com/{REPO} (git LFS; the WAV files from {MEDIA_URL.format(path="AudioWAV/<name>.wav")}, '
        'fetched by `python -m nn.cremad fetch`)',
        '- Contents: CREMA-D, the Crowd-sourced Emotional Multimodal Actors Dataset - 91 actors (48 male, 43 female, '
        'ages 20-74, several races and ethnicities) reading 12 sentences as angry, disgusted, fearful, happy, neutral, '
        'and sad (the sentence "It\'s eleven o\'clock" also at low, medium, and high intensity); 7,442 clips, 16 kHz '
        'mono WAV, with the crowd-sourced perceptual ratings (emotion and intensity) of each clip '
        '(processedResults/, finished*Responses.csv) and the actors\' demographics (VideoDemographics.csv); the video '
        'and MP3 copies are not fetched',
        f'- License: {LICENSE}',
        f'- Size: {size:.0f} MB',
        '',
        'Not fetched: EmoV-DB (OpenSLR 115 - four speakers reading CMU ARCTIC sentences as neutral, amused, angry, sleepy, '
        'and disgusted) - its license (github.com/numediart/EmoV-DB LICENSE.md) allows only "Non-commercial Purposes" '
        '(research, teaching, publication, personal experimentation) and not incorporation into a product, which would '
        'restrict an MIT-licensed model trained on it',
        README_END,
        ''
    ])
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text.rstrip('\n') + '\n\n' + section)
    return 0


#
# The clips
#

BUILD_DIR = os.path.join(ROOT, 'build', 'nn', 'cremad')
ALIGNMENTS_PATH = os.path.join(BUILD_DIR, 'alignments.jsonl')
HF_HOME = os.path.join(BUILD_DIR, 'hf')


# The CREMA-D clips - [{id, actor, sentence, emotion, intensity, text, wav}], sorted by id
def clips():
    result = []
    for name in sorted(os.listdir(WAV_DIR)):
        if not name.endswith('.wav'):
            continue
        clip_id = name[:-len('.wav')]
        actor, sentence, emotion, intensity = clip_id.split('_')
        result.append({'id': clip_id, 'actor': actor, 'sentence': sentence, 'emotion': EMOTIONS[emotion],
                       'intensity': INTENSITIES[intensity], 'text': SENTENCES[sentence],
                       'wav': os.path.join(WAV_DIR, name)})
    return result


#
# Alignment - forced alignment of our tokens with a wav2vec2 phoneme recognizer's CTC posteriors
#

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
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    from transformers import Wav2Vec2ForCTC  # pylint: disable=import-outside-toplevel
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
    import soundfile  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    samples, rate = soundfile.read(path, dtype='float64', always_2d=True)
    samples = samples.mean(axis=1)
    if rate != CTC_SAMPLE_RATE:
        samples = soxr.resample(samples, rate, CTC_SAMPLE_RATE, quality='VHQ')
    return np.ascontiguousarray(samples)


# Align the CREMA-D clips -> build/nn/cremad/alignments.jsonl ({id, tokens, labels, confidences, score} per clip)
def align_command(args):
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    os.makedirs(BUILD_DIR, exist_ok=True)
    items = clips()
    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]
    model, vocab, device = load_ctc(args.device)
    start = time.time()
    failures = 0
    with open(ALIGNMENTS_PATH if not args.limit else ALIGNMENTS_PATH + '.sample', 'w', encoding='utf-8') as file:
        for offset in range(0, len(items), 256):
            group = items[offset:offset + 256]
            samples_list = [read_16k(item['wav']) for item in group]
            posteriors = ctc_log_posteriors(model, device, samples_list)
            for item, samples, log_posteriors in zip(group, samples_list, posteriors):
                tokens = tokenize(item['text'])['tokens']
                record = {'id': item['id'], 'seconds': round(len(samples) / CTC_SAMPLE_RATE, 4), 'tokens': tokens}
                result = ctc_labels(tokens, log_posteriors, vocab, samples) if len(samples) > 0.3 * CTC_SAMPLE_RATE else None
                if result is None:
                    failures += 1
                    record['labels'] = None
                else:
                    labels, confidences, score = result
                    record['labels'] = [[label, round(begin, 4), round(end, 4)] for label, begin, end in labels]
                    record['confidences'] = [round(value, 4) for value in confidences]
                    record['score'] = round(score, 4)
                    record['greedy'] = greedy_decode(log_posteriors, vocab)
                file.write(json.dumps(record, ensure_ascii=False) + '\n')
            log(f'  {offset + len(group)}/{len(items)} clips, {time.time() - start:.0f} s')
    log(f'aligned {len(items) - failures} of {len(items)} clips ({failures} failed) in {time.time() - start:.0f} s')
    return 0


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


#
# Metadata - the actors and the crowd's audio-only ratings
#

# The actors' demographics - {actor: {sex, age, race, ethnicity}}
def actors():
    import csv  # pylint: disable=import-outside-toplevel
    result = {}
    with open(os.path.join(CREMAD_DIR, 'VideoDemographics.csv'), encoding='utf-8') as file:
        for row in csv.DictReader(file):
            result[row['ActorID']] = {'sex': row['Sex'].lower(), 'age': int(row['Age']), 'race': row['Race'],
                                      'ethnicity': row['Ethnicity']}
    return result


RATING_EMOTIONS = {'A': 'angry', 'D': 'disgusted', 'F': 'fearful', 'H': 'happy', 'N': 'neutral', 'S': 'sad'}


# The crowd's audio-only (voice) ratings of each clip (finishedResponses.csv, queryType 1) - {clip: {votes {emotion:
# count}, count, vote (the plurality emotion, ties joined by ':'), agreement (the share voting the intended emotion),
# level (the mean rated intensity, 0-100, of the votes for the intended emotion; None without any)}}
def voice_ratings():
    import csv  # pylint: disable=import-outside-toplevel
    votes = {}
    levels = {}
    with open(os.path.join(CREMAD_DIR, 'finishedResponses.csv'), encoding='utf-8') as file:
        for row in csv.DictReader(file):
            if row['queryType'] != '1' or row['respEmo'] not in RATING_EMOTIONS:
                continue
            emotion = RATING_EMOTIONS[row['respEmo']]
            clip_votes = votes.setdefault(row['clipName'], {name: 0 for name in RATING_EMOTIONS.values()})
            clip_votes[emotion] += 1
            if row['respLevel'].isdigit():
                levels.setdefault((row['clipName'], emotion), []).append(int(row['respLevel']))
    result = {}
    for clip_id, clip_votes in votes.items():
        intended = EMOTIONS[clip_id.split('_')[2]]
        count = sum(clip_votes.values())
        most = max(clip_votes.values())
        intended_levels = levels.get((clip_id, intended), [])
        result[clip_id] = {
            'votes': clip_votes, 'count': count,
            'vote': ':'.join(name for name, value in clip_votes.items() if value == most),
            'agreement': round(clip_votes[intended] / count, 3) if count else None,
            'level': round(sum(intended_levels) / len(intended_levels), 1) if intended_levels else None
        }
    return result


#
# Build - build/nn/data/cremad/ in data.py's format, with style labels
#

DATA_DIR = os.path.join(os.environ.get('NN_DATA_DIR', os.path.join(ROOT, 'build', 'nn', 'data')), 'cremad')
SUMMARY_PATH = os.path.join(ROOT, 'data', 'nn', 'cremad.json')
FORMANT_CEILINGS = {'female': 5500, 'male': 5000}

# Inclusion - the greedy phone error rate above which a clip is too garbled to trust its alignment, the missing tail
# (phones) that marks a clip cut short (with under TRUNCATED_SILENCE seconds of trailing silence, or at any silence
# from TRUNCATED_ALWAYS phones), a sentence read in place of the intended one (its error at most WRONG_BEST, the
# intended sentence's at least WRONG_INTENDED - the clip is relabeled), and the clipped-sample share
MAX_PHONE_ERROR = 0.6
TRUNCATED_TAIL = 3
TRUNCATED_ALWAYS = 5
TRUNCATED_SILENCE = 0.05
WRONG_BEST = 0.4
WRONG_INTENDED = 0.6
MAX_CLIPPED = 0.005


# The split of an actor's sentence (all six emotions together) - 'test' or 'val' (about 5% each, by a hash), else 'train'
def clip_split(actor, sentence):
    import hashlib  # pylint: disable=import-outside-toplevel
    bucket = int(hashlib.sha1(f'cremad/{actor}/{sentence}'.encode('utf-8')).hexdigest(), 16) % 100
    return 'test' if bucket < 5 else ('val' if bucket < 10 else 'train')


# First-pass F0 of a clip (Praat, 60-700 Hz) - the voiced values (a worker task)
def first_pass_f0(path):
    import numpy as np  # pylint: disable=import-outside-toplevel
    import parselmouth  # pylint: disable=import-outside-toplevel
    sound = parselmouth.Sound(path)
    f0 = sound.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=700).selected_array['frequency']
    return f0[f0 > 0].astype(np.float32)


# Process one clip - tokens, durations, audio, and tracks (a worker task)
def process_clip(task):
    import numpy as np  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    from . import align, data  # pylint: disable=import-outside-toplevel
    item, tokens, labels, mean_seconds, settings = task
    samples = read_16k(item['wav'])
    seconds, report = label_durations(tokens, [tuple(label) for label in labels], samples, CTC_SAMPLE_RATE, mean_seconds)
    wav_seconds = len(samples) / CTC_SAMPLE_RATE

    # Cut the edge silences to at most data.py's MAX_EDGE_SILENCE
    speech_start = seconds[0]
    speech_end = wav_seconds - seconds[-1]
    cut_start = max(0.0, speech_start - data.MAX_EDGE_SILENCE)
    cut_end = min(wav_seconds, speech_end + data.MAX_EDGE_SILENCE)
    seconds[0] = speech_start - cut_start
    seconds[-1] = cut_end - speech_end
    frames = align.durations_to_frames(seconds, data.FRAME_SECONDS, tokens)
    frame_count = sum(frames)
    resampled = soxr.resample(samples, CTC_SAMPLE_RATE, data.SAMPLE_RATE, quality='VHQ')
    start = int(round(cut_start * data.SAMPLE_RATE))
    audio = resampled[start:start + frame_count * data.HOP]
    audio = np.pad(audio, (0, frame_count * data.HOP - len(audio)))
    tracks = data.frame_tracks(audio, frame_count, *settings)
    pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)
    clipped = float(np.mean(np.abs(samples) >= 0.999))
    energy = tracks[:, data.CHANNELS.index('energy')]
    return {'item': item, 'tokens': tokens, 'seconds': seconds, 'frames': frames, 'cutStart': cut_start,
            'report': report, 'audio': pcm, 'tracks': tracks, 'clipped': clipped,
            'peakDb': round(float(20 * np.log10(max(1e-9, np.max(np.abs(samples))))), 2),
            'levelDb': round(float(np.percentile(energy, 95)), 2),
            'noiseFloorDb': round(float(np.percentile(energy, 5) - np.percentile(energy, 95)), 2)}


# Build build/nn/data/cremad/ (or --out) and data/nn/cremad.json (or the summary beside it)
def build(args):
    import collections  # pylint: disable=import-outside-toplevel
    import multiprocessing  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    from . import align, data  # pylint: disable=import-outside-toplevel
    from .frontend import BOUNDARIES, TOKEN_IDS, tokenize, vocabulary  # pylint: disable=import-outside-toplevel
    start_time = time.time()
    out_dir = args.out or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    items = clips()
    actor_info = actors()
    ratings = voice_ratings()
    with open(ALIGNMENTS_PATH, encoding='utf-8') as file:
        alignments = {record['id']: record for record in map(json.loads, file)}
    sentence_tokens = {code: tokenize(text)['tokens'] for code, text in SENTENCES.items()}
    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]

    # Check each clip - the sentence read, a cut-short tail, garbling
    checks = {}
    for item in items:
        record = alignments[item['id']]
        check = {'sentence': item['sentence'], 'reasons': []}
        if not record['labels']:
            check['reasons'].append('no alignment (silent or too short)')
        else:
            errors = {code: greedy_phone_error(record['greedy'], tokens) for code, tokens in sentence_tokens.items()}
            best = min(errors, key=lambda code: errors[code][0])
            if best != item['sentence'] and errors[best][0] <= WRONG_BEST and errors[item['sentence']][0] >= WRONG_INTENDED:
                check['sentence'] = best
            check['phoneError'], check['tail'] = errors[check['sentence']]
            check['trailingSilence'] = record['labels'][-1][2] - record['labels'][-1][1] if record['labels'][-1][0] == 'pau' else 0.0
            if check['tail'] >= TRUNCATED_ALWAYS or (check['tail'] >= TRUNCATED_TAIL and check['trailingSilence'] < TRUNCATED_SILENCE):
                check['reasons'].append(f'cut short ({check["tail"]} final phones missing)')
            if check['phoneError'] > MAX_PHONE_ERROR:
                check['reasons'].append(f'garbled (greedy phone error {check["phoneError"]:.2f} > {MAX_PHONE_ERROR})')
        checks[item['id']] = check

    # Relabeled clips are realigned to the sentence actually read (alignments.jsonl holds the intended sentence's)
    relabeled = [item for item in items if checks[item['id']]['sentence'] != item['sentence']]
    for item in relabeled:
        log(f'  {item["id"]}: reads {checks[item["id"]]["sentence"]} ("{SENTENCES[checks[item["id"]]["sentence"]]}") - realigning')
    if relabeled:
        realign_path = os.path.join(BUILD_DIR, 'realigned.jsonl')
        if not os.path.exists(realign_path) or args.realign:
            import subprocess  # pylint: disable=import-outside-toplevel
            python = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')
            ids = [f'{item["id"]}={checks[item["id"]]["sentence"]}' for item in relabeled]
            subprocess.run([python, '-B', '-m', 'picovox_train.cremad', 'realign', *ids], check=True, cwd=ROOT,
                           env={**os.environ, 'PYTHONPATH': SOURCE_DIR})
        with open(realign_path, encoding='utf-8') as file:
            for record in map(json.loads, file):
                alignments[record['id']] = record

    # Typical phone durations (for splitting alignment blocks - with exact alignment, unused but required)
    totals = collections.defaultdict(lambda: [0.0, 0])
    for record in alignments.values():
        for label, begin, end in record['labels'] or []:
            key = align.label_key(label)
            totals[key][0] += end - begin
            totals[key][1] += 1
    mean_seconds = {key: total / count for key, (total, count) in totals.items()}

    # Pitch ranges per actor and emotion (two-pass, Hirst 2011 - floor 0.75 x Q1, ceiling 1.5 x Q3)
    with multiprocessing.Pool(args.jobs) as pool:
        first_f0 = pool.map(first_pass_f0, [item['wav'] for item in items], chunksize=8)
    pooled = collections.defaultdict(list)
    for item, values in zip(items, first_f0):
        pooled[(item['actor'], item['emotion'])].append(values)
        pooled[item['actor']].append(values)
    ranges = {}
    for key, arrays in pooled.items():
        values = np.concatenate(arrays) if arrays else np.zeros(0)
        if len(values) < 50:
            continue
        ranges[key] = (round(max(50.0, 0.75 * float(np.percentile(values, 25))), 1),
                       round(min(900.0, 1.5 * float(np.percentile(values, 75))), 1))

    tasks = []
    for item in items:
        record = alignments[item['id']]
        if not record['labels']:
            continue
        sex = actor_info[item['actor']]['sex']
        floor, ceiling = ranges.get((item['actor'], item['emotion']), ranges[item['actor']])
        clip = dict(item, sentence=checks[item['id']]['sentence'], text=SENTENCES[checks[item['id']]['sentence']])
        tokens = sentence_tokens[clip['sentence']]
        if [align.token_key(token) for token in tokens] != [align.token_key(token) for token in record['tokens']]:
            raise ValueError(f'{item["id"]}: the front end\'s phones of sentence {clip["sentence"]} differ from its alignment\'s - '
                             'run align again (and realign the relabeled clips)')
        tasks.append((clip, tokens, record['labels'], mean_seconds, (floor, ceiling, FORMANT_CEILINGS[sex])))
    with multiprocessing.Pool(args.jobs) as pool:
        results = pool.map(process_clip, tasks, chunksize=4)

    # The actors - speaker entries (sorted by actor id)
    speakers = {}
    for index, actor in enumerate(sorted(actor_info)):
        info = actor_info[actor]
        speakers[f'cremad-{actor}'] = {
            'index': index, 'sex': info['sex'], 'corpus': 'CREMA-D', 'actor': actor, 'age': info['age'],
            'race': info['race'], 'ethnicity': info['ethnicity'], 'sourceSampleRate': 16000, 'bandwidth': 8000,
            'pitchFloor': ranges[actor][0], 'pitchCeiling': ranges[actor][1], 'formantCeiling': FORMANT_CEILINGS[info['sex']],
            'pitchRanges': {emotion: list(ranges[(actor, emotion)]) for emotion in EMOTIONS.values() if (actor, emotion) in ranges},
            # (the actors' accents - the former accent classification's, in git history - unmeasured)
            'accent': None, 'accentShare': None, 'northAmerican': None, 'neutralNorthAmerican': None
        }

    # Concatenate
    tokens, durations, seconds, audio, frames, entries = [], [], [], [], [], []
    token_offset = frame_offset = 0
    emotion_ids = {name: index for index, name in enumerate(EMOTIONS.values())}
    intensity_ids = {name: index for index, name in enumerate(INTENSITIES.values())}
    by_id = {result['item']['id']: result for result in results}
    with open(os.path.join(out_dir, 'alignment.jsonl'), 'w', encoding='utf-8') as alignment_file:
        for item in items:
            result = by_id.get(item['id'])
            check = checks[item['id']]
            if result is None:
                continue
            record = alignments[item['id']]
            reasons = list(check['reasons'])
            if result['clipped'] > MAX_CLIPPED:
                reasons.append(f'clipped ({100 * result["clipped"]:.2f}% of samples at full scale)')
            if any(count == 0 for token, count in zip(result['tokens'], result['frames']) if token not in BOUNDARIES):
                reasons.append('a phone with no frames')
            frame_count = sum(result['frames'])
            rating = ratings.get(item['id'], {})
            confidences = record['confidences']
            entry = {
                'id': f'cremad/{item["id"]}',
                'speaker': f'cremad-{item["actor"]}',
                'sentence': check['sentence'],
                'split': clip_split(item['actor'], check['sentence']),
                'text': SENTENCES[check['sentence']],
                'include': not reasons,
                'reason': '; '.join(reasons),
                'tokenOffset': token_offset,
                'tokenCount': len(result['tokens']),
                'frameOffset': frame_offset,
                'frameCount': frame_count,
                'sampleOffset': frame_offset * data.HOP,
                'sampleCount': frame_count * data.HOP,
                'trimStart': round(result['cutStart'], 5),
                'emotion': item['emotion'],
                'emotionId': emotion_ids[item['emotion']],
                'intensity': item['intensity'],
                'intensityId': intensity_ids[item['intensity']],
                'voiceVote': rating.get('vote'),
                'voiceAgreement': rating.get('agreement'),
                'voiceLevel': rating.get('level'),
                'voiceRatings': rating.get('count', 0),
                'alignScore': record['score'],
                'phoneConfidence': round(sum(confidences) / len(confidences), 3),
                'heard': round(sum(1 for value in confidences if value >= 0.3) / len(confidences), 3),
                'phoneError': round(check['phoneError'], 3),
                'clipped': round(result['clipped'], 5),
                'peakDb': result['peakDb'],
                'levelDb': result['levelDb'],
                'noiseFloorDb': result['noiseFloorDb'],
                'relabeled': check['sentence'] != item['sentence']
            }
            entries.append(entry)
            tokens.append(np.asarray([TOKEN_IDS[token] for token in result['tokens']], dtype=np.int16))
            durations.append(np.asarray(result['frames'], dtype=np.int16))
            seconds.append(np.asarray(result['seconds'], dtype=np.float32))
            audio.append(result['audio'])
            frames.append(result['tracks'])
            token_offset += len(result['tokens'])
            frame_offset += frame_count
            alignment_file.write(json.dumps({'id': entry['id'], 'tokens': ' '.join(result['tokens']),
                                             'frames': result['frames'], 'greedy': record['greedy'],
                                             'confidences': confidences}, ensure_ascii=False) + '\n')
    np.save(os.path.join(out_dir, 'tokens.npy'), np.concatenate(tokens))
    np.save(os.path.join(out_dir, 'durations.npy'), np.concatenate(durations))
    np.save(os.path.join(out_dir, 'seconds.npy'), np.concatenate(seconds))
    np.save(os.path.join(out_dir, 'audio.npy'), np.concatenate(audio))
    np.save(os.path.join(out_dir, 'frames.npy'), np.concatenate(frames))
    index = {
        'version': data.FORMAT_VERSION,
        'corpus': 'CREMA-D',
        'license': LICENSE,
        'sampleRate': data.SAMPLE_RATE,
        'hop': data.HOP,
        'frameRate': data.SAMPLE_RATE / data.HOP,
        'channels': data.CHANNELS,
        'channelNotes': dict(data.CHANNEL_NOTES, f0='Praat autocorrelation F0 (Hz), 0 where unvoiced - per actor and '
                             'emotion two-pass range (speakers[...].pitchRanges)'),
        'vocab': vocabulary(),
        'maxEdgeSilence': data.MAX_EDGE_SILENCE,
        'alignment': {'method': f'CTC forced alignment with {CTC_REPO} (nn/cremad.py)', 'shift': CTC_SHIFT,
                      'split': CTC_SPLIT, 'edgeExtend': EDGE_EXTEND},
        'emotions': list(EMOTIONS.values()),
        'intensities': list(INTENSITIES.values()),
        'styleNotes': {
            'emotion': 'the intended emotion (the file name) - emotionId indexes "emotions"',
            'intensity': 'the intended intensity - low/medium/high only for sentence IEO, else unspecified',
            'voiceVote': 'the crowd\'s plurality emotion for the audio alone (ties joined by ":")',
            'voiceAgreement': 'the share of audio-only raters choosing the intended emotion',
            'voiceLevel': 'the mean rated intensity (0-100) of the audio-only votes for the intended emotion',
            'heard': 'the share of phones the recognizer found (best posterior >= 0.3) - a clarity measure',
            'phoneError': 'the recognizer\'s greedy phone error rate against the sentence',
            'noiseFloorDb': '5th- minus 95th-percentile frame energy (dB) - higher is noisier',
            'relabeled': 'the actor read another of the sentences than the file name says (text corrected)'
        },
        'speakers': speakers,
        'utterances': entries
    }
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=1)
        file.write('\n')
    # (the committed summary only for the repository's default directory - a build elsewhere, as make train's with
    # NN_DATA_DIR, writes it beside its data)
    default_dir = os.path.abspath(out_dir) == os.path.abspath(os.path.join(ROOT, 'build', 'nn', 'data', 'cremad'))
    write_summary(index, SUMMARY_PATH if default_dir else os.path.join(out_dir, 'cremad.json'))
    included = [entry for entry in entries if entry['include']]
    print(f'Built {out_dir}: {len(entries)} clips ({len(included)} included), {frame_offset / 100 / 3600:.2f} hours '
          f'({sum(entry["frameCount"] for entry in included) / 100 / 3600:.2f} included), {time.time() - start_time:.0f} s')
    reasons = collections.Counter(reason.split(' (')[0] for entry in entries for reason in entry['reason'].split('; ') if reason)
    print(f'  excluded: {dict(reasons)}')
    return 0


# Write data/nn/cremad.json - the small committed summary: the corpus, its license, the counts by emotion, intensity,
# and split, the actors (demographics, accent, pitch range, clips included), the alignment's check against ARCTIC, and
# the emotion classifier's recognition of the recordings
def write_summary(index, path=SUMMARY_PATH):
    import collections  # pylint: disable=import-outside-toplevel
    entries = index['utterances']
    included = [entry for entry in entries if entry['include']]
    counts = collections.defaultdict(lambda: collections.Counter())
    for entry in included:
        counts['emotion'][entry['emotion']] += 1
        counts['intensity'][entry['intensity']] += 1
        counts['split'][entry['split']] += 1
        counts['sex'][index['speakers'][entry['speaker']]['sex']] += 1
    per_actor = collections.Counter(entry['speaker'] for entry in included)
    reasons = collections.Counter(reason.split(' (')[0] for entry in entries for reason in entry['reason'].split('; ') if reason)
    summary = {
        'description': 'CREMA-D emotional speech for the style modifiers (nn/cremad.py; the data in build/nn/data/cremad/, '
                       'nn/data.py\'s format with style labels) - counts, actors, alignment quality, and the listener '
                       'panel\'s emotion recognition of the recordings',
        'corpus': 'CREMA-D (Cao et al., 2014) - https://github.com/' + REPO,
        'license': LICENSE,
        'clips': len(entries),
        'included': len(included),
        'hours': round(sum(entry['frameCount'] for entry in entries) / 100 / 3600, 3),
        'includedHours': round(sum(entry['frameCount'] for entry in included) / 100 / 3600, 3),
        'excluded': dict(reasons),
        'relabeled': [entry['id'] for entry in entries if entry['relabeled']],
        'counts': {key: dict(sorted(value.items())) for key, value in counts.items()},
        'emotions': index['emotions'],
        'intensities': index['intensities'],
        'alignment': dict(index['alignment']),
        'actors': {name: {key: speaker[key] for key in ('index', 'sex', 'age', 'race', 'ethnicity', 'pitchFloor',
                                                         'pitchCeiling', 'accent', 'northAmerican')}
                   | {'included': per_actor[name]} for name, speaker in index['speakers'].items()}
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(summary, indent=1) + '\n')


# Realign clips to the sentence actually read - ids as clip=SENTENCE -> build/nn/cremad/realigned.jsonl
def realign_command(args):
    from .frontend import tokenize  # pylint: disable=import-outside-toplevel
    model, vocab, device = load_ctc(args.device)
    pairs = [item.split('=') for item in args.ids]
    samples_list = [read_16k(os.path.join(WAV_DIR, clip_id + '.wav')) for clip_id, _ in pairs]
    posteriors = ctc_log_posteriors(model, device, samples_list)
    with open(os.path.join(BUILD_DIR, 'realigned.jsonl'), 'w', encoding='utf-8') as file:
        for (clip_id, sentence), samples, log_posteriors in zip(pairs, samples_list, posteriors):
            tokens = tokenize(SENTENCES[sentence])['tokens']
            labels, confidences, score = ctc_labels(tokens, log_posteriors, vocab, samples)
            file.write(json.dumps({'id': clip_id, 'sentence': sentence, 'seconds': round(len(samples) / CTC_SAMPLE_RATE, 4),
                                   'tokens': tokens, 'labels': [[label, round(begin, 4), round(end, 4)] for label, begin, end in labels],
                                   'confidences': [round(value, 4) for value in confidences], 'score': round(score, 4),
                                   'greedy': greedy_decode(log_posteriors, vocab)}, ensure_ascii=False) + '\n')
    return 0


# The command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog='picovox_train.cremad', description='CREMA-D emotional speech for the style modifiers')
    commands = parser.add_subparsers(dest='command', required=True)
    fetch_parser = commands.add_parser('fetch', help='download CREMA-D into data/corpora/crema_d/')
    fetch_parser.add_argument('-j', '--jobs', type=int, default=8, help='parallel connections')
    align_parser = commands.add_parser('align', help='CTC forced alignment of the clips (build/venv-listeners)')
    align_parser.add_argument('--limit', type=int, help='a sample of this many clips (alignments.jsonl.sample)')
    align_parser.add_argument('--device', default=None, help='torch device (default mps if available)')
    build_parser = commands.add_parser('build', help='build build/nn/data/cremad/ (build/venv-nn)')
    build_parser.add_argument('--limit', type=int, help='a sample of this many clips (a quick test)')
    build_parser.add_argument('--realign', action='store_true', help='realign the relabeled clips again')
    build_parser.add_argument('--out', help='the data directory (default: build/nn/data/cremad/, its summary data/nn/cremad.json)')
    build_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    realign_parser = commands.add_parser('realign', help='realign clips to another sentence (clip=SENTENCE ...)')
    realign_parser.add_argument('ids', nargs='+')
    realign_parser.add_argument('--device', default=None, help='torch device (default mps if available)')
    args = parser.parse_args(argv)
    handler = {'build': build, 'realign': realign_command, 'fetch': fetch, 'align': align_command}
    return handler[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
