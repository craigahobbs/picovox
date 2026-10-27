# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training data's featurization and format, shared by the corpora (multi.py - the reader pool, and cremad.py -
the emotional speech): each utterance's tokens (our front end's, frontend.py), each token's duration in frames
(pauses included, reconciled with the corpus's phone labels - align.py), the audio (24 kHz, 16-bit), and auxiliary
frame tracks (Praat F0, voicing, pitch strength, harmonicity, energy, formants F1-F4 and their bandwidths).

A corpus directory (build/nn/data/<corpus>/) - concatenated arrays and an index of offsets into them:

- index.json - format constants, the channel names, the speakers, and the utterances: {id, speaker, sentence,
  split, text, include, reason, tokenOffset, tokenCount, frameOffset, frameCount, sampleOffset, sampleCount, trimStart,
  match}. Every array is memory-mapped by ArcticData.
- tokens.npy (int16) - token ids (vocab.json)
- durations.npy (int16) - each token's duration in frames; an utterance's durations sum to its frame count
- seconds.npy (float32) - each token's duration in seconds, before quantization
- audio.npy (int16) - 24 kHz audio, frameCount x 240 samples per utterance (sampleOffset = frameOffset x 240)
- frames.npy (float32, [frames, channels]) - the auxiliary tracks at 100 frames per second (frame k is the 10 ms of
  samples [240k, 240k + 240), its center at (k + 0.5) x 10 ms)
- alignment.jsonl - per utterance, the alignment's edits (diagnostics)

An ARCTIC utterance's split is by sentence (the same sentences held out for every ARCTIC speaker): about 5% validation
and 5% test, by a hash of the sentence id. (The single-speaker ARCTIC slt/bdl corpus this module once built, and its
statistics and LibriTTS-R survey, are in git history.)
"""

import collections
import hashlib
import json
import os
import re
import subprocess

import numpy as np

from . import ROOT, align
from .frontend import tokenize
from .wav import read_wav


CORPORA_DIR = os.path.join(ROOT, 'data', 'corpora')
ARCTIC_DIR = os.path.join(CORPORA_DIR, 'arctic')

# The LibriTTS-R alignments (LibriTTSLabel's MFA phone labels) and the LibriTTS-R subsets in the pool (multi.py,
# readings.py)
LABEL_DIR = os.path.join(CORPORA_DIR, 'libritts_label', 'lab', 'phone')
SUBSETS = ['train-clean-100', 'dev-clean']
DATA_DIR = os.environ.get('NN_DATA_DIR', os.path.join(ROOT, 'build', 'nn', 'data'))

# The format
SAMPLE_RATE = 24000
HOP = 240                                  # samples per frame - 10 ms
FRAME_SECONDS = HOP / SAMPLE_RATE
MAX_EDGE_SILENCE = 0.2                     # seconds of leading and trailing silence kept
FORMAT_VERSION = 1
CHANNELS = ['f0', 'voiced', 'strength', 'hnr', 'energy', 'f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4']
CHANNEL_NOTES = {
    'f0': 'Praat autocorrelation F0 (Hz), 0 where unvoiced - speaker-adaptive range from data/targets/arctic-<speaker>.json',
    'voiced': '1 where Praat finds the frame voiced, else 0',
    'strength': 'Praat pitch strength (autocorrelation peak, 0-1) of the selected candidate, 0 where unvoiced',
    'hnr': 'Harmonics-to-noise ratio (dB, cross-correlation), clipped to [-20, 40] (-20 in silence)',
    'energy': 'RMS level (dB re full scale) of a 20 ms Hann window centered on the frame, floor -100',
    'f1': 'Burg formant 1 (Hz) at the speaker\'s formant ceiling (5 formants, 25 ms window), 0 where undefined',
    'b1': 'Bandwidth of formant 1 (Hz), 0 where undefined'
}

# The minimum share of our phones paired exactly or nearly with a label phone for an utterance to be used in training,
# and the minimum for its worst word (of two or more phones) - our letter-to-sound fallback's spelled-out guesses at
# unknown words (names, possessives of names) don't match what was said, and would teach the network nonsense
MIN_MATCH = 0.85
MIN_WORD_MATCH = 0.5


#
# Corpus reading
#

# Read an ARCTIC/Festival prompt file - {id: text}
def read_prompts(path):
    prompts = {}
    with open(path, encoding='utf-8', errors='replace') as file:
        for line in file:
            match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)\s*$', line.strip())
            if match:
                prompts[match.group(1)] = match.group(2)
    return prompts


# Read an xwaves (ARCTIC) label file - [(label, start, end)] in seconds
def read_xwaves(path):
    labels = []
    start = 0.0
    in_body = False
    with open(path, encoding='utf-8', errors='replace') as file:
        for line in file:
            if not in_body:
                in_body = line.strip() == '#'
                continue
            fields = line.split()
            if len(fields) >= 3:
                end = float(fields[0])
                labels.append((fields[2], start, end))
                start = end
    return labels


# An utterance's phone labels - [(label, start, end)] in seconds, from its xwaves (ARCTIC) or MFA (LibriTTSLabel,
# utterance['labelFormat'] == 'mfa') label file
def read_labels(utterance):
    if utterance.get('labelFormat') == 'mfa':
        return read_mfa(utterance['lab'])
    return read_xwaves(utterance['lab'])


# Read a LibriTTSLabel (MFA) tab-separated label file - [(label, start, end)], gaps as pauses
def read_mfa(path):
    labels = []
    previous_end = 0.0
    with open(path, encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 2:
                start, end = float(fields[0]), float(fields[1])
                if start > previous_end + 1e-4:
                    labels.append(('sil', previous_end, start))
                labels.append((fields[2] if len(fields) > 2 else '', start, end))
                previous_end = end
    return labels


# The ARCTIC utterances of a speaker - [{speaker, sentence, wav, lab, text}]
def arctic_utterances(speaker):
    directory = os.path.join(ARCTIC_DIR, f'cmu_us_{speaker}_arctic')
    prompts = read_prompts(os.path.join(directory, 'etc', 'txt.done.data'))
    utterances = []
    for name in sorted(os.listdir(os.path.join(directory, 'wav'))):
        # Only the recordings (earlier evaluations cached whisper.cpp transcripts beside them)
        if not name.endswith('.wav'):
            continue
        sentence = os.path.splitext(name)[0]
        utterances.append({'speaker': speaker, 'sentence': sentence, 'wav': os.path.join(directory, 'wav', name),
                           'lab': os.path.join(directory, 'lab', sentence + '.lab'), 'text': prompts.get(sentence)})
    return utterances


# The split of a sentence - 'test' or 'val' (about 5% each, by a hash of the sentence id), else 'train'
def sentence_split(sentence):
    bucket = int(hashlib.sha1(sentence.encode('utf-8')).hexdigest(), 16) % 100
    return 'test' if bucket < 5 else ('val' if bucket < 10 else 'train')


# A speaker's analysis settings from data/targets/arctic-<speaker>.json - (pitch floor, pitch ceiling, formant ceiling)
def speaker_settings(speaker):
    with open(os.path.join(ROOT, 'data', 'targets', f'arctic-{speaker}.json'), encoding='utf-8') as file:
        analysis = json.load(file)['analysis']
    return analysis['pitchFloor'], analysis['pitchCeiling'], analysis['formantCeiling']


# Typical label durations (seconds) by comparison key over a speaker's labels - for splitting alignment blocks
def mean_label_seconds(utterances):
    totals = collections.defaultdict(lambda: [0.0, 0])
    for utterance in utterances:
        if os.path.exists(utterance['lab']):
            for label, start, end in read_labels(utterance):
                key = align.label_key(label)
                totals[key][0] += end - start
                totals[key][1] += 1
    return {key: total / count for key, (total, count) in totals.items() if count}


#
# Utterance processing
#

# The silent runs of audio [(start, end)] in seconds - 10 ms frames (20 ms Hann windows) more than SILENCE_DB below
# the utterance's 95th-percentile level, at least 60 ms long
SILENCE_DB = 35


def silent_runs(samples, rate):
    hop = int(round(rate * 0.01))
    window = np.hanning(2 * hop)
    padded = np.pad(samples, (hop // 2, hop // 2 + 2 * hop))
    count = len(samples) // hop
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * hop)[:count * hop:hop]
    level = 10 * np.log10(np.sum((windows * window) ** 2, axis=1) / np.sum(window ** 2) + 1e-12)
    silent = level < np.percentile(level, 95) - SILENCE_DB
    runs = []
    index = 0
    while index < count:
        if silent[index]:
            end = index
            while end < count and silent[end]:
                end += 1
            if end - index >= 6:
                runs.append((index * 0.01, end * 0.01))
            index = end
        else:
            index += 1
    return runs


# Align an utterance's tokens with its labels - (tokens, seconds per token, cut start, cut end, report)
def align_utterance(utterance, mean_seconds, samples, rate):
    tokenized = tokenize(utterance['text'], readings=utterance.get('readings'))   # heteronym readings (readings.py)
    tokens = tokenized['tokens']
    labels = read_labels(utterance)
    wav_seconds = len(samples) / rate

    # Clip the labels to the audio (the last label extends to its end)
    labels = [(label, start, min(end, wav_seconds)) for label, start, end in labels if start < wav_seconds]
    if labels[-1][2] < wav_seconds:
        labels.append(('pau', labels[-1][2], wav_seconds))
    seconds, report = align.assign_durations(tokens, [(label, end - start) for label, start, end in labels], mean_seconds)

    # Move silences folded into the phones next to boundaries into the boundary tokens
    seconds, report['pausesRefined'], report['pauseSecondsMoved'] = align.refine_pauses(tokens, seconds, silent_runs(samples, rate))

    # Cut the leading and trailing silences to at most MAX_EDGE_SILENCE
    speech_start = seconds[0]
    speech_end = wav_seconds - seconds[-1]
    cut_start = max(0.0, speech_start - MAX_EDGE_SILENCE)
    cut_end = min(wav_seconds, speech_end + MAX_EDGE_SILENCE)
    seconds[0] = speech_start - cut_start
    seconds[-1] = cut_end - speech_end
    # Word agreement - the share of each word's phones paired exactly (or AH/AX), an inserted label phone counting
    # as one more phone; the worst word of the utterance
    report['words'] = len(tokenized['words'])
    report['worstWord'], report['worstWordMatch'] = None, 1.0
    for word in tokenized['words']:
        statuses = report['status'][word['start']:word['end']]
        good = sum(1 for item in statuses if item in ('exact', 'near'))
        total = len(statuses) + sum(1 for item in statuses if item and item.endswith('+inserted'))
        word_match = good / max(1, total)
        if total >= 2 and word_match < report['worstWordMatch']:
            report['worstWord'], report['worstWordMatch'] = word['text'], round(word_match, 3)
    report['ids'] = tokenized['ids']
    return tokens, seconds, cut_start, cut_end, report


# Frame-center sampling of a Praat track - values at times (k + 0.5) x FRAME_SECONDS, from track times and values
def sample_track(track_times, track_values, frame_count, tolerance=0.006):
    centers = (np.arange(frame_count) + 0.5) * FRAME_SECONDS
    if len(track_times) == 0:
        return np.zeros(frame_count), np.zeros(frame_count, dtype=bool)
    step = track_times[1] - track_times[0] if len(track_times) > 1 else FRAME_SECONDS
    nearest = np.clip(np.round((centers - track_times[0]) / step).astype(int), 0, len(track_times) - 1)
    valid = np.abs(track_times[nearest] - centers) <= tolerance
    return np.where(valid, track_values[nearest], 0.0), valid


# The auxiliary frame tracks of 24 kHz audio - [frames, channels]
def frame_tracks(samples, frame_count, pitch_floor, pitch_ceiling, formant_ceiling):
    import parselmouth  # pylint: disable=import-outside-toplevel, import-error
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    tracks = np.zeros((frame_count, len(CHANNELS)), dtype=np.float32)

    # F0, voicing, and pitch strength
    pitch = sound.to_pitch_ac(time_step=FRAME_SECONDS, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    selected = pitch.selected_array
    f0, valid = sample_track(np.asarray(pitch.xs()), selected['frequency'], frame_count)
    strength, _ = sample_track(np.asarray(pitch.xs()), selected['strength'], frame_count)
    voiced = valid & (f0 > 0)
    tracks[:, CHANNELS.index('f0')] = np.where(voiced, f0, 0.0)
    tracks[:, CHANNELS.index('voiced')] = voiced
    tracks[:, CHANNELS.index('strength')] = np.where(voiced, strength, 0.0)

    # Harmonicity
    harmonicity = sound.to_harmonicity_cc(time_step=FRAME_SECONDS, minimum_pitch=pitch_floor)
    hnr, valid = sample_track(np.asarray(harmonicity.xs()), harmonicity.values[0], frame_count)
    tracks[:, CHANNELS.index('hnr')] = np.clip(np.where(valid, hnr, -20.0), -20.0, 40.0)

    # Energy - a 20 ms Hann window centered on each frame
    window = np.hanning(2 * HOP)
    padded = np.pad(samples.astype(np.float64), (HOP // 2, HOP // 2 + 2 * HOP))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * HOP)[:frame_count * HOP:HOP]
    rms = np.sqrt(np.sum((windows * window) ** 2, axis=1) / np.sum(window ** 2))
    tracks[:, CHANNELS.index('energy')] = np.maximum(20 * np.log10(np.maximum(rms, 1e-12)), -100.0)

    # Formants and bandwidths
    formant = sound.to_formant_burg(time_step=FRAME_SECONDS, max_number_of_formants=5, maximum_formant=formant_ceiling,
                                    window_length=0.025, pre_emphasis_from=50)
    centers = (np.arange(frame_count) + 0.5) * FRAME_SECONDS
    for number in range(1, 5):
        frequencies = np.array([formant.get_value_at_time(number, center) for center in centers])
        bandwidths = np.array([formant.get_bandwidth_at_time(number, center) for center in centers])
        defined = np.isfinite(frequencies) & np.isfinite(bandwidths)
        tracks[:, CHANNELS.index(f'f{number}')] = np.where(defined, frequencies, 0.0)
        tracks[:, CHANNELS.index(f'b{number}')] = np.where(defined, bandwidths, 0.0)
    return tracks


# Process one utterance - its tokens, durations, audio, and tracks (a worker task)
def process_utterance(task):
    utterance, mean_seconds, settings = task
    import soxr  # pylint: disable=import-outside-toplevel
    samples, rate = read_wav(utterance['wav'])
    samples = samples.mean(axis=1)
    tokens, seconds, cut_start, cut_end, report = align_utterance(utterance, mean_seconds, samples, rate)
    frames = align.durations_to_frames(seconds, FRAME_SECONDS, tokens)
    frame_count = sum(frames)

    # Resample to 24 kHz, then cut to the frames
    resampled = samples if rate == SAMPLE_RATE else soxr.resample(samples, rate, SAMPLE_RATE, quality='VHQ')
    start = int(round(cut_start * SAMPLE_RATE))
    audio = resampled[start:start + frame_count * HOP]
    audio = np.pad(audio, (0, frame_count * HOP - len(audio)))
    tracks = frame_tracks(audio, frame_count, *settings)
    pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)
    return {'utterance': utterance, 'tokens': tokens, 'seconds': seconds, 'frames': frames, 'cutStart': cut_start,
            'cutEnd': cut_end, 'report': report, 'audio': pcm, 'tracks': tracks}


# The match measures of an alignment report - the share of our phones paired exactly or nearly with a label phone
def match_rate(report):
    return (report['exact'] + report['near']) / max(1, report['phones'])


#
# Loading (for the trainer)
#

class ArcticData:
    """The memory-mapped data of a corpus directory (build/nn/data/<corpus>/)"""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, 'index.json'), encoding='utf-8') as file:
            self.index = json.load(file)
        self.tokens = np.load(os.path.join(directory, 'tokens.npy'), mmap_mode='r')
        self.durations = np.load(os.path.join(directory, 'durations.npy'), mmap_mode='r')
        self.seconds = np.load(os.path.join(directory, 'seconds.npy'), mmap_mode='r')
        self.audio = np.load(os.path.join(directory, 'audio.npy'), mmap_mode='r')
        self.frames = np.load(os.path.join(directory, 'frames.npy'), mmap_mode='r')

    # The utterance entries of a split ('train', 'val', 'test', or None for all), of the speakers (None for all),
    # included in training unless include_excluded
    def entries(self, split=None, speakers=None, include_excluded=False):
        return [entry for entry in self.index['utterances']
                if (split is None or entry['split'] == split) and (speakers is None or entry['speaker'] in speakers)
                and (include_excluded or entry['include'])]

    # One utterance's arrays - {tokens, durations, seconds, audio (float32, -1 to 1), frames}
    def arrays(self, entry):
        tokens = slice(entry['tokenOffset'], entry['tokenOffset'] + entry['tokenCount'])
        frames = slice(entry['frameOffset'], entry['frameOffset'] + entry['frameCount'])
        samples = slice(entry['sampleOffset'], entry['sampleOffset'] + entry['sampleCount'])
        return {
            'tokens': np.array(self.tokens[tokens], dtype=np.int64),
            'durations': np.array(self.durations[tokens], dtype=np.int64),
            'seconds': np.array(self.seconds[tokens], dtype=np.float32),
            'audio': np.array(self.audio[samples], dtype=np.float32) / 32768.0,
            'frames': np.array(self.frames[frames], dtype=np.float32)
        }


#
# Token verification - lib/tokens.js against frontend.py
#

# Tokenize texts with lib/tokens.js
def js_tokenize(texts):
    process = subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'tokenizeLines.js')],
                             input=''.join(json.dumps(text) + '\n' for text in texts), capture_output=True,
                             text=True, check=True, cwd=ROOT)
    return [json.loads(line) for line in process.stdout.splitlines()]
