# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The training data - CMU ARCTIC slt (female) and bdl (male): each utterance's tokens (our front end's, nn/frontend.py),
each token's duration in frames (pauses included, reconciled with the corpus's phone labels - nn/align.py), the audio
(24 kHz, 16-bit), and auxiliary frame tracks (Praat F0, voicing, pitch strength, harmonicity, energy, formants F1-F4
and their bandwidths).

    build/venv-nn/bin/python -B -m nn.data build            # build/nn/data/arctic/ (about 1 minute on 16 cores)
    build/venv-nn/bin/python -B -m nn.data verify-tokens    # nn/js/tokens.js == nn/frontend.py on every ARCTIC text
    build/venv-nn/bin/python -B -m nn.data stats            # hours, token counts, durations -> stats.json
    build/venv-nn/bin/python -B -m nn.data listen           # UTMOS of originals and of the processed audio
    build/venv-nn/bin/python -B -m nn.data libritts         # survey LibriTTS-R dev-clean + MFA alignments

The data (build/nn/data/arctic/) - concatenated arrays and an index of offsets into them:

- index.json - format constants, the channel names, the speakers, and the utterances: {id, speaker, sentence,
  split, text, include, reason, tokenOffset, tokenCount, frameOffset, frameCount, sampleOffset, sampleCount, trimStart,
  match}. Every array is memory-mapped by ArcticData; ArcticDataset gives one utterance as tensors.
- tokens.npy (int16) - token ids (nn/vocab.json)
- durations.npy (int16) - each token's duration in frames; an utterance's durations sum to its frame count
- seconds.npy (float32) - each token's duration in seconds, before quantization
- audio.npy (int16) - 24 kHz audio, frameCount x 240 samples per utterance (sampleOffset = frameOffset x 240)
- frames.npy (float32, [frames, channels]) - the auxiliary tracks at 100 frames per second (frame k is the 10 ms of
  samples [240k, 240k + 240), its center at (k + 0.5) x 10 ms)
- alignment.jsonl - per utterance, the alignment's edits (diagnostics)

The split is by sentence (the same sentences are held out for both speakers): about 5% validation and 5% test,
by a hash of the sentence id; data/nn/arctic-split.json records it, with the test texts for evaluation.
"""

import argparse
import collections
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
import time

import numpy as np

from . import align
from .frontend import BOUNDARIES, PHRASE_ENDS, ROOT, WORD, tokenize, vocabulary


CORPORA_DIR = os.path.join(ROOT, 'data', 'corpora')
ARCTIC_DIR = os.path.join(CORPORA_DIR, 'arctic')
DATA_DIR = os.path.join(ROOT, 'build', 'nn', 'data')
SPLIT_PATH = os.path.join(ROOT, 'data', 'nn', 'arctic-split.json')

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

# The speakers - their sex; the pitch range and the formant ceiling come from their data/targets/ statistics
SPEAKERS = {'slt': 'female', 'bdl': 'male'}

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


# The ARCTIC utterances of a speaker - [{speaker, sentence, wav, lab, text}]
def arctic_utterances(speaker):
    directory = os.path.join(ARCTIC_DIR, f'cmu_us_{speaker}_arctic')
    prompts = read_prompts(os.path.join(directory, 'etc', 'txt.done.data'))
    utterances = []
    for name in sorted(os.listdir(os.path.join(directory, 'wav'))):
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
            for label, start, end in read_xwaves(utterance['lab']):
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
    tokenized = tokenize(utterance['text'])
    tokens = tokenized['tokens']
    labels = read_xwaves(utterance['lab'])
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
    import parselmouth  # pylint: disable=import-outside-toplevel
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
    import soundfile  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    samples, rate = soundfile.read(utterance['wav'], dtype='float64')
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    tokens, seconds, cut_start, cut_end, report = align_utterance(utterance, mean_seconds, samples, rate)
    frames = align.durations_to_frames(seconds, FRAME_SECONDS, tokens)
    frame_count = sum(frames)

    # Resample to 24 kHz, then cut to the frames
    resampled = soxr.resample(samples, rate, SAMPLE_RATE, quality='VHQ')
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
# Build
#

# Build build/nn/data/arctic/ and data/nn/arctic-split.json
def build(args):
    start_time = time.time()
    out_dir = os.path.join(DATA_DIR, 'arctic')
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for speaker in args.speakers:
        utterances = [utterance for utterance in arctic_utterances(speaker) if utterance['text'] is not None]
        if args.limit:
            utterances = utterances[:args.limit]
        mean_seconds = mean_label_seconds(utterances)
        settings = speaker_settings(speaker)
        tasks.extend((utterance, mean_seconds, settings) for utterance in utterances)
    with multiprocessing.Pool(args.jobs) as pool:
        results = pool.map(process_utterance, tasks, chunksize=4)

    # Concatenate
    tokens, durations, seconds, audio, frames, entries = [], [], [], [], [], []
    token_offset = frame_offset = 0
    with open(os.path.join(out_dir, 'alignment.jsonl'), 'w', encoding='utf-8') as alignment_file:
        for result in results:
            utterance, report = result['utterance'], result['report']
            rate = match_rate(report)
            reasons = []
            if rate < MIN_MATCH:
                reasons.append(f'phone match {rate:.3f} < {MIN_MATCH}')
            if report['worstWordMatch'] < MIN_WORD_MATCH:
                reasons.append(f'word "{report["worstWord"]}" match {report["worstWordMatch"]:.2f} < {MIN_WORD_MATCH}')
            if any(count == 0 for token, count in zip(result['tokens'], result['frames']) if token not in BOUNDARIES):
                reasons.append('a phone with no frames')
            frame_count = sum(result['frames'])
            entry = {
                'id': f'{utterance["speaker"]}/{utterance["sentence"]}',
                'speaker': utterance['speaker'],
                'sentence': utterance['sentence'],
                'split': sentence_split(utterance['sentence']),
                'text': utterance['text'],
                'include': not reasons,
                'reason': '; '.join(reasons),
                'tokenOffset': token_offset,
                'tokenCount': len(result['tokens']),
                'frameOffset': frame_offset,
                'frameCount': frame_count,
                'sampleOffset': frame_offset * HOP,
                'sampleCount': frame_count * HOP,
                'trimStart': round(result['cutStart'], 5),
                'worstWord': report['worstWord'],
                'worstWordMatch': report['worstWordMatch'],
                'match': {key: report[key] for key in ('phones', 'labelPhones', 'exact', 'near', 'substituted',
                                                        'deleted', 'inserted', 'pauseAtBoundary', 'pauseInsideWord',
                                                        'pausesRefined')}
            }
            entries.append(entry)
            tokens.append(np.asarray(report['ids'], dtype=np.int16))
            durations.append(np.asarray(result['frames'], dtype=np.int16))
            seconds.append(np.asarray(result['seconds'], dtype=np.float32))
            audio.append(result['audio'])
            frames.append(result['tracks'])
            token_offset += len(result['tokens'])
            frame_offset += frame_count
            alignment_file.write(json.dumps({'id': entry['id'], 'tokens': ' '.join(result['tokens']),
                                             'frames': result['frames'], 'edits': report['edits'],
                                             'cost': report['cost']}) + '\n')
    np.save(os.path.join(out_dir, 'tokens.npy'), np.concatenate(tokens))
    np.save(os.path.join(out_dir, 'durations.npy'), np.concatenate(durations))
    np.save(os.path.join(out_dir, 'seconds.npy'), np.concatenate(seconds))
    np.save(os.path.join(out_dir, 'audio.npy'), np.concatenate(audio))
    np.save(os.path.join(out_dir, 'frames.npy'), np.concatenate(frames))
    speakers = {}
    for index, speaker in enumerate(args.speakers):
        pitch_floor, pitch_ceiling, formant_ceiling = speaker_settings(speaker)
        speakers[speaker] = {'index': index, 'sex': SPEAKERS[speaker], 'corpus': 'CMU ARCTIC 0.95',
                             'sourceSampleRate': 16000, 'bandwidth': 8000, 'pitchFloor': pitch_floor,
                             'pitchCeiling': pitch_ceiling, 'formantCeiling': formant_ceiling}
    index = {
        'version': FORMAT_VERSION,
        'sampleRate': SAMPLE_RATE,
        'hop': HOP,
        'frameRate': SAMPLE_RATE / HOP,
        'channels': CHANNELS,
        'channelNotes': CHANNEL_NOTES,
        'vocab': vocabulary(),
        'maxEdgeSilence': MAX_EDGE_SILENCE,
        'minMatch': MIN_MATCH,
        'minWordMatch': MIN_WORD_MATCH,
        'speakers': speakers,
        'utterances': entries
    }
    with open(os.path.join(out_dir, 'index.json'), 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=1)
        file.write('\n')

    # The split record
    if not args.limit:
        write_split(entries)
    print_match_report(entries, results)
    print(f'Built {out_dir}: {len(entries)} utterances, {frame_offset / 100 / 3600:.2f} hours, '
          f'{time.time() - start_time:.0f} s')


# Write data/nn/arctic-split.json - the validation and test sentences, and the test texts
def write_split(entries):
    sentences = {}
    for entry in entries:
        sentences.setdefault(entry['sentence'], entry['text'])
    split = {
        'description': 'The CMU ARCTIC split of nn/data.py - by sentence, the same for every speaker: "test" and "val" '
                       'are held out (about 5% each, by a hash of the sentence id); the rest is training.',
        'val': sorted(sentence for sentence in sentences if sentence_split(sentence) == 'val'),
        'test': [{'sentence': sentence, 'text': sentences[sentence]}
                 for sentence in sorted(sentences) if sentence_split(sentence) == 'test']
    }
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    with open(SPLIT_PATH, 'w', encoding='utf-8') as file:
        file.write(json.dumps(split, indent=4) + '\n')


# Print the match report - rates per speaker, the commonest edits, and the excluded utterances
def print_match_report(entries, results):
    for speaker in sorted(set(entry['speaker'] for entry in entries)):
        mine = [(entry, result) for entry, result in zip(entries, results) if entry['speaker'] == speaker]
        totals = collections.Counter()
        exact_utterances = 0
        edits = collections.Counter()
        for entry, result in mine:
            totals.update({key: value for key, value in entry['match'].items()})
            if entry['match']['substituted'] + entry['match']['deleted'] + entry['match']['inserted'] == 0:
                exact_utterances += 1
            edits.update(result['report']['edits'])
        phones = totals['phones']
        print(f'{speaker}: {len(mine)} utterances, {exact_utterances} ({100 * exact_utterances / len(mine):.1f}%) '
              f'with identical phone sequences; phones {phones}: exact {100 * totals["exact"] / phones:.2f}%, '
              f'AH/AX {100 * totals["near"] / phones:.2f}%, substituted {100 * totals["substituted"] / phones:.2f}%, '
              f'ours missing from labels {100 * totals["deleted"] / phones:.2f}%, '
              f'label phones not ours {100 * totals["inserted"] / phones:.2f}%; pauses at boundaries '
              f'{totals["pauseAtBoundary"]}, inside words {totals["pauseInsideWord"]}, refined from the audio '
              f'{totals["pausesRefined"]}; '
              f'excluded {sum(1 for entry, _ in mine if not entry["include"])}')
        print('  commonest edits: ' + ', '.join(f'{edit} {count}' for edit, count in edits.most_common(20)))


#
# Loading (for the trainer)
#

# The memory-mapped data of a corpus directory (build/nn/data/<corpus>/)
class ArcticData:
    def __init__(self, directory=os.path.join(DATA_DIR, 'arctic')):
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


# A PyTorch dataset of utterances - each item a dict of tensors and the entry's id and speaker index
def arctic_dataset(split='train', speakers=None, data=None):
    import torch  # pylint: disable=import-outside-toplevel

    class ArcticDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = data or ArcticData()
            self.items = self.data.entries(split, speakers)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            entry = self.items[index]
            item = {key: torch.from_numpy(value) for key, value in self.data.arrays(entry).items()}
            item['speaker'] = self.data.index['speakers'][entry['speaker']]['index']
            item['id'] = entry['id']
            return item

    return ArcticDataset()


# Collate utterances into a padded batch - tokens (pad id 0), durations, audio, and frames padded with zeros, with
# their lengths ('tokenLengths', 'frameLengths'; the audio length is frameLengths x hop)
def collate(items):
    import torch  # pylint: disable=import-outside-toplevel
    pad = torch.nn.utils.rnn.pad_sequence
    return {
        'tokens': pad([item['tokens'] for item in items], batch_first=True, padding_value=0),
        'durations': pad([item['durations'] for item in items], batch_first=True),
        'seconds': pad([item['seconds'] for item in items], batch_first=True),
        'audio': pad([item['audio'] for item in items], batch_first=True),
        'frames': pad([item['frames'] for item in items], batch_first=True),
        'tokenLengths': torch.tensor([len(item['tokens']) for item in items]),
        'frameLengths': torch.tensor([len(item['frames']) for item in items]),
        'speaker': torch.tensor([item['speaker'] for item in items]),
        'id': [item['id'] for item in items]
    }


# Length-bucketed batches of dataset indices, each at most max_frames frames in all (shuffled with a seed)
def length_batches(dataset, max_frames, seed=0):
    rng = np.random.default_rng(seed)
    order = sorted(range(len(dataset)), key=lambda index: dataset.items[index]['frameCount'] + rng.random())
    batches, batch, longest = [], [], 0
    for index in order:
        length = dataset.items[index]['frameCount']
        if batch and max(longest, length) * (len(batch) + 1) > max_frames:
            batches.append(batch)
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        batches.append(batch)
    rng.shuffle(batches)
    return batches


#
# Token verification - nn/js/tokens.js against nn/frontend.py
#

# Tokenize texts with nn/js/tokens.js
def js_tokenize(texts):
    process = subprocess.run(['node', os.path.join(ROOT, 'nn', 'js', 'tokenizeLines.js')],
                             input=''.join(json.dumps(text) + '\n' for text in texts), capture_output=True,
                             text=True, check=True, cwd=ROOT)
    return [json.loads(line) for line in process.stdout.splitlines()]


# Texts exercising the front end's corners - numbers, decimals, abbreviations, quotes, unknown words, questions
EDGE_TEXTS = [
    '', '...', 'Hello!!', 'Mr. and Mrs. Smith vs. Dr. Jones, etc.', 'It cost 1,000,000 dollars - or 3.14159?',
    'Why not?', 'Who are you?', 'Is it 2026 yet?', '\u2018Quoted,\u2019 she said; \u201cdouble\u201d too: yes.',
    "Don't won't ''tis o'clock rock'n'roll", 'U.S.A. and the U.K.', 'Zyxwvut qqqq xxlly flibbertigibbets jumped.',
    'Caf\u00e9 na\u00efve \u00c9cole', '42nd 7th 1st', '0 00 007 100 101 1000000000001',
    'ALL CAPS SHOUTING!', 'tabs\tand\nnewlines\r\nhere'
]


# Verify nn/js/tokens.js and nn/frontend.py agree on every ARCTIC text (all eight speakers' prompts) and LibriTTS-R
# dev-clean's normalized texts
def verify_tokens(args):
    texts = set()
    for directory in sorted(os.listdir(ARCTIC_DIR)):
        prompts = os.path.join(ARCTIC_DIR, directory, 'etc', 'txt.done.data')
        if os.path.exists(prompts):
            texts.update(read_prompts(prompts).values())
    arctic_count = len(texts)
    texts.update(EDGE_TEXTS)
    if not args.arctic_only:
        for path in libritts_text_paths():
            with open(path, encoding='utf-8') as file:
                texts.add(file.read().strip())
    texts = sorted(texts)
    js_results = js_tokenize(texts)
    mismatches = 0
    for text, js_result in zip(texts, js_results):
        python_result = tokenize(text)
        if python_result != js_result:
            mismatches += 1
            if mismatches <= 5:
                print(f'MISMATCH: {text!r}\n  python: {python_result["tokens"]}\n  js:     {js_result["tokens"]}')
    tokens = sum(len(result['ids']) for result in js_results)
    print(f'verify-tokens: {len(texts)} texts ({arctic_count} ARCTIC, {len(EDGE_TEXTS)} edge cases, the rest LibriTTS-R), '
          f'{tokens} tokens: {len(texts) - mismatches} identical, {mismatches} different')
    return 1 if mismatches else 0


#
# Statistics
#

# Percentiles of values, rounded
def percentiles(values, points=(5, 25, 50, 75, 95)):
    if len(values) == 0:
        return {}
    return {f'p{point}': round(float(np.percentile(values, point)), 3) for point in points}


# Print and write the dataset statistics (build/nn/data/arctic/stats.json)
def stats(_args):
    data = ArcticData()
    vocab = data.index['vocab']
    word_id = vocab.index(WORD)
    end_ids = {vocab.index(token) for token in PHRASE_ENDS.values()}
    bos_id = vocab.index('<bos>')
    result = {'speakers': {}}
    for speaker in data.index['speakers']:
        speaker_stats = {}
        all_entries = data.entries(speakers=[speaker], include_excluded=True)
        for split in ('train', 'val', 'test', 'all'):
            entries = [entry for entry in all_entries if split == 'all' or entry['split'] == split]
            included = [entry for entry in entries if entry['include']]
            speaker_stats[split] = {
                'utterances': len(entries),
                'included': len(included),
                'hours': round(sum(entry['frameCount'] for entry in entries) / 100 / 3600, 3),
                'includedHours': round(sum(entry['frameCount'] for entry in included) / 100 / 3600, 3)
            }
        entries = data.entries(speakers=[speaker])
        token_counts = np.array([entry['tokenCount'] for entry in entries])
        frame_counts = np.array([entry['frameCount'] for entry in entries])
        phone_frames = collections.defaultdict(list)
        pauses = collections.defaultdict(list)
        speech_frames = 0
        f0_values = []
        voiced_frames = 0
        for entry in entries:
            arrays = data.arrays(entry)
            for token_id, frames in zip(arrays['tokens'], arrays['durations']):
                name = vocab[token_id]
                if token_id == word_id:
                    pauses['word boundary'].append(frames)
                elif token_id in end_ids:
                    pauses[name].append(frames)
                elif token_id == bos_id:
                    pauses['<bos>'].append(frames)
                else:
                    phone_frames[name.rstrip('012')].append(frames)
                    speech_frames += frames
            # The last phrase end carries the trailing silence - counted separately
            pauses[vocab[arrays['tokens'][-1]]].pop()
            pauses['trailing'].append(arrays['durations'][-1])
            f0 = arrays['frames'][:, CHANNELS.index('f0')]
            f0_values.append(f0[f0 > 0])
            voiced_frames += int(np.sum(f0 > 0))
        f0_all = np.concatenate(f0_values)
        all_phones = np.concatenate([np.asarray(values) for values in phone_frames.values()])
        speaker_stats['tokens'] = {'perUtterance': {'mean': round(float(token_counts.mean()), 1), 'max': int(token_counts.max()),
                                                    **percentiles(token_counts)},
                                   'total': int(token_counts.sum())}
        speaker_stats['frames'] = {'perUtterance': {'mean': round(float(frame_counts.mean()), 1), 'max': int(frame_counts.max()),
                                                    **percentiles(frame_counts)}}
        speaker_stats['speechHours'] = round(speech_frames / 100 / 3600, 3)
        speaker_stats['phoneRate'] = round(len(all_phones) / (speech_frames / 100), 2)
        speaker_stats['phoneFrames'] = {'all': {'mean': round(float(all_phones.mean()), 2), 'zero': int(np.sum(all_phones == 0)),
                                                'max': int(all_phones.max()), **percentiles(all_phones)}}
        for phone in sorted(phone_frames):
            values = np.asarray(phone_frames[phone])
            speaker_stats['phoneFrames'][phone] = {'count': len(values), 'mean': round(float(values.mean()), 2),
                                                   **percentiles(values, (10, 50, 90))}
        speaker_stats['pauseFrames'] = {}
        for name in sorted(pauses):
            values = np.asarray(pauses[name])
            if len(values):
                speaker_stats['pauseFrames'][name] = {'count': len(values), 'withPause': int(np.sum(values > 0)),
                                                      'mean': round(float(values.mean()), 2),
                                                      'meanWhenPaused': round(float(values[values > 0].mean()), 2) if np.any(values > 0) else 0,
                                                      **percentiles(values, (50, 90, 99))}
        speaker_stats['f0'] = {'voicedShare': round(float(voiced_frames / frame_counts.sum()), 3), **percentiles(f0_all)}
        result['speakers'][speaker] = speaker_stats
    with open(os.path.join(data.directory, 'stats.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
        file.write('\n')

    # Print the summary
    for speaker, speaker_stats in result['speakers'].items():
        print(f'{speaker}:')
        for split in ('all', 'train', 'val', 'test'):
            item = speaker_stats[split]
            print(f'  {split:5s} {item["utterances"]:5d} utterances ({item["included"]} included) '
                  f'{item["hours"]:.3f} h ({item["includedHours"]:.3f} h included)')
        print(f'  speech (phones, no pauses) {speaker_stats["speechHours"]:.3f} h, {speaker_stats["phoneRate"]} phones/s')
        print(f'  tokens per utterance {speaker_stats["tokens"]["perUtterance"]}, total {speaker_stats["tokens"]["total"]}')
        print(f'  frames per utterance {speaker_stats["frames"]["perUtterance"]}')
        print(f'  phone frames {speaker_stats["phoneFrames"]["all"]}')
        for name, item in speaker_stats['pauseFrames'].items():
            print(f'  {name:14s} {item}')
        print(f'  F0 {speaker_stats["f0"]}')
    print(f'Wrote {os.path.join(data.directory, "stats.json")}')


#
# Listening proxy - UTMOS of originals and of the processed audio
#

# Score a few test utterances per speaker, original and processed, with tools/listeners.py
def listen(args):
    import soundfile  # pylint: disable=import-outside-toplevel
    data = ArcticData()
    out_dir = os.path.join(ROOT, 'build', 'nn', 'listen')
    os.makedirs(out_dir, exist_ok=True)
    pairs = []
    for speaker in data.index['speakers']:
        for entry in data.entries('test', [speaker])[:args.count]:
            processed = os.path.join(out_dir, f'{speaker}_{entry["sentence"]}_24k.wav')
            soundfile.write(processed, data.arrays(entry)['audio'], SAMPLE_RATE, subtype='PCM_16')
            original = os.path.join(ARCTIC_DIR, f'cmu_us_{speaker}_arctic', 'wav', entry['sentence'] + '.wav')
            pairs.append((speaker, original, processed))
    files = [path for _, original, processed in pairs for path in (original, processed)]
    python = os.path.join(ROOT, 'build', 'venv-listeners', 'bin', 'python')
    process = subprocess.run([python, '-I', os.path.join(ROOT, 'tools', 'listeners.py'), '--no-embedding', '--no-emotion',
                              *files], capture_output=True, text=True, check=True, cwd=ROOT)
    scores = {}
    for line in process.stdout.splitlines():
        record = json.loads(line)
        scores[record['file']] = record['mos']
    for speaker in data.index['speakers']:
        original = [scores[original] for name, original, _ in pairs if name == speaker]
        processed = [scores[processed] for name, _, processed in pairs if name == speaker]
        print(f'{speaker}: UTMOS of {len(original)} test utterances - original 16 kHz {np.mean(original):.2f} '
              f'(min {np.min(original):.2f}), processed 24 kHz {np.mean(processed):.2f} (min {np.min(processed):.2f})')


#
# LibriTTS-R dev-clean survey (the multi-speaker path)
#

LIBRITTS_DIR = os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R')
LIBRITTS_LABEL_DIR = os.path.join(CORPORA_DIR, 'libritts_label', 'lab')


# The LibriTTS-R dev-clean normalized text files
def libritts_text_paths():
    paths = []
    base = os.path.join(LIBRITTS_DIR, 'dev-clean')
    if os.path.isdir(base):
        for dirpath, _dirnames, filenames in os.walk(base):
            paths.extend(os.path.join(dirpath, name) for name in filenames if name.endswith('.normalized.txt'))
    return sorted(paths)


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


# Survey one LibriTTS utterance - (speaker, seconds, report, ER F3 ratio inputs) or None
def libritts_utterance(path):
    import soundfile  # pylint: disable=import-outside-toplevel
    utterance_id = os.path.basename(path)[:-len('.normalized.txt')]
    speaker, chapter = utterance_id.split('_')[:2]
    wav = path[:-len('.normalized.txt')] + '.wav'
    lab = os.path.join(LIBRITTS_LABEL_DIR, 'phone', 'dev-clean', speaker, chapter, utterance_id + '.lab')
    seconds = soundfile.info(wav).duration
    if not os.path.exists(lab):
        return {'speaker': speaker, 'seconds': seconds, 'report': None}
    with open(path, encoding='utf-8') as file:
        text = file.read().strip()
    labels = read_mfa(lab)
    tokens = tokenize(text)['tokens']
    try:
        _, report = align.assign_durations(tokens, [(label, end - start) for label, start, end in labels], {})
    except ValueError:
        report = None
    spn = sum(1 for label, _, _ in labels if label == 'spn')
    return {'speaker': speaker, 'seconds': seconds, 'report': report, 'spn': spn, 'wav': wav,
            'er': [(start, end) for label, start, end in labels if label in ('ER1', 'ER2') and end - start >= 0.06],
            'vowels': [(start, end) for label, start, end in labels
                       if label[:-1] in ('IY', 'IH', 'EH', 'AE', 'AA', 'AO', 'UH', 'UW', 'EY', 'OW')
                       and label[-1] in '12' and end - start >= 0.06]}


# Rhoticity of a speaker's speech - the median F3 of stressed ER over the median F3 of other stressed vowels (the
# NURSE vowel's low F3 is the hallmark of a rhotic, North American accent: about 0.65-0.8; non-rhotic English, about 0.9+)
def rhoticity(task):
    import parselmouth  # pylint: disable=import-outside-toplevel
    sex, utterances = task
    ceiling = 5500 if sex == 'female' else 5000
    er_f3, vowel_f3 = [], []
    for utterance in utterances:
        if not utterance['er']:
            continue
        sound = parselmouth.Sound(utterance['wav'])
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling)
        for spans, values in ((utterance['er'], er_f3), (utterance['vowels'], vowel_f3)):
            for start, end in spans:
                value = formant.get_value_at_time(3, (start + end) / 2)
                if np.isfinite(value):
                    values.append(value)
    if len(er_f3) < 5 or len(vowel_f3) < 20:
        return None, len(er_f3)
    return float(np.median(er_f3) / np.median(vowel_f3)), len(er_f3)


# Survey LibriTTS-R dev-clean - alignment format, phone mapping match rates, hours per speaker, and rhoticity
def libritts(args):
    paths = libritts_text_paths()
    with multiprocessing.Pool(args.jobs) as pool:
        results = pool.map(libritts_utterance, paths, chunksize=16)
    speakers_meta = {}
    with open(os.path.join(LIBRITTS_DIR, 'speakers.tsv'), encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if fields[0].isdigit():
                speakers_meta[fields[0]] = {'sex': {'F': 'female', 'M': 'male'}.get(fields[1]),
                                            'name': fields[3] if len(fields) > 3 else ''}
    by_speaker = collections.defaultdict(list)
    for result in results:
        by_speaker[result['speaker']].append(result)
    with multiprocessing.Pool(args.jobs) as pool:
        speaker_ids = sorted(by_speaker, key=int)
        rhotic = pool.map(rhoticity, [(speakers_meta.get(speaker, {}).get('sex'), [item for item in by_speaker[speaker] if item['report']])
                                      for speaker in speaker_ids])
    totals = collections.Counter()
    edits = collections.Counter()
    summary = []
    for speaker, (ratio, er_count) in zip(speaker_ids, rhotic):
        items = by_speaker[speaker]
        aligned = [item for item in items if item['report']]
        speaker_totals = collections.Counter()
        for item in aligned:
            speaker_totals.update({key: value for key, value in item['report'].items() if isinstance(value, int)})
            edits.update(item['report']['edits'])
        totals.update(speaker_totals)
        rate = (speaker_totals['exact'] + speaker_totals['near']) / max(1, speaker_totals['phones'])
        summary.append({'speaker': speaker, 'sex': speakers_meta.get(speaker, {}).get('sex'),
                        'name': speakers_meta.get(speaker, {}).get('name'), 'utterances': len(items),
                        'aligned': len(aligned), 'hours': round(sum(item['seconds'] for item in items) / 3600, 3),
                        'match': round(rate, 4), 'spn': sum(item['spn'] for item in aligned),
                        'erF3Ratio': None if ratio is None else round(ratio, 3), 'erCount': er_count})
    for row in summary:
        print(f'{row["speaker"]:>5s} {row["sex"] or "?":6s} {row["utterances"]:4d} utts {row["hours"]:.3f} h '
              f'match {100 * row["match"]:.1f}% spn {row["spn"]:3d} ER/V F3 {row["erF3Ratio"]} (n={row["erCount"]}) {row["name"]}')
    phones = totals['phones']
    print(f'all: {len(results)} utterances, {sum(row["hours"] for row in summary):.2f} h, {len(summary)} speakers; '
          f'phones {phones}: exact {100 * totals["exact"] / phones:.2f}%, AH/AX {100 * totals["near"] / phones:.2f}%, '
          f'substituted {100 * totals["substituted"] / phones:.2f}%, deleted {100 * totals["deleted"] / phones:.2f}%, '
          f'inserted {100 * totals["inserted"] / phones:.2f}%')
    print('  commonest edits: ' + ', '.join(f'{edit} {count}' for edit, count in edits.most_common(25)))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'libritts-survey.json'), 'w', encoding='utf-8') as file:
        json.dump({'speakers': summary, 'edits': edits.most_common(100), 'totals': dict(totals)}, file, indent=1)
        file.write('\n')


# The command line
def main(argv=None):
    parser = argparse.ArgumentParser(prog='nn.data', description='Build and inspect the neural-network training data')
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build', help='build build/nn/data/arctic/')
    build_parser.add_argument('--speakers', nargs='+', default=list(SPEAKERS), choices=list(SPEAKERS))
    build_parser.add_argument('--limit', type=int, help='at most this many utterances per speaker (a quick test)')
    build_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    verify_parser = commands.add_parser('verify-tokens', help='check nn/js/tokens.js against nn/frontend.py')
    verify_parser.add_argument('--arctic-only', action='store_true', help='only the ARCTIC texts')
    commands.add_parser('stats', help='dataset statistics')
    listen_parser = commands.add_parser('listen', help='UTMOS of original and processed test utterances')
    listen_parser.add_argument('-n', '--count', type=int, default=8, help='utterances per speaker')
    libritts_parser = commands.add_parser('libritts', help='survey LibriTTS-R dev-clean and its MFA alignments')
    libritts_parser.add_argument('-j', '--jobs', type=int, default=None, help='worker processes (default: all cores)')
    args = parser.parse_args(argv)
    handler = {'build': build, 'verify-tokens': verify_tokens, 'stats': stats, 'listen': listen, 'libritts': libritts}
    return handler[args.command](args) or 0


if __name__ == '__main__':
    sys.exit(main())
