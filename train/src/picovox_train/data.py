# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training data's featurization and format (tts.py builds it - the training speech): each utterance's tokens (our
front end's, frontend.py), each token's duration in frames (pauses included, from its CTC alignment - ctc.py, align.py),
the audio (24 kHz, 16-bit), and auxiliary frame tracks (Praat F0, voicing, pitch strength, harmonicity, energy,
formants F1-F4 and their bandwidths).

A corpus directory (build/nn/data/<corpus>/) - concatenated arrays and an index of offsets into them:

- index.json - format constants, the channel names, the speakers, and the utterances: {id, speaker, sentence,
  split, text, include, reason, tokenOffset, tokenCount, frameOffset, frameCount, sampleOffset, sampleCount, trimStart}.
  Every array is memory-mapped by CorpusData.
- tokens.npy (int16) - token ids (vocab.json)
- durations.npy (int16) - each token's duration in frames; an utterance's durations sum to its frame count
- seconds.npy (float32) - each token's duration in seconds, before quantization
- audio.npy (int16) - 24 kHz audio, frameCount x 240 samples per utterance (sampleOffset = frameOffset x 240)
- frames.npy (float32, [frames, channels]) - the auxiliary tracks at 100 frames per second (frame k is the 10 ms of
  samples [240k, 240k + 240), its center at (k + 0.5) x 10 ms)
- alignment.jsonl - per utterance, the alignment's tokens, frames, and the recognizer's greedy phones (diagnostics)

(The human corpora this module read - CMU ARCTIC's and LibriTTS-R's labels - are in git history.)
"""

import json
import os

import numpy as np

from . import ROOT


DATA_DIR = os.environ.get('NN_DATA_DIR', os.path.join(ROOT, 'build', 'nn', 'data'))

# The format
SAMPLE_RATE = 24000
HOP = 240                                  # samples per frame - 10 ms
FRAME_SECONDS = HOP / SAMPLE_RATE
MAX_EDGE_SILENCE = 0.2                     # seconds of leading and trailing silence kept
FORMAT_VERSION = 1
CHANNELS = ['f0', 'voiced', 'strength', 'hnr', 'energy', 'f1', 'f2', 'f3', 'f4', 'b1', 'b2', 'b3', 'b4']
CHANNEL_NOTES = {
    'f0': 'Praat autocorrelation F0 (Hz), 0 where unvoiced - the voice\'s two-pass range (index.json\'s speakers)',
    'voiced': '1 where Praat finds the frame voiced, else 0',
    'strength': 'Praat pitch strength (autocorrelation peak, 0-1) of the selected candidate, 0 where unvoiced',
    'hnr': 'Harmonics-to-noise ratio (dB, cross-correlation), clipped to [-20, 40] (-20 in silence)',
    'energy': 'RMS level (dB re full scale) of a 20 ms Hann window centered on the frame, floor -100',
    'f1': 'Burg formant 1 (Hz) at the speaker\'s formant ceiling (5 formants, 25 ms window), 0 where undefined',
    'b1': 'Bandwidth of formant 1 (Hz), 0 where undefined'
}


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


# Convert a raw array file to .npy (in chunks), removing the raw file
def raw_to_npy(raw_path, npy_path, dtype, shape):
    output = np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=shape)
    flat = output.reshape(-1)
    chunk = 1 << 26
    with open(raw_path, 'rb') as file:
        offset = 0
        while True:
            buffer = file.read(chunk * np.dtype(dtype).itemsize)
            if not buffer:
                break
            values = np.frombuffer(buffer, dtype=dtype)
            flat[offset:offset + len(values)] = values
            offset += len(values)
    output.flush()
    del output
    os.remove(raw_path)


#
# Loading (for the trainer)
#

class CorpusData:
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
