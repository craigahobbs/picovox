# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
Human speech targets - measure a speaker's acoustic and prosodic statistics, identically for a human corpus speaker and
for the synthesizer, and compare two speakers.

    python -I tools/targets.py extract --arctic bdl [-o data/targets/arctic-bdl.json]
    python -I tools/targets.py extract --libritts 84 [-o data/targets/libritts-84.json]
    python -I tools/targets.py extract --dir out/synthstats/male --name synth-male [-o out/synthstats/male.json]
    python -I tools/targets.py all                       # every downloaded corpus speaker -> data/targets/
    python -I tools/targets.py compare ours.json target.json [--json]

A speaker directory in ARCTIC layout (--dir) holds wav/<id>.wav, lab/<id>.lab (xwaves phone labels - ARPAbet, "pau"
for silence, "ax" for the reduced vowel), and etc/txt.done.data (the prompts). All audio is resampled to 16 kHz (ARCTIC's
rate) before analysis, so every speaker is measured with the same bandwidth.

Measures (Praat, via parselmouth, plus numpy): F0 distribution and contour (two-pass speaker-adaptive pitch range),
per-vowel formants F1-F4 at 20/50/80% of the vowel (speaker-adaptive formant ceiling), per-phone durations, speaking
rate, pauses, the long-term average spectrum, voice quality (H1*-H2*, H1-A1, H1-A3, HNR, jitter, shimmer, CPP),
intensity dynamics and phone-class levels, and fricative spectral centers of gravity.
"""

import argparse
import csv
import glob
import json
import math
import multiprocessing
import os
import re
import sys
import time

import numpy as np
import parselmouth
from parselmouth.praat import call
import scipy.signal
import soundfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA_DIR = os.path.join(ROOT, 'data', 'corpora')
TARGETS_DIR = os.path.join(ROOT, 'data', 'targets')
SAMPLE_RATE = 16000
VERSION = 1

# Phone classes (ARPAbet, stress removed; AX is the reduced vowel - ARCTIC's "ax", CMUdict's AH0)
VOWELS = ['IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AO', 'OW', 'UH', 'UW', 'AH', 'AX', 'ER', 'AY', 'AW', 'OY']
PHONE_CLASSES = {
    'vowel': VOWELS,
    'stop-voiceless': ['P', 'T', 'K'],
    'stop-voiced': ['B', 'D', 'G'],
    'affricate': ['CH', 'JH'],
    'fricative-voiceless': ['F', 'TH', 'S', 'SH', 'HH'],
    'fricative-voiced': ['V', 'DH', 'Z', 'ZH'],
    'nasal': ['M', 'N', 'NG'],
    'liquid': ['L', 'R'],
    'glide': ['W', 'Y']
}
PHONE_CLASS = {phone: name for name, phones in PHONE_CLASSES.items() for phone in phones}
FRICATIVES = ['F', 'TH', 'S', 'SH', 'HH', 'V', 'DH', 'Z', 'ZH']
DIPHTHONGS = ['EY', 'AY', 'AW', 'OY', 'OW']
PAUSE = '_'

# Long-term average spectrum bands (Hz) - third-octave-like up to ARCTIC's 8 kHz Nyquist, and coarse summary bands
LTAS_EDGES = [63, 125, 250, 400, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
LTAS_SUMMARY = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]

# Formant ceiling candidates (Hz) by sex - the one giving the least within-vowel formant variance is chosen
FORMANT_CEILINGS = {'male': [4000, 4250, 4500, 4750, 5000, 5250, 5500], 'female': [4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250]}

# Minimum vowel durations (s) for formant and voice-quality measurement
MIN_VOWEL_FORMANT = 0.04
MIN_VOWEL_VOICE = 0.06


#
# Corpus loading
#

# Read an ARCTIC/Festival prompt file - {id: text}
def read_prompts(path):
    prompts = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8', errors='replace') as file:
            for line in file:
                match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)\s*$', line.strip())
                if match:
                    prompts[match.group(1)] = match.group(2)
    return prompts


# Normalize a corpus phone label to stressless ARPAbet (AX for the reduced vowel), PAUSE for silence, or None for an
# unknown (spoken noise) label
def normalize_phone(label):
    label = label.strip()
    if label in ('', 'pau', 'ssil', 'sil', 'sp', 'h#'):
        return PAUSE
    if label in ('spn',):
        return None
    upper = label.upper()
    if upper == 'AH0':
        return 'AX'
    phone = upper.rstrip('012')
    return phone if phone in PHONE_CLASS else None


# Read an xwaves (ARCTIC) label file - [(start, end, phone)] in seconds
def read_xwaves_labels(path):
    phones = []
    start = 0.0
    in_body = False
    with open(path, encoding='utf-8', errors='replace') as file:
        for line in file:
            if not in_body:
                in_body = line.strip() == '#'
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            end = float(fields[0])
            phone = normalize_phone(fields[2])
            if phone is None:
                return None
            phones.append((start, end, phone))
            start = end
    return phones


# Read a tab-separated (MFA/LibriTTSLabel) label file - [(start, end, phone)] in seconds, gaps filled with pauses
def read_tsv_labels(path):
    phones = []
    previous_end = 0.0
    with open(path, encoding='utf-8', errors='replace') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 2:
                continue
            start, end = float(fields[0]), float(fields[1])
            phone = normalize_phone(fields[2] if len(fields) > 2 else '')
            if phone is None:
                return None
            if start > previous_end + 1e-3:
                phones.append((previous_end, start, PAUSE))
            phones.append((start, end, phone))
            previous_end = end
    return phones


# Merge adjacent pauses into one
def merge_pauses(phones):
    merged = []
    for start, end, phone in phones:
        if merged and phone == PAUSE and merged[-1][2] == PAUSE:
            merged[-1] = (merged[-1][0], end, PAUSE)
        else:
            merged.append((start, end, phone))
    return merged


# The utterances of an ARCTIC-layout directory - [{id, wav, lab, text}]
def arctic_layout_utterances(directory):
    prompts = read_prompts(os.path.join(directory, 'etc', 'txt.done.data'))
    utterances = []
    for wav in sorted(glob.glob(os.path.join(directory, 'wav', '*.wav'))):
        utterance_id = os.path.splitext(os.path.basename(wav))[0]
        lab = os.path.join(directory, 'lab', utterance_id + '.lab')
        utterances.append({'id': utterance_id, 'wav': wav, 'lab': lab if os.path.exists(lab) else None,
                           'labFormat': 'xwaves', 'text': prompts.get(utterance_id, '')})
    return utterances


# The utterances of a LibriTTS-R dev-clean speaker - [{id, wav, lab, text}], with the LibriTTSLabel alignments if present
def libritts_utterances(speaker):
    base = os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'dev-clean', speaker)
    labels = os.path.join(CORPORA_DIR, 'libritts_label', 'lab', 'phone', 'dev-clean', speaker)
    utterances = []
    for wav in sorted(glob.glob(os.path.join(base, '*', '*.wav'))):
        utterance_id = os.path.splitext(os.path.basename(wav))[0]
        chapter = os.path.basename(os.path.dirname(wav))
        lab = os.path.join(labels, chapter, utterance_id + '.lab')
        text_path = os.path.splitext(wav)[0] + '.normalized.txt'
        text = ''
        if os.path.exists(text_path):
            with open(text_path, encoding='utf-8', errors='replace') as file:
                text = file.read().strip()
        utterances.append({'id': utterance_id, 'wav': wav, 'lab': lab if os.path.exists(lab) else None,
                           'labFormat': 'tsv', 'text': text})
    return utterances


# The LibriTTS-R speakers' metadata - {speaker: {sex, name, subset}}
def libritts_speakers():
    speakers = {}
    for path in glob.glob(os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'speakers.tsv')) + \
            glob.glob(os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'SPEAKERS.txt')):
        with open(path, encoding='utf-8', errors='replace') as file:
            for row in csv.reader(file, delimiter='\t'):
                if row and not row[0].startswith(('READER', ';')) and len(row) >= 3:
                    sex = {'F': 'female', 'M': 'male'}.get(row[1].strip(), None)
                    speakers[row[0].strip()] = {'sex': sex, 'subset': row[2].strip(), 'name': row[3].strip() if len(row) > 3 else ''}
    return speakers


# Choose a deterministic, evenly spread subset of utterances
def subset(utterances, limit):
    if not limit or len(utterances) <= limit:
        return utterances
    step = len(utterances) / limit
    return [utterances[int(ix * step)] for ix in range(limit)]


# Read a WAV as mono float64 at the analysis sample rate
def read_audio(path):
    samples, rate = soundfile.read(path, dtype='float64', always_2d=True)
    samples = samples[:, 0]
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        samples = scipy.signal.resample_poly(samples, SAMPLE_RATE // divisor, int(rate) // divisor)
    peak = np.max(np.abs(samples)) if len(samples) else 0
    return samples / peak * 0.9 if peak > 0 else samples


#
# Per-utterance analysis
#

# Hawks and Miller (1995) formant bandwidth (Hz) from the formant frequency and F0
def hawks_miller_bandwidth(frequency, f0):
    scale = 1 + 0.25 * (f0 - 132) / 88
    if frequency < 500:
        coefficients = [165.327516, -6.73636734e-1, 1.80874446e-3, -4.52201682e-6, 7.49514000e-9, -4.70219241e-12]
    else:
        coefficients = [15.8146139, 8.10159009e-2, -9.79728215e-5, 5.28725064e-8, -1.07099364e-11, 7.91528509e-16]
    return scale * sum(coefficient * frequency ** power for power, coefficient in enumerate(coefficients))


# Iseli and Alwan (2004) correction (dB) - the boost of a harmonic at frequency by the formants, to subtract
def formant_boost(frequency, formants, f0):
    total = 0.0
    for formant in formants:
        if not formant or math.isnan(formant):
            continue
        bandwidth = hawks_miller_bandwidth(formant, f0)
        radius = math.exp(-math.pi * bandwidth / SAMPLE_RATE)
        omega_formant = 2 * math.pi * formant / SAMPLE_RATE
        omega = 2 * math.pi * frequency / SAMPLE_RATE
        above = radius ** 2 + 1 - 2 * radius * math.cos(omega_formant + omega)
        below = radius ** 2 + 1 - 2 * radius * math.cos(omega_formant - omega)
        numerator = radius ** 2 + 1 - 2 * radius * math.cos(omega_formant)
        total += 20 * math.log10(numerator) - 10 * math.log10(above) - 10 * math.log10(below)
    return total


# Harmonic and formant-peak amplitudes (dB) of a window of samples - H1, H2, H4, A1, A2, A3
def harmonic_amplitudes(samples, center, f0, formants):
    width = int(round(3 * SAMPLE_RATE / f0))
    start = int(round(center * SAMPLE_RATE)) - width // 2
    if start < 0 or start + width > len(samples) or width < 16:
        return None
    window = samples[start:start + width] * np.hanning(width)
    size = 8192
    spectrum = 20 * np.log10(np.abs(np.fft.rfft(window, size)) + 1e-12)
    bin_hz = SAMPLE_RATE / size

    # The peak within a frequency range
    def peak(low, high):
        low_bin, high_bin = max(1, int(low / bin_hz)), min(len(spectrum) - 1, int(high / bin_hz) + 1)
        return float(np.max(spectrum[low_bin:high_bin])) if high_bin > low_bin else None

    amplitudes = {f'h{harmonic}': peak(harmonic * f0 * 0.9, harmonic * f0 * 1.1) for harmonic in (1, 2, 4)}
    for ix, formant in enumerate(formants[:3]):
        amplitudes[f'a{ix + 1}'] = None if formant is None or math.isnan(formant) else peak(formant * 0.9, formant * 1.1)
    return amplitudes


# Mean of a parselmouth Formant's value over times (ignoring undefined)
def formant_value(formant, number, times):
    values = [formant.get_value_at_time(number, sample_time) for sample_time in times]
    values = [value for value in values if not math.isnan(value)]
    return float(np.median(values)) if values else None


# Analyze one utterance - the raw measurements the speaker statistics are pooled from
def analyze_utterance(task):
    utterance, pitch_floor, pitch_ceiling, ceilings = task
    samples = read_audio(utterance['wav'])
    if utterance['lab'] is None:
        phones = None
    elif utterance['labFormat'] == 'xwaves':
        phones = read_xwaves_labels(utterance['lab'])
    else:
        phones = read_tsv_labels(utterance['lab'])
    if phones is not None:
        phones = merge_pauses([(start, min(end, len(samples) / SAMPLE_RATE), phone) for start, end, phone in phones
                               if start < len(samples) / SAMPLE_RATE])
    sound = parselmouth.Sound(samples, SAMPLE_RATE)
    result = {'id': utterance['id'], 'text': utterance['text'], 'seconds': len(samples) / SAMPLE_RATE, 'phones': phones}

    # Pitch (autocorrelation, 10 ms) - the voiced frames' times and F0
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    f0 = pitch.selected_array['frequency']
    times = pitch.xs()
    voiced = f0 > 0
    result['f0Times'] = times[voiced].round(3).tolist()
    result['f0'] = f0[voiced].round(2).tolist()

    # Speech regions - from the labels, else from intensity (frames within 30 dB of the utterance's 95th percentile)
    intensity = sound.to_intensity(minimum_pitch=max(pitch_floor, 50), time_step=0.01)
    intensity_db = intensity.values[0]
    intensity_times = intensity.xs()
    if phones is not None:
        speech_mask = np.zeros(len(intensity_times), dtype=bool)
        for start, end, phone in phones:
            if phone != PAUSE:
                speech_mask |= (intensity_times >= start) & (intensity_times < end)
    else:
        speech_mask = intensity_db > np.percentile(intensity_db, 95) - 30
    result['intensity'] = intensity_db[speech_mask].round(2).tolist()

    # Pauses, found acoustically (labels often absorb pauses into the neighboring phones) - runs of at least 100 ms more
    # than 25 dB below the utterance's 95th-percentile intensity, between the first and last speech
    speech_times = intensity_times[speech_mask]
    result['speechSpan'] = [float(speech_times[0]), float(speech_times[-1])] if len(speech_times) else None
    if phones is not None:
        spoken = [(start, end) for start, end, phone in phones if phone != PAUSE]
        result['speechSpan'] = [spoken[0][0], spoken[-1][1]] if spoken else None
    pauses = []
    if result['speechSpan']:
        quiet = intensity_db < np.percentile(intensity_db, 95) - 25
        run_start = None
        for time_value, is_quiet in zip(intensity_times, quiet):
            inside = result['speechSpan'][0] <= time_value <= result['speechSpan'][1]
            if is_quiet and inside:
                run_start = time_value if run_start is None else run_start
            else:
                if run_start is not None and time_value - run_start >= 0.1:
                    pauses.append([round(float(run_start), 3), round(float(time_value), 3)])
                run_start = None
    result['pauses'] = pauses

    # Long-term average spectrum of the speech frames (power per FFT bin, summed), and of the vowel frames
    frame, hop = 512, 160
    window = np.hanning(frame)
    starts = np.arange(0, max(1, len(samples) - frame), hop)
    centers = (starts + frame / 2) / SAMPLE_RATE
    if phones is not None:
        frame_speech = np.zeros(len(starts), dtype=bool)
        frame_vowel = np.zeros(len(starts), dtype=bool)
        for start, end, phone in phones:
            inside = (centers >= start) & (centers < end)
            if phone != PAUSE:
                frame_speech |= inside
            if phone in VOWELS:
                frame_vowel |= inside
    else:
        frame_speech = np.interp(centers, intensity_times, speech_mask.astype(float)) > 0.5
        frame_vowel = np.zeros(len(starts), dtype=bool)
    power_speech = np.zeros(frame // 2 + 1)
    power_vowel = np.zeros(frame // 2 + 1)
    for ix, start in enumerate(starts):
        if frame_speech[ix] or frame_vowel[ix]:
            chunk = samples[start:start + frame]
            if len(chunk) < frame:
                continue
            power = np.abs(np.fft.rfft(chunk * window)) ** 2
            if frame_speech[ix]:
                power_speech += power
            if frame_vowel[ix]:
                power_vowel += power
    result['ltasSpeech'] = power_speech.tolist()
    result['ltasSpeechFrames'] = int(np.sum(frame_speech))
    result['ltasVowel'] = power_vowel.tolist()
    result['ltasVowelFrames'] = int(np.sum(frame_vowel))

    # Utterance-level voice quality - jitter, shimmer, and HNR over the whole utterance
    point_process = call(sound, 'To PointProcess (periodic, cc)', pitch_floor, pitch_ceiling)
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=pitch_floor)
    harmonicity_values = harmonicity.values[0]
    harmonicity_times = harmonicity.xs()
    cepstrogram = call(sound, 'To PowerCepstrogram', 60, 0.002, 5000, 50)
    if phones is None:
        result['jitter'] = [call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)]
        result['shimmer'] = [call([sound, point_process], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)]
        voiced_hnr = harmonicity_values[harmonicity_values > -100]
        result['hnr'] = [float(np.mean(voiced_hnr))] if len(voiced_hnr) else []
        result['cpps'] = [call(cepstrogram, 'Get CPPS', 'no', 0.02, 0.0005, 60, 330, 0.05, 'parabolic', 0.001, 0.05,
                               'Straight', 'Robust')]
        return result

    # Per-phone measurements - duration, intensity, and (vowels) formants at each ceiling and voice quality
    # Phrase-final vowels - no other vowel before the next (labeled or acoustic) pause, or the end
    final = [False] * len(phones)
    later_vowel = False
    for ix in range(len(phones) - 1, -1, -1):
        phone = phones[ix][2]
        pause_after = any(phones[ix][0] < pause_start <= phones[ix][1] + 0.03 for pause_start, _ in pauses)
        if phone == PAUSE or pause_after:
            later_vowel = False
        if phone in VOWELS:
            final[ix] = not later_vowel
            later_vowel = True

    tokens = []
    formants_by_ceiling = {ceiling: sound.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                                                          maximum_formant=ceiling, window_length=0.025,
                                                          pre_emphasis_from=50) for ceiling in ceilings}
    for ix, (start, end, phone) in enumerate(phones):
        if phone == PAUSE:
            continue
        duration = end - start
        inside = (intensity_times >= start) & (intensity_times < end)
        token = {'phone': phone, 'start': start, 'dur': duration,
                 'db': float(10 * np.log10(np.mean(10 ** (intensity_db[inside] / 10)))) if np.any(inside) else None,
                 'final': final[ix]}
        if phone in VOWELS and duration >= MIN_VOWEL_FORMANT:
            token['formants'] = {}
            for ceiling, formant in formants_by_ceiling.items():
                measures = {}
                for label, fraction in (('onset', 0.2), ('mid', 0.5), ('offset', 0.8)):
                    center = start + fraction * duration
                    sample_times = [center - 0.01, center - 0.005, center, center + 0.005, center + 0.01] \
                        if label == 'mid' else [center]
                    measures[label] = [formant_value(formant, number, sample_times) for number in (1, 2, 3, 4)]
                token['formants'][ceiling] = measures
        if phone in VOWELS and duration >= MIN_VOWEL_VOICE:
            middle = start + duration / 2
            vowel_f0 = pitch.get_value_at_time(middle)
            if not math.isnan(vowel_f0):
                token['f0'] = vowel_f0
                token['jitter'] = call(point_process, 'Get jitter (local)', start, end, 0.0001, 0.02, 1.3)
                token['shimmer'] = call([sound, point_process], 'Get shimmer (local)', start, end, 0.0001, 0.02, 1.3, 1.6)
                hnr_inside = (harmonicity_times >= start) & (harmonicity_times < end) & (harmonicity_values > -100)
                token['hnr'] = float(np.mean(harmonicity_values[hnr_inside])) if np.any(hnr_inside) else None
                cepstrum = call(cepstrogram, 'To PowerCepstrum (slice)', middle)
                token['cpp'] = call(cepstrum, 'Get peak prominence', 60, 330, 'parabolic', 0.001, 0.05, 'Straight', 'Robust')
                token['harmonics'] = {}
                for ceiling, formant in formants_by_ceiling.items():
                    mid_formants = [formant_value(formant, number, [middle]) for number in (1, 2, 3)]
                    token['harmonics'][ceiling] = (mid_formants, harmonic_amplitudes(samples, middle, vowel_f0, mid_formants))
        if phone in FRICATIVES and duration >= 0.04:
            # The spectral center of gravity of the middle of the fricative (a 20 ms Hann window above 500 Hz)
            middle = int((start + duration / 2) * SAMPLE_RATE)
            half = int(0.01 * SAMPLE_RATE)
            chunk = samples[max(0, middle - half):middle + half]
            if len(chunk) == 2 * half:
                power = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), 1024)) ** 2
                freqs = np.fft.rfftfreq(1024, 1 / SAMPLE_RATE)
                keep = freqs >= 500
                if np.sum(power[keep]) > 0:
                    token['cog'] = float(np.sum(freqs[keep] * power[keep]) / np.sum(power[keep]))
        tokens.append(token)
    result['tokens'] = tokens
    return result


# A first pitch pass over an utterance (60-600 Hz) - the voiced F0 values, for the speaker's pitch range
def first_pass_pitch(path):
    sound = parselmouth.Sound(read_audio(path), SAMPLE_RATE)
    f0 = sound.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600).selected_array['frequency']
    return f0[f0 > 0].tolist()


#
# Speaker statistics
#

# Semitones relative to a reference frequency
def semitones(values, reference):
    return 12 * np.log2(np.asarray(values, dtype=float) / reference)


# Rounded float (None for missing or NaN)
def rounded(value, digits=3):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else round(value, digits)


# Median of the defined values (None if none)
def median(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    return float(np.median(values)) if values else None


# Mean of the defined values (None if none)
def mean(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    return float(np.mean(values)) if values else None


# Linear regression slope of y on x (None for fewer than 3 points or no spread)
def slope(x_values, y_values):
    if len(x_values) < 3 or np.ptp(x_values) == 0:
        return None
    return float(np.polyfit(x_values, y_values, 1)[0])


# The F0 statistics of the utterances
def f0_statistics(results):
    all_f0 = np.concatenate([np.asarray(result['f0']) for result in results if result['f0']])
    reference = float(np.median(all_f0))
    all_st = semitones(all_f0, reference)
    utterance_slopes, utterance_ranges, finals, onsets, velocities = [], [], [], [], []
    for result in results:
        if len(result['f0']) < 10:
            continue
        times = np.asarray(result['f0Times'])
        values = semitones(result['f0'], reference)
        utterance_median = float(np.median(values))
        utterance_slopes.append(slope(times, values))
        utterance_ranges.append(float(np.percentile(values, 95) - np.percentile(values, 5)))
        onsets.append(float(np.median(values[times <= times[0] + 0.15]) - utterance_median))

        # The final fall - the last 150 ms of voicing before the end of speech, relative to the utterance median
        end_time = times[-1]
        if result['phones']:
            speech_end = max((end for _, end, phone in result['phones'] if phone != PAUSE), default=end_time)
            end_time = min(end_time, speech_end)
        final = values[(times <= end_time) & (times >= end_time - 0.15)]
        if len(final) and result['text'].rstrip().endswith('.'):
            finals.append(float(np.median(final) - utterance_median))

        # F0 velocity - the median absolute rate of change between consecutive voiced frames (semitones per second)
        steps = np.diff(times)
        contiguous = steps < 0.015
        if np.any(contiguous):
            velocities.append(float(np.median(np.abs(np.diff(values)[contiguous]) / steps[contiguous])))
    return {
        'medianHz': rounded(reference, 1),
        'meanHz': rounded(np.mean(all_f0), 1),
        'p5Hz': rounded(np.percentile(all_f0, 5), 1),
        'p95Hz': rounded(np.percentile(all_f0, 95), 1),
        'sdSt': rounded(np.std(all_st), 2),
        'rangeSt': rounded(np.percentile(all_st, 95) - np.percentile(all_st, 5), 2),
        'utteranceRangeSt': rounded(median(utterance_ranges), 2),
        'declinationStPerS': rounded(median(utterance_slopes), 2),
        'onsetSt': rounded(median(onsets), 2),
        'finalFallSt': rounded(median(finals), 2),
        'velocityStPerS': rounded(median(velocities), 1),
        'voicedFrames': int(len(all_f0))
    }


# The formant ceiling with the least within-vowel variance of log F1 and log F2 (Escudero et al., 2009)
def choose_formant_ceiling(tokens, ceilings):
    best, best_score = ceilings[0], None
    scores = {}
    for ceiling in ceilings:
        score = 0.0
        for vowel in VOWELS:
            values = [token['formants'][ceiling]['mid'] for token in tokens if token['phone'] == vowel and 'formants' in token]
            values = [value for value in values if value[0] and value[1]]
            if len(values) >= 5:
                score += np.var(np.log([value[0] for value in values])) + np.var(np.log([value[1] for value in values]))
        scores[ceiling] = round(float(score), 4)
        if best_score is None or score < best_score:
            best, best_score = ceiling, score
    return best, scores


# Per-vowel formant statistics at the chosen ceiling, the vowel space, and the formant dispersion
def formant_statistics(tokens, ceiling):
    vowels = {}
    for vowel in VOWELS:
        measured = [token['formants'][ceiling] for token in tokens if token['phone'] == vowel and 'formants' in token]
        if not measured:
            continue
        entry = {'n': len(measured)}
        for number in range(4):
            values = [measure['mid'][number] for measure in measured]
            entry[f'f{number + 1}'] = rounded(median(values), 0)
            defined = [value for value in values if value is not None]
            entry[f'f{number + 1}Sd'] = rounded(np.std(defined), 0) if len(defined) > 1 else None
        entry['onset'] = [rounded(median([measure['onset'][number] for measure in measured]), 0) for number in range(2)]
        entry['offset'] = [rounded(median([measure['offset'][number] for measure in measured]), 0) for number in range(2)]
        vowels[vowel] = entry

    # The vowel quadrilateral (IY, AE, AA, UW) area in kHz squared, and the formant dispersion and apparent tract length
    corners = [vowels.get(vowel) for vowel in ('IY', 'AE', 'AA', 'UW')]
    area = None
    if all(corner and corner['f1'] and corner['f2'] for corner in corners):
        points = [(corner['f2'] / 1000, corner['f1'] / 1000) for corner in corners]
        area = 0.5 * abs(sum(points[ix][0] * points[(ix + 1) % 4][1] - points[(ix + 1) % 4][0] * points[ix][1] for ix in range(4)))
    dispersions = [(entry['f4'] - entry['f1']) / 3 for entry in vowels.values() if entry['f4'] and entry['f1'] and entry['n'] >= 5]
    dispersion = median(dispersions)
    return vowels, {'quadrilateralKHz2': rounded(area, 4), 'dispersionHz': rounded(dispersion, 0),
                    'tractLengthCm': rounded(35000 / (2 * dispersion), 2) if dispersion else None}


# Per-phone and per-class durations, final lengthening, speaking rate, and pauses
def timing_statistics(results):
    durations = {}
    for result in results:
        for token in result['tokens']:
            durations.setdefault(token['phone'], []).append(token['dur'])
    phone_durations = {phone: {'n': len(values), 'mean': rounded(np.mean(values) * 1000, 1),
                               'median': rounded(np.median(values) * 1000, 1)} for phone, values in sorted(durations.items())}
    classes = {}
    for name, members in PHONE_CLASSES.items():
        values = [value for phone in members for value in durations.get(phone, [])]
        if values:
            classes[name] = {'n': len(values), 'mean': rounded(np.mean(values) * 1000, 1)}

    # Phrase-final lengthening - final vowels' durations relative to their vowel's overall mean
    phone_means = {phone: np.mean(values) for phone, values in durations.items()}
    final_ratios = [token['dur'] / phone_means[token['phone']] for result in results for token in result['tokens']
                    if token['phone'] in VOWELS and token['final']]
    nonfinal_ratios = [token['dur'] / phone_means[token['phone']] for result in results for token in result['tokens']
                       if token['phone'] in VOWELS and not token['final']]

    # Rates - over the time from the first to the last speech, less the acoustic pauses
    articulation_seconds, vowel_count, phone_count = 0.0, 0, 0
    for result in results:
        if result['speechSpan']:
            articulation_seconds += result['speechSpan'][1] - result['speechSpan'][0] - sum(end - start for start, end in result['pauses'])
            spoken = [phone for _, _, phone in result['phones'] if phone != PAUSE]
            phone_count += len(spoken)
            vowel_count += sum(phone in VOWELS for phone in spoken)
    rates = {
        'phonesPerS': rounded(phone_count / articulation_seconds, 2),
        'syllablesPerS': rounded(vowel_count / articulation_seconds, 2)
    }
    lengthening = rounded(np.mean(final_ratios) / np.mean(nonfinal_ratios), 3) if final_ratios and nonfinal_ratios else None
    return {'phones': phone_durations, 'classes': classes, 'finalLengthening': lengthening}, rates


# Pause statistics (acoustic pauses between the first and last speech of each utterance)
def pause_statistics(results):
    pauses = [end - start for result in results for start, end in result['pauses']]
    span_seconds = sum(result['speechSpan'][1] - result['speechSpan'][0] for result in results if result['speechSpan'])
    commas = sum(len(re.findall(r'[,;:]', result['text'])) for result in results)
    sentences = sum(max(0, len(re.findall(r'[.!?]+(?=\s+\S)', result['text']))) for result in results)
    return {
        'count': len(pauses),
        'perUtterance': rounded(len(pauses) / len(results), 2),
        'perMinute': rounded(60 * len(pauses) / span_seconds, 2) if span_seconds else None,
        'perPunctuation': rounded(len(pauses) / (commas + sentences), 2) if commas + sentences else None,
        'fraction': rounded(sum(pauses) / span_seconds, 3) if span_seconds else None,
        'meanMs': rounded(np.mean(pauses) * 1000, 0) if pauses else None,
        'medianMs': rounded(np.median(pauses) * 1000, 0) if pauses else None,
        'p90Ms': rounded(np.percentile(pauses, 90) * 1000, 0) if pauses else None
    }


# Band levels (dB, relative to the total power) and tilt measures of a summed power spectrum
def ltas_statistics(power, frames):
    if frames == 0:
        return None
    power = np.asarray(power) / frames
    freqs = np.fft.rfftfreq(2 * (len(power) - 1), 1 / SAMPLE_RATE)
    total = np.sum(power[freqs >= LTAS_EDGES[0]])

    # The level of a frequency band (dB re total)
    def band(low, high):
        inside = (freqs >= low) & (freqs < high)
        return float(10 * np.log10(np.sum(power[inside]) / total + 1e-20))

    # The level of the strongest bin in a frequency band (dB)
    def peak(low, high):
        inside = (freqs >= low) & (freqs < high)
        return float(10 * np.log10(np.max(power[inside]) + 1e-20))

    bands = [[low, high, rounded(band(low, high), 2)] for low, high in zip(LTAS_EDGES[:-1], LTAS_EDGES[1:])]
    centers = np.sqrt([low * high for low, high, _ in bands])
    levels = [level - 10 * math.log10(high - low) for low, high, level in bands]
    fit = (centers >= 500) & (centers <= 6000)
    return {
        'bands': bands,
        'summary': {f'{low}-{high}': rounded(band(max(low, LTAS_EDGES[0]), high), 2) for low, high in LTAS_SUMMARY},
        'alphaRatioDb': rounded(band(1000, 5000) - band(50, 1000), 2),
        'hammarbergDb': rounded(peak(0, 2000) - peak(2000, 5000), 2),
        'slopeDbPerOctave': rounded(np.polyfit(np.log2(centers[fit]), np.asarray(levels)[fit], 1)[0], 2)
    }


# Voice quality statistics over the vowel tokens (corrected H1*-H2* and H1*-A3* use the chosen ceiling's formants)
def voice_statistics(tokens, ceiling):
    h1h2, h1h2c, h1a1, h1a3, h1a3c = [], [], [], [], []
    for token in tokens:
        if 'harmonics' not in token:
            continue
        formants, amplitudes = token['harmonics'][ceiling]
        if amplitudes is None or amplitudes['h1'] is None or amplitudes['h2'] is None:
            continue
        f0 = token['f0']
        h1, h2 = amplitudes['h1'], amplitudes['h2']
        h1_corrected = h1 - formant_boost(f0, formants[:2], f0)
        h2_corrected = h2 - formant_boost(2 * f0, formants[:2], f0)
        h1h2.append(h1 - h2)
        h1h2c.append(h1_corrected - h2_corrected)
        if amplitudes.get('a1') is not None:
            h1a1.append(h1 - amplitudes['a1'])
        if amplitudes.get('a3') is not None:
            h1a3.append(h1 - amplitudes['a3'])
            if formants[2]:
                a3_corrected = amplitudes['a3'] - formant_boost(formants[2], formants[:2], f0)
                h1a3c.append(h1_corrected - a3_corrected)
    return {
        'h1h2Db': rounded(median(h1h2), 2),
        'h1h2cDb': rounded(median(h1h2c), 2),
        'h1a1Db': rounded(median(h1a1), 2),
        'h1a3Db': rounded(median(h1a3), 2),
        'h1a3cDb': rounded(median(h1a3c), 2),
        'hnrDb': rounded(median([token.get('hnr') for token in tokens]), 2),
        'jitterPercent': rounded(100 * median([token.get('jitter') for token in tokens]), 3),
        'shimmerPercent': rounded(100 * median([token.get('shimmer') for token in tokens]), 2),
        'cppDb': rounded(median([token.get('cpp') for token in tokens]), 2),
        'tokens': len(h1h2)
    }


# Intensity dynamics - speech-frame spread, phone-class levels relative to vowels, reduced vowels, and declination
def intensity_statistics(results):
    spreads, ranges, declinations = [], [], []
    class_offsets = {name: [] for name in PHONE_CLASSES if name != 'vowel'}
    reduced = []
    for result in results:
        values = np.asarray(result['intensity'])
        if len(values) < 10:
            continue
        spreads.append(float(np.std(values)))
        ranges.append(float(np.percentile(values, 95) - np.percentile(values, 5)))
        tokens = [token for token in result.get('tokens', []) if token['db'] is not None]
        full_vowels = [token for token in tokens if token['phone'] in VOWELS and token['phone'] != 'AX']
        if len(full_vowels) < 3:
            continue
        vowel_level = float(np.mean([token['db'] for token in full_vowels]))
        for token in tokens:
            name = PHONE_CLASS[token['phone']]
            if name in class_offsets:
                class_offsets[name].append(token['db'] - vowel_level)
            elif token['phone'] == 'AX':
                reduced.append(token['db'] - vowel_level)
        declinations.append(slope([token['start'] for token in full_vowels], [token['db'] for token in full_vowels]))
    return {
        'sdDb': rounded(median(spreads), 2),
        'rangeDb': rounded(median(ranges), 2),
        'declinationDbPerS': rounded(median(declinations), 2),
        'reducedVowelDb': rounded(median(reduced), 2),
        'classesDb': {name: rounded(median(values), 2) for name, values in class_offsets.items() if values}
    }


# Pool a speaker's per-utterance results into the speaker's statistics
def speaker_statistics(name, results, sex, pitch_range, ceilings, source):
    labeled = [result for result in results if result['phones'] is not None]
    statistics = {
        'version': VERSION,
        'speaker': name,
        'source': source,
        'sex': sex,
        'utterances': len(results),
        'labeledUtterances': len(labeled),
        'audioSeconds': rounded(sum(result['seconds'] for result in results), 1),
        'analysis': {'sampleRate': SAMPLE_RATE, 'pitchFloor': rounded(pitch_range[0], 1), 'pitchCeiling': rounded(pitch_range[1], 1)},
        'f0': f0_statistics(results)
    }
    speech_power = np.sum([result['ltasSpeech'] for result in results], axis=0)
    statistics['ltas'] = ltas_statistics(speech_power, sum(result['ltasSpeechFrames'] for result in results))
    statistics['intensity'] = intensity_statistics(results)
    statistics['pauses'] = pause_statistics(results)
    if labeled:
        tokens = [token for result in labeled for token in result['tokens']]
        ceiling, scores = choose_formant_ceiling(tokens, ceilings)
        statistics['analysis']['formantCeiling'] = ceiling
        statistics['analysis']['formantCeilingScores'] = scores
        statistics['formants'], statistics['vowelSpace'] = formant_statistics(tokens, ceiling)
        statistics['durations'], statistics['rate'] = timing_statistics(labeled)
        vowel_power = np.sum([result['ltasVowel'] for result in labeled], axis=0)
        statistics['ltasVowel'] = ltas_statistics(vowel_power, sum(result['ltasVowelFrames'] for result in labeled))
        statistics['voice'] = voice_statistics(tokens, ceiling)
        cogs = {}
        for token in tokens:
            if 'cog' in token:
                cogs.setdefault(token['phone'], []).append(token['cog'])
        statistics['fricativeCogHz'] = {phone: {'n': len(values), 'cog': rounded(np.median(values), 0)}
                                        for phone, values in sorted(cogs.items())}
    else:
        statistics['voice'] = {
            'hnrDb': rounded(median([value for result in results for value in result['hnr']]), 2),
            'jitterPercent': rounded(100 * median([value for result in results for value in result['jitter']]), 3),
            'shimmerPercent': rounded(100 * median([value for result in results for value in result['shimmer']]), 2),
            'cppsDb': rounded(median([value for result in results for value in result['cpps']]), 2),
            'note': 'utterance-level (no phone labels)'
        }
    return statistics


# Extract a speaker's statistics from its utterances (in parallel)
def extract(name, utterances, sex=None, source=None, jobs=None, limit=None, ceiling=None):
    utterances = subset(utterances, limit)
    if not utterances:
        raise ValueError(f'{name}: no utterances')
    start = time.time()
    with multiprocessing.Pool(jobs) as pool:
        # Two-pass pitch range (Hirst, 2011) - floor 0.75 x the first quartile, ceiling 1.5 x the third quartile
        sample = subset(utterances, 120)
        first_f0 = np.concatenate([np.asarray(values) for values in pool.map(first_pass_pitch, [item['wav'] for item in sample])])
        quartile1, quartile3 = np.percentile(first_f0, 25), np.percentile(first_f0, 75)
        pitch_range = (max(40.0, 0.75 * quartile1), min(800.0, 1.5 * quartile3))
        if sex is None:
            sex = 'male' if np.median(first_f0) < 160 else 'female'
        ceilings = [ceiling] if ceiling else FORMANT_CEILINGS[sex]
        tasks = [(utterance, pitch_range[0], pitch_range[1], ceilings) for utterance in utterances]
        results = pool.map(analyze_utterance, tasks, chunksize=4)
    statistics = speaker_statistics(name, results, sex, pitch_range, ceilings, source or {})
    statistics['analysis']['seconds'] = rounded(time.time() - start, 1)
    return statistics


#
# Comparison
#

# Bark (Traunmuller, 1990)
def bark(frequency):
    return 26.81 * frequency / (1960 + frequency) - 0.53


# Look up a dotted path in nested dictionaries (None if missing)
def lookup(data, path):
    for key in path.split('.'):
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


# Scalar comparison metrics - (group, label, path, kind, scale, weight); kind 'diff' compares values, 'log' compares
# log ratios (scale is a fraction), 'st' compares Hz in semitones
SCALAR_METRICS = [
    ('f0', 'F0 median', 'f0.medianHz', 'st', 1.0, 2.0),
    ('f0', 'F0 5th percentile', 'f0.p5Hz', 'st', 1.5, 0.5),
    ('f0', 'F0 95th percentile', 'f0.p95Hz', 'st', 1.5, 0.5),
    ('f0', 'F0 sd (st)', 'f0.sdSt', 'diff', 0.5, 1.0),
    ('f0', 'F0 range 5-95% (st)', 'f0.rangeSt', 'diff', 2.0, 0.5),
    ('f0', 'Utterance range (st)', 'f0.utteranceRangeSt', 'diff', 2.0, 1.0),
    ('f0', 'Declination (st/s)', 'f0.declinationStPerS', 'diff', 1.0, 0.5),
    ('f0', 'Onset re median (st)', 'f0.onsetSt', 'diff', 1.5, 0.5),
    ('f0', 'Final fall (st)', 'f0.finalFallSt', 'diff', 1.5, 1.0),
    ('f0', 'F0 velocity (st/s)', 'f0.velocityStPerS', 'log', 0.25, 1.0),
    ('vowel space', 'Quadrilateral area (kHz^2)', 'vowelSpace.quadrilateralKHz2', 'log', 0.2, 1.0),
    ('vowel space', 'Formant dispersion (Hz)', 'vowelSpace.dispersionHz', 'log', 0.05, 1.0),
    ('timing', 'Articulation rate (syll/s)', 'rate.syllablesPerS', 'log', 0.08, 2.0),
    ('timing', 'Phones per second', 'rate.phonesPerS', 'log', 0.08, 0.5),
    ('timing', 'Final lengthening ratio', 'durations.finalLengthening', 'diff', 0.2, 1.0),
    ('pauses', 'Pauses per punctuation mark', 'pauses.perPunctuation', 'diff', 0.3, 0.5),
    ('pauses', 'Pauses per minute', 'pauses.perMinute', 'log', 0.3, 0.5),
    ('pauses', 'Pause fraction', 'pauses.fraction', 'diff', 0.03, 0.5),
    ('pauses', 'Pause mean (ms)', 'pauses.meanMs', 'log', 0.3, 0.5),
    ('spectrum', 'Alpha ratio (dB)', 'ltas.alphaRatioDb', 'diff', 3.0, 1.0),
    ('spectrum', 'Hammarberg index (dB)', 'ltas.hammarbergDb', 'diff', 3.0, 0.5),
    ('spectrum', 'LTAS slope 0.5-6k (dB/oct)', 'ltas.slopeDbPerOctave', 'diff', 1.5, 1.0),
    ('spectrum', 'Vowel LTAS slope (dB/oct)', 'ltasVowel.slopeDbPerOctave', 'diff', 1.5, 0.5),
    ('voice', 'H1*-H2* (dB)', 'voice.h1h2cDb', 'diff', 3.0, 1.0),
    ('voice', 'H1-H2 (dB)', 'voice.h1h2Db', 'diff', 3.0, 0.5),
    ('voice', 'H1-A1 (dB)', 'voice.h1a1Db', 'diff', 4.0, 0.5),
    ('voice', 'H1-A3 (dB)', 'voice.h1a3Db', 'diff', 4.0, 0.5),
    ('voice', 'H1*-A3* (dB)', 'voice.h1a3cDb', 'diff', 4.0, 0.5),
    ('voice', 'HNR (dB)', 'voice.hnrDb', 'diff', 3.0, 1.0),
    ('voice', 'Jitter (%)', 'voice.jitterPercent', 'log', 0.4, 0.5),
    ('voice', 'Shimmer (%)', 'voice.shimmerPercent', 'log', 0.4, 0.5),
    ('voice', 'CPP (dB)', 'voice.cppDb', 'diff', 2.0, 1.0),
    ('intensity', 'Intensity sd (dB)', 'intensity.sdDb', 'diff', 1.5, 1.0),
    ('intensity', 'Intensity range 5-95% (dB)', 'intensity.rangeDb', 'diff', 3.0, 0.5),
    ('intensity', 'Intensity declination (dB/s)', 'intensity.declinationDbPerS', 'diff', 1.0, 0.5),
    ('intensity', 'Reduced vowel re vowels (dB)', 'intensity.reducedVowelDb', 'diff', 2.0, 0.5)
]


# Compare two speakers' statistics - returns rows of (group, label, ours, target, difference, normalized, weight)
def compare_statistics(ours, target):
    rows = []

    # Add a comparison row
    def add(group, label, ours_value, target_value, kind, scale, weight):
        if ours_value is None or target_value is None:
            rows.append((group, label, ours_value, target_value, None, None, weight))
            return
        if kind == 'st':
            difference = 12 * math.log2(ours_value / target_value)
        elif kind == 'log':
            if ours_value <= 0 or target_value <= 0:
                rows.append((group, label, ours_value, target_value, None, None, weight))
                return
            difference = math.log(ours_value / target_value)
        else:
            difference = ours_value - target_value
        rows.append((group, label, ours_value, target_value, difference, difference / scale, weight))

    for group, label, path, kind, scale, weight in SCALAR_METRICS:
        add(group, label, lookup(ours, path), lookup(target, path), kind, scale, weight)

    # Per-vowel formants in Bark (weighted by token counts), and the vowel-inherent movement (offset - onset F2)
    for vowel in VOWELS:
        mine, theirs = lookup(ours, f'formants.{vowel}'), lookup(target, f'formants.{vowel}')
        if not mine or not theirs:
            continue
        count_weight = min(1.0, min(mine['n'], theirs['n']) / 30) / len(VOWELS)
        if vowel in DIPHTHONGS:
            # A diphthong's middle is a transition - compare its F1 and F2 at the onset (20%) and offset (80%) instead
            for point in ('onset', 'offset'):
                for number, scale in ((1, 0.3), (2, 0.3)):
                    mine_value, their_value = mine[point][number - 1], theirs[point][number - 1]
                    if mine_value and their_value:
                        difference = bark(mine_value) - bark(their_value)
                        rows.append(('formants', f'{vowel} F{number} {point} (Hz)', mine_value, their_value, difference,
                                     difference / scale, 3 * count_weight))
        for number, scale in ((1, 0.3), (2, 0.3), (3, 0.4)):
            key = f'f{number}'
            if mine[key] and theirs[key] and (number == 3 or vowel not in DIPHTHONGS):
                difference = bark(mine[key]) - bark(theirs[key])
                rows.append(('formants', f'{vowel} F{number} (Hz)', mine[key], theirs[key], difference, difference / scale,
                             6 * count_weight))
        if vowel not in DIPHTHONGS and None not in mine['onset'] + mine['offset'] + theirs['onset'] + theirs['offset']:
            ours_glide = bark(mine['offset'][1]) - bark(mine['onset'][1])
            target_glide = bark(theirs['offset'][1]) - bark(theirs['onset'][1])
            rows.append(('formants', f'{vowel} F2 glide (Bark)', round(ours_glide, 2), round(target_glide, 2),
                         ours_glide - target_glide, (ours_glide - target_glide) / 0.4, 2 * count_weight))

    # Durations by phone class and by phone - log ratios, rate-normalized (less the overall phone-rate ratio, which has
    # its own row), so they measure the timing pattern rather than the tempo
    ours_rate, target_rate = lookup(ours, 'rate.phonesPerS'), lookup(target, 'rate.phonesPerS')
    tempo = math.log(target_rate / ours_rate) if ours_rate and target_rate else 0.0

    # Add a rate-normalized duration row
    def add_duration(group, label, mine, theirs, scale, weight):
        difference = math.log(mine / theirs) - tempo
        rows.append((group, label, mine, theirs, difference, difference / scale, weight))

    for name in PHONE_CLASSES:
        mine, theirs = lookup(ours, f'durations.classes.{name}'), lookup(target, f'durations.classes.{name}')
        if mine and theirs and mine['mean'] and theirs['mean']:
            add_duration('durations', f'{name} (ms)', mine['mean'], theirs['mean'], 0.15, 0.5)
    mine_phones, their_phones = lookup(ours, 'durations.phones') or {}, lookup(target, 'durations.phones') or {}
    for phone in sorted(set(mine_phones) & set(their_phones)):
        weight = min(1.0, min(mine_phones[phone]['n'], their_phones[phone]['n']) / 50) * 3 / len(PHONE_CLASS)
        add_duration('phone durations', f'{phone} (ms)', mine_phones[phone]['mean'], their_phones[phone]['mean'], 0.2, weight)

    # Spectrum bands, phone-class levels, and fricative centers of gravity
    for (low, high, mine), (_, _, theirs) in zip(lookup(ours, 'ltas.bands') or [], lookup(target, 'ltas.bands') or []):
        add('spectrum', f'LTAS {low}-{high} Hz (dB)', mine, theirs, 'diff', 3.0, 3 / len(LTAS_EDGES))
    for name in PHONE_CLASSES:
        mine, theirs = lookup(ours, f'intensity.classesDb.{name}'), lookup(target, f'intensity.classesDb.{name}')
        if mine is not None and theirs is not None:
            add('intensity', f'{name} re vowels (dB)', mine, theirs, 'diff', 3.0, 0.4)
    for phone in FRICATIVES:
        mine, theirs = lookup(ours, f'fricativeCogHz.{phone}'), lookup(target, f'fricativeCogHz.{phone}')
        if mine and theirs:
            add('fricatives', f'{phone} center of gravity (Hz)', mine['cog'], theirs['cog'], 'log', 0.1, 0.3)
    return rows


# A speaker's measures on comparable scales - {label: (group, value, display value)}: Hz as semitones re 100 Hz,
# formants in Bark, rates and centers of gravity as logs, phone-class durations as logs of their multiple of the
# speaker's mean phone duration (rate-independent), and levels in dB
def population_measures(statistics):
    measures = {}
    for group, label, path, kind, _scale, _weight in SCALAR_METRICS:
        value = lookup(statistics, path)
        if value is None or (kind != 'diff' and value <= 0):
            continue
        scaled = 12 * math.log2(value / 100) if kind == 'st' else (math.log(value) if kind == 'log' else value)
        measures[label] = (group, scaled, value)
    for vowel, entry in (statistics.get('formants') or {}).items():
        if entry['n'] < 10:
            continue
        points = [('onset', 0), ('offset', 0)] if vowel in DIPHTHONGS else [('mid', 0)]
        for point, _unused in points:
            for number in (1, 2):
                value = entry[f'f{number}'] if point == 'mid' else entry[point][number - 1]
                if value:
                    suffix = '' if point == 'mid' else f' {point}'
                    measures[f'{vowel} F{number}{suffix} (Hz)'] = ('formants', bark(value), value)
    phones_per_second = lookup(statistics, 'rate.phonesPerS')
    for name, entry in (lookup(statistics, 'durations.classes') or {}).items():
        if phones_per_second and entry['mean']:
            measures[f'{name} duration (ms)'] = ('durations', math.log(entry['mean'] / 1000 * phones_per_second), entry['mean'])
    for name, value in (lookup(statistics, 'intensity.classesDb') or {}).items():
        if value is not None:
            measures[f'{name} re vowels (dB)'] = ('intensity', value, value)
    for phone, entry in (statistics.get('fricativeCogHz') or {}).items():
        if entry['cog'] and entry['n'] >= 10:
            measures[f'{phone} center of gravity (Hz)'] = ('fricatives', math.log(entry['cog']), entry['cog'])
    return measures


# Place a speaker within a population of human speakers - per measure, the population mean and sd and the speaker's
# z-score - returns rows (group, label, ours, mean, low, high, z, count), largest |z| first
def population_statistics(ours, population):
    mine = population_measures(ours)
    others = [population_measures(statistics) for statistics in population]
    rows = []
    for label, (group, value, display) in mine.items():
        values = [measures[label] for measures in others if label in measures]
        if len(values) < 3:
            continue
        scaled = np.array([entry[1] for entry in values])
        displays = [entry[2] for entry in values]
        spread = float(np.std(scaled, ddof=1))
        z_score = (value - float(np.mean(scaled))) / spread if spread > 0 else 0.0
        center = float(np.mean(displays))
        rows.append((group, label, display, center, min(displays), max(displays), z_score, len(values)))
    rows.sort(key=lambda row: -abs(row[6]))
    return rows


# Print a speaker's place in a population of human speakers
def print_population(ours, population, rows, limit):
    outside = [row for row in rows if abs(row[6]) > 2]
    print(f'{ours["speaker"]} vs {len(population)} human speakers ({", ".join(item["speaker"] for item in population)})')
    print(f'{len(outside)} of {len(rows)} measures are more than 2 sd from the human mean')
    print(f'{"group":12s} {"measure":34s} {"ours":>8s} {"mean":>8s} {"min":>8s} {"max":>8s} {"z":>6s}')
    for group, label, display, center, low, high, z_score, _count in rows[:limit]:
        print(f'{group:12s} {label:34s} {cell(display):>8s} {cell(center):>8s} {cell(low):>8s} {cell(high):>8s} {z_score:+6.1f}')
    by_group = {}
    for row in rows:
        by_group.setdefault(row[0], []).append(row[6] ** 2)
    print('RMS z by group: ' + ', '.join(f'{group} {math.sqrt(np.mean(values)):.2f}' for group, values in by_group.items()))
    print(f'RMS z: {math.sqrt(np.mean([row[6] ** 2 for row in rows])):.3f}')


# The weighted distance of comparison rows - the root weighted mean square of the normalized differences, overall
# and by group
def distance(rows):
    groups = {}
    total, total_weight = 0.0, 0.0
    for group, _label, _ours, _target, _difference, normalized, weight in rows:
        if normalized is None:
            continue
        sums = groups.setdefault(group, [0.0, 0.0])
        sums[0] += weight * normalized ** 2
        sums[1] += weight
        total += weight * normalized ** 2
        total_weight += weight
    by_group = {group: math.sqrt(sums[0] / sums[1]) for group, sums in groups.items() if sums[1]}
    return (math.sqrt(total / total_weight) if total_weight else None), by_group, total_weight


# Format a table cell value
def cell(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.3g}' if abs(value) < 10 else f'{value:.0f}'
    return str(value)


# Print a comparison of two statistics files, and the largest discrepancies (by contribution to the distance)
def print_comparison(ours, target, rows, top, brief=False):
    score, by_group, total_weight = distance(rows)
    print(f'{ours["speaker"]} vs {target["speaker"]}')
    if not brief:
        print('(diff: Hz in semitones, formants in Bark, log ratios for rates/durations (rate-normalized), else units; '
              'norm: diff / scale)')
        print(f'{"group":16s} {"measure":34s} {"ours":>9s} {"target":>9s} {"diff":>8s} {"norm":>7s}')
        for group, label, mine, theirs, difference, normalized, _weight in rows:
            print(f'{group:16s} {label:34s} {cell(mine):>9s} {cell(theirs):>9s} {cell(difference):>8s} {cell(normalized):>7s}')
        print()
    print('Distance by group: ' + ', '.join(f'{group} {value:.2f}' for group, value in by_group.items()))
    contributions = sorted((row for row in rows if row[5] is not None), key=lambda row: -row[6] * row[5] ** 2)
    print('Largest discrepancies (share of the squared distance):')
    for group, label, mine, theirs, _difference, normalized, weight in contributions[:top]:
        share = weight * normalized ** 2 / (score ** 2 * total_weight) if score else 0
        print(f'  {share * 100:5.1f}%  {group}: {label} - ours {cell(mine)}, target {cell(theirs)} ({normalized:+.1f} units)')
    print(f'Distance: {score:.3f}')


#
# Command line
#

# The corpus speakers available locally - [(name, utterance loader, sex, source)]
def corpus_speakers():
    speakers = []
    for directory in sorted(glob.glob(os.path.join(CORPORA_DIR, 'arctic', 'cmu_us_*_arctic'))):
        speaker = os.path.basename(directory).split('_')[2]
        speakers.append((f'arctic-{speaker}', directory, None, {'corpus': 'CMU ARCTIC', 'speaker': speaker}))
    metadata = libritts_speakers()
    for directory in sorted(glob.glob(os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R', 'dev-clean', '*'))):
        speaker = os.path.basename(directory)
        info = metadata.get(speaker, {})
        speakers.append((f'libritts-{speaker}', speaker, info.get('sex'),
                         {'corpus': 'LibriTTS-R dev-clean', 'speaker': speaker, 'reader': info.get('name')}))
    return speakers


# Load the utterances of a named corpus speaker
def speaker_utterances(name, location):
    if name.startswith('arctic-'):
        return arctic_layout_utterances(location)
    return libritts_utterances(location)


# Write statistics JSON
def write_json(statistics, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(statistics, indent=1)
    # Keep lists of numbers on one line
    text = re.sub(r'\[\s+([-0-9.e, \n]+?)\s+\]', lambda match: '[' + re.sub(r'\s+', ' ', match.group(1)) + ']', text)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text + '\n')
    print(f'{path} - {statistics["utterances"]} utterances, {statistics["audioSeconds"]} s, '
          f'{statistics["analysis"]["seconds"]} s analysis', file=sys.stderr)


# Command-line entry point
def main():
    parser = argparse.ArgumentParser(description='Human speech targets - extract and compare speaker statistics')
    commands = parser.add_subparsers(dest='command', required=True)
    extract_parser = commands.add_parser('extract', help="extract one speaker's statistics")
    source = extract_parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--arctic', metavar='SPEAKER', help='a CMU ARCTIC speaker (bdl, slt, ...)')
    source.add_argument('--libritts', metavar='SPEAKER', help='a LibriTTS-R dev-clean speaker id')
    source.add_argument('--dir', help='a directory in ARCTIC layout (wav/, lab/, etc/txt.done.data)')
    extract_parser.add_argument('--name', help='the speaker name (default from the source)')
    extract_parser.add_argument('--sex', choices=['male', 'female'], help='for the formant ceiling (default from F0)')
    extract_parser.add_argument('--ceiling', type=int, help='a fixed formant ceiling in Hz (default: speaker-adaptive)')
    extract_parser.add_argument('-o', '--output', help='the output JSON (default data/targets/<name>.json)')
    all_parser = commands.add_parser('all', help='extract every downloaded corpus speaker into data/targets/')
    all_parser.add_argument('--corpus', choices=['arctic', 'libritts'], help='only this corpus')
    all_parser.add_argument('--force', action='store_true', help='re-extract existing targets')
    all_parser.add_argument('--min-utterances', type=int, default=40, help='skip speakers with fewer utterances (default 40)')
    for command_parser in (extract_parser, all_parser):
        command_parser.add_argument('--limit', type=int, help='at most this many utterances (evenly spread)')
        command_parser.add_argument('-j', '--jobs', type=int, help='worker processes (default: CPU count)')
    compare_parser = commands.add_parser('compare', help='compare two statistics files')
    compare_parser.add_argument('ours')
    compare_parser.add_argument('target')
    compare_parser.add_argument('--top', type=int, default=12, help='the number of largest discrepancies to list')
    compare_parser.add_argument('--json', action='store_true', help='print the comparison as JSON')
    compare_parser.add_argument('--brief', action='store_true', help='print only the distances and largest discrepancies')
    population_parser = commands.add_parser('population', help='place a speaker among human speakers (z-scores)')
    population_parser.add_argument('ours')
    population_parser.add_argument('targets', nargs='*', help='target statistics (default: data/targets/ of the same sex, '
                                   'with phone labels)')
    population_parser.add_argument('--top', type=int, default=30, help='the number of measures to list')
    args = parser.parse_args()

    if args.command == 'compare':
        with open(args.ours, encoding='utf-8') as file:
            ours = json.load(file)
        with open(args.target, encoding='utf-8') as file:
            target = json.load(file)
        rows = compare_statistics(ours, target)
        if args.json:
            score, by_group, _ = distance(rows)
            print(json.dumps({'ours': ours['speaker'], 'target': target['speaker'], 'distance': score, 'groups': by_group,
                              'rows': [dict(zip(('group', 'measure', 'ours', 'target', 'difference', 'normalized', 'weight'), row))
                                       for row in rows]}, indent=1))
        else:
            print_comparison(ours, target, rows, args.top, args.brief)
    elif args.command == 'population':
        with open(args.ours, encoding='utf-8') as file:
            ours = json.load(file)
        population = []
        for path in args.targets or sorted(glob.glob(os.path.join(TARGETS_DIR, '*.json'))):
            with open(path, encoding='utf-8') as file:
                statistics = json.load(file)
            # Same-sex labeled speakers - a speaker whose F0 contradicts its metadata's sex is left out (LibriSpeech's
            # reader metadata has errors)
            f0_consistent = (statistics['f0']['medianHz'] < 165) == (statistics['sex'] == 'male')
            if args.targets or (statistics['sex'] == ours['sex'] and statistics['labeledUtterances'] >= 40 and
                                statistics['speaker'] != ours['speaker'] and f0_consistent):
                population.append(statistics)
        print_population(ours, population, population_statistics(ours, population), args.top)
    elif args.command == 'extract':
        if args.arctic:
            name = args.name or f'arctic-{args.arctic}'
            utterances = arctic_layout_utterances(os.path.join(CORPORA_DIR, 'arctic', f'cmu_us_{args.arctic}_arctic'))
            source = {'corpus': 'CMU ARCTIC', 'speaker': args.arctic}
            sex = args.sex
        elif args.libritts:
            name = args.name or f'libritts-{args.libritts}'
            utterances = libritts_utterances(args.libritts)
            info = libritts_speakers().get(args.libritts, {})
            source = {'corpus': 'LibriTTS-R dev-clean', 'speaker': args.libritts, 'reader': info.get('name')}
            sex = args.sex or info.get('sex')
        else:
            name = args.name or os.path.basename(os.path.normpath(args.dir))
            utterances = arctic_layout_utterances(args.dir)
            source = {'directory': os.path.relpath(os.path.abspath(args.dir), ROOT)}
            sex = args.sex
        statistics = extract(name, utterances, sex, source, args.jobs, args.limit, args.ceiling)
        write_json(statistics, args.output or os.path.join(TARGETS_DIR, f'{name}.json'))
    else:
        for name, location, sex, source in corpus_speakers():
            if args.corpus and not name.startswith(args.corpus):
                continue
            output = os.path.join(TARGETS_DIR, f'{name}.json')
            if os.path.exists(output) and not args.force:
                continue
            utterances = speaker_utterances(name, location)
            if len(utterances) < args.min_utterances:
                print(f'{name}: skipped - {len(utterances)} utterances', file=sys.stderr)
                continue
            statistics = extract(name, utterances, sex, source, args.jobs, args.limit)
            write_json(statistics, output)


if __name__ == '__main__':
    main()
