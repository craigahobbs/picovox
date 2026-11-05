# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Voice quality measures - each of the package's voices, synthesized, measured against its source: the training
speech's clips of its engine voice (tts.py - the voices' tinniness and warble, 2026-10-23):

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality measure --out build/quality \\
        [--per-voice 16] [--edits '{"add": {"shelf": -5}}'] [--model build/train/voiceModel.js] [-j 8]
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality report build/quality/results.json
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality gap --out build/gap \
        [--per-voice 40] [--model build/train/voiceModel.js] [-j 8]

measure renders each voice over the panel texts (tools/panel-texts.json) and over its held-out clips' texts (the
corpus's test, then val clips, to --per-voice) with the package (tools/renderLines.js - --edits are frame edits, for
an experiment; --model another voice model module, a trained one), and measures them and those clips (downloads/tts)
in voiced frames (Praat's pitch):

- bands - third-octave levels (dB re the voiced power from 90 Hz to 95% of the source's bandwidth), and their summaries:
  tilt (dB/octave, 300 Hz to 5 kHz), alpha (50 Hz-1 kHz less 1-5 kHz), hammarberg (the 0-2 kHz peak less the 2-5 kHz),
  presence (2-5 kHz), and air (5-8 kHz)
- hnr - Praat's harmonics-to-noise ratio (cross-correlation), and hnr2k - of the 2-5 kHz band
- crest2k - the 2-5 kHz band's envelope crest factor (30 ms windows) - how pulse-like the upper harmonics are
- f0Flutter - the pitch's 4-20 Hz movement (cents RMS) and f0Wobble - its 1.5-6 Hz movement, in voiced runs of 250 ms+
- variance, flux, and onset - how the spectrum moves (24 log bands, 100 Hz-10 kHz, every 5 ms, in the speech frames -
  within 35 dB of the loud ones): each band's level's standard deviation over time (dB, the bands' mean), the median
  RMS over bands of the level's change in 20 ms (dB), and the low band's (60 Hz-1 kHz) median rise from silence to
  within 3 dB of its peak (ms) - the over-smoothing of a regression-trained voice shows as less variance and flux and
  slower onsets (2.0's voices, against their Phonon source voices: variance -0.3 to -1.2 dB, flux -8 to -11%, onsets
  about 5 ms slower)

gap measures how far each voice is from its source in a pretrained speech model's features - the pinned wav2vec2
phoneme recognizer's (ctc.py) - over per_voice held-out clips of 3-9 s (the test split's, by id), each synthesized by
the package from its text (--model another voice model module) and its source clip, both trimmed of silence:

- fdL12 and fdL6 - each voice's Frechet distance between its synthesized clips' frames and its source clips' (layers 12
  and 6) - the gap's measure
- pairedL3 and pairedL12 - each clip's mean cosine distance to its source clip's frames along their DTW alignment (on the
  feature encoder's frames), averaged

The measure was chosen and validated in November 2026 (paper Section 13.9): of 20 candidates over 480 held-out clips, fd
on layer 12 had the largest gap to Phonon relative to the spread between trained seeds (about 400 to 1), with a second
take of the engine's own clips far closer than any voice of ours (the floor - every voice), each step of 8-6-5-4-bit
rounding farther, stage A alone (without the critic) farther, and the shipped 2.0 voice closer than the simpler recipe's
retrained voice in all twelve voices - as two blind A/B rounds (8 to 0) and the listener panel judged them. It compares
sets of clips of the same texts, so it suits evaluation, not a training loss; the paired distances are per clip.

report prints each voice's synthesized measures (its held-out texts, and the panel texts) less its source clips'. The
measures are guides, not verdicts: listeners heard the voices as tinny, and lowering the source
shelf fixed it while moving presence further below the recordings' (the energy isn't the cue), and phase dispersion cut
crest2k below the recordings' without helping - blind listening decides (tools/voicelab/). Formant and
band modulation and spectral peakiness were measured too and dropped - they didn't track what listeners heard.
"""

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys

from . import ROOT


# The measures' names (report's columns), and the third-octave band centers (Hz)
MEASURES = ['tilt', 'alpha', 'hammarberg', 'presence', 'air', 'hnr', 'hnr2k', 'crest2k', 'f0Flutter', 'f0Wobble', 'variance',
            'flux', 'onset']
BAND_CENTERS = [100 * 2 ** (index / 3) for index in range(22)]

# The package's audio bandwidth (Hz - its 24 kHz output)
VOICE_BANDWIDTH = 12000

# The analysis frames (samples) and Praat's pitch step (s)
FFT_SIZE = 1024
FFT_HOP = 256
PITCH_STEP = 0.005

# The temporal measures' frames (s), bands (log-spaced from 100 Hz to 10 kHz), speech floor (dB below the loud frames -
# the 95th percentile), and onset rule (frames below the onset level before it, frames to its peak, dB of the peak)
TEMPORAL_WINDOW = 0.02
TEMPORAL_HOP = 0.005
TEMPORAL_BANDS = 24
SPEECH_FLOOR = 35
ONSET_LEVEL = 30
ONSET_QUIET = 6
ONSET_PEAK = 20
ONSET_REACH = 3


# A centered moving average of a track (its length - the edges averaged over what's there)
def moving_average(values, width):
    import numpy  # pylint: disable=import-outside-toplevel
    padded = numpy.pad(values, width // 2, mode='edge')
    return numpy.convolve(padded, numpy.ones(width) / width, mode='same')[width // 2:width // 2 + len(values)]


# Runs of True of at least minimum - [(start, end)]
def voiced_runs(mask, minimum):
    runs, start = [], None
    for index, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    return runs


# The RMS of a track's band (its fast moving average less its slow one, in seconds) over voiced runs of 250 ms or more -
# None without one
def band_rms(track, voiced, rate, fast, slow):
    import numpy  # pylint: disable=import-outside-toplevel
    fast_width, slow_width = max(1, round(fast * rate)), max(1, round(slow * rate))
    parts = []
    for start, end in voiced_runs(voiced, int(0.25 * rate)):
        band = moving_average(track[start:end], fast_width) - moving_average(track[start:end], slow_width)
        parts.append(band[slow_width // 2:len(band) - slow_width // 2])
    parts = [part for part in parts if len(part)]
    return float(numpy.sqrt(numpy.mean(numpy.concatenate(parts) ** 2))) if parts else None


# Samples band-passed (FFT bins outside [low, high) zeroed)
def band_pass(samples, rate, low, high):
    import numpy  # pylint: disable=import-outside-toplevel
    spectrum = numpy.fft.rfft(samples)
    frequencies = numpy.fft.rfftfreq(len(samples), 1 / rate)
    spectrum[(frequencies < low) | (frequencies >= high)] = 0
    return numpy.fft.irfft(spectrum, n=len(samples))


# A signal's envelope (the analytic signal's magnitude)
def envelope(signal):
    import numpy  # pylint: disable=import-outside-toplevel
    count = len(signal)
    gain = numpy.zeros(count)
    gain[0] = 1
    gain[1:(count + 1) // 2] = 2
    if count % 2 == 0:
        gain[count // 2] = 1
    return numpy.abs(numpy.fft.ifft(numpy.fft.fft(signal) * gain))


# The voiced frames' spectral balance - {bands: {center: dB}, tilt, alpha, hammarberg, presence, air} (None without
# voiced frames)
def spectral_balance(samples, rate, times, voiced, bandwidth):
    import numpy  # pylint: disable=import-outside-toplevel
    window = numpy.hanning(FFT_SIZE)
    starts = numpy.arange(0, max(0, len(samples) - FFT_SIZE), FFT_HOP)
    nearest = numpy.clip(numpy.searchsorted(times, (starts + FFT_SIZE / 2) / rate), 0, len(times) - 1)
    power = numpy.zeros(FFT_SIZE // 2 + 1)
    for start in starts[voiced[nearest]]:
        power += numpy.abs(numpy.fft.rfft(samples[start:start + FFT_SIZE] * window)) ** 2
    frequencies = numpy.fft.rfftfreq(FFT_SIZE, 1 / rate)
    top = 0.95 * bandwidth
    total = power[(frequencies >= 90) & (frequencies <= top)].sum()
    if total <= 0:
        return {'bands': {}, **{name: None for name in ('tilt', 'alpha', 'hammarberg', 'presence', 'air')}}

    def level(low, high, peak=False):
        in_band = power[(frequencies >= low) & (frequencies < high)]
        return 10 * math.log10((in_band.max() if peak else in_band.sum()) / total + 1e-20)
    bands = {round(center): level(center * 2 ** (-1 / 6), center * 2 ** (1 / 6)) for center in BAND_CENTERS
             if center * 2 ** (1 / 6) <= top}
    centers = numpy.array(list(bands))
    fit = (centers >= 300) & (centers <= min(5000, top))
    return {'bands': bands, 'tilt': float(numpy.polyfit(numpy.log2(centers[fit]), numpy.array(list(bands.values()))[fit], 1)[0]),
            'alpha': level(50, 1000) - level(1000, 5000), 'hammarberg': level(0, 2000, True) - level(2000, 5000, True),
            'presence': level(2000, 5000), 'air': level(5000, min(8000, top))}


# The spectrum's movement over time - {variance, flux, onset} (each None for silence or under 40 ms; onset None
# without one)
def temporal(samples, rate):
    import numpy  # pylint: disable=import-outside-toplevel
    size, hop = round(TEMPORAL_WINDOW * rate), round(TEMPORAL_HOP * rate)
    if len(samples) < size + 4 * hop or not numpy.any(samples):
        return {'variance': None, 'flux': None, 'onset': None}
    frames = numpy.lib.stride_tricks.sliding_window_view(samples, size)[::hop] * numpy.hanning(size)
    fft_size = 1 << (size - 1).bit_length()
    power = numpy.abs(numpy.fft.rfft(frames, fft_size)) ** 2
    frequencies = numpy.fft.rfftfreq(fft_size, 1 / rate)
    band_index = numpy.digitize(frequencies, 100 * 100 ** (numpy.arange(TEMPORAL_BANDS + 1) / TEMPORAL_BANDS)) - 1
    bands = [band for band in range(TEMPORAL_BANDS) if (band_index == band).any()]
    band_power = numpy.stack([power[:, band_index == band].sum(axis=1) for band in bands], axis=1) + 1e-12
    levels = 10 * numpy.log10(band_power)
    total = 10 * numpy.log10(band_power.sum(axis=1))
    speech = total > numpy.percentile(total, 95) - SPEECH_FLOOR
    lag = round(0.02 / TEMPORAL_HOP)
    moved = speech[lag:] & speech[:-lag]
    variance = float(levels[speech].std(axis=0).mean())
    flux = float(numpy.median(numpy.sqrt(((levels[lag:] - levels[:-lag]) ** 2).mean(axis=1))[moved]))

    # Onsets - the low band's upward crossings of its onset level after ONSET_QUIET frames below it, whose next ONSET_PEAK
    # frames' peak is a sound (within 15 dB of the loud frames): the frames to within ONSET_REACH dB of that peak
    low = 10 * numpy.log10(power[:, (frequencies >= 60) & (frequencies < 1000)].sum(axis=1) + 1e-12)
    loud = numpy.percentile(low, 95)
    rises = []
    for index in range(ONSET_QUIET, len(low) - ONSET_PEAK):
        if low[index] >= loud - ONSET_LEVEL and (low[index - ONSET_QUIET:index] < loud - ONSET_LEVEL).all():
            peak = low[index:index + ONSET_PEAK].max()
            if peak >= loud - 15:
                rises.append(int(numpy.argmax(low[index:index + ONSET_PEAK] >= peak - ONSET_REACH)) * TEMPORAL_HOP * 1000)
    return {'variance': variance, 'flux': flux, 'onset': float(numpy.median(rises)) if rises else None}


# Analyze a WAV file - (path, pitch floor, pitch ceiling, bandwidth) -> {path, the measures, bands}
def analyze(task):
    import numpy  # pylint: disable=import-outside-toplevel
    import parselmouth  # pylint: disable=import-outside-toplevel, import-error
    from .wav import read_wav  # pylint: disable=import-outside-toplevel
    path, floor, ceiling, bandwidth = task
    samples, rate = read_wav(path)
    samples = samples.mean(axis=1)
    sound = parselmouth.Sound(samples, sampling_frequency=rate)
    pitch = sound.to_pitch_ac(time_step=PITCH_STEP, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array['frequency']
    times = pitch.xs()
    voiced = f0 > 0
    result = {'path': path, **spectral_balance(samples, rate, times, voiced, bandwidth), **temporal(samples, rate)}

    # The harmonics-to-noise ratios - the whole and the 2-5 kHz band (None without a periodic frame)
    for name, signal in (('hnr', samples), ('hnr2k', band_pass(samples, rate, 2000, 5000))):
        band_sound = parselmouth.Sound(signal, sampling_frequency=rate)
        values = band_sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=floor).values[0]
        values = values[values > -100]
        result[name] = float(numpy.mean(values)) if len(values) else None

    # The 2-5 kHz envelope's crest factor in 30 ms windows at the voiced frames (the median)
    band_envelope = envelope(band_pass(samples, rate, 2000, 5000))
    crests = []
    for time in times[voiced]:
        window = band_envelope[max(0, int((time - 0.015) * rate)):int((time + 0.015) * rate)]
        if len(window) and window.mean() > 1e-6:
            crests.append(window.max() / window.mean())
    result['crest2k'] = float(numpy.median(crests)) if crests else None

    # The pitch's flutter (4-20 Hz) and wobble (1.5-6 Hz), in cents of the voiced frames' median
    cents = 1200 * numpy.log2(numpy.where(voiced, f0, 1) / (numpy.median(f0[voiced]) if voiced.any() else 1))
    result['f0Flutter'] = band_rms(cents, voiced, 1 / PITCH_STEP, 0.025, 0.125)
    result['f0Wobble'] = band_rms(cents, voiced, 1 / PITCH_STEP, 0.08, 0.33)
    return result


# The render jobs (tools/renderLines.js's items) and analysis tasks [(kind, who, task)] of a measurement: each voice over
# the panel texts ('voice') and over per_voice of its held-out clips' texts ('synth'), and those clips ('source');
# speakers, the training data's ({key: {pitchFloor, pitchCeiling, bandwidth}}); clips, each voice's held-out clips
# [{id, text, wav}] (test, then val)
def plan(out_dir, per_voice, edits, speakers, clips, panel_texts):
    jobs, tasks = [], []
    for key, info in speakers.items():
        settings = (info['pitchFloor'], info['pitchCeiling'], info['bandwidth'])
        for text in panel_texts:
            wav = os.path.join(out_dir, 'voices', f'{key}-{text["id"]}.wav')
            jobs.append({'text': text['text'], 'voice': key, 'wav': wav, **({'edits': edits} if edits else {})})
            tasks.append(('voice', key, (wav, info['pitchFloor'], info['pitchCeiling'], VOICE_BANDWIDTH)))
        for clip in clips.get(key, [])[:per_voice]:
            wav = os.path.join(out_dir, 'synth', f'{key}-{clip["id"]}.wav')
            jobs.append({'text': clip['text'], 'voice': key, 'wav': wav, **({'edits': edits} if edits else {})})
            tasks.append(('synth', key, (wav, *settings)))
            tasks.append(('source', key, (clip['wav'], *settings)))
    return jobs, tasks


# Render jobs with the package (tools/renderLines.js - its voice model, or a voice model module's), in processes Node
# processes at once
def render(jobs, processes, model=None):
    groups = [jobs[index::processes] for index in range(processes)]
    command = ['node', os.path.join('tools', 'renderLines.js'), *([os.path.abspath(model)] if model else [])]
    children = [subprocess.Popen(command, cwd=ROOT,  # pylint: disable=consider-using-with
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True) for group in groups if group]
    for child, group in zip(children, [group for group in groups if group]):
        child.stdin.write(''.join(json.dumps(job) + '\n' for job in group))
        child.stdin.close()
    failed = [child.wait() for child in children]
    if any(failed):
        raise RuntimeError('tools/renderLines.js failed')


def measure_command(args):
    from . import tts  # pylint: disable=import-outside-toplevel
    from .data import DATA_DIR, CorpusData  # pylint: disable=import-outside-toplevel
    settings = tts.load_settings()
    directory = tts.corpus_dir(settings)
    speakers = CorpusData(os.path.join(DATA_DIR, 'tts')).index['speakers']
    clips = {}
    for split in ('test', 'val'):
        for clip in tts.selection(settings):
            if clip['split'] == split and os.path.exists(tts.wav_path(directory, clip)):
                clips.setdefault(clip['key'], []).append({**clip, 'wav': tts.wav_path(directory, clip)})
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        panel_texts = json.load(file)['texts']
    edits = json.loads(args.edits) if args.edits else None
    jobs, tasks = plan(args.out, args.per_voice, edits, speakers, clips, panel_texts)
    for name in ('voices', 'synth'):
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    render(jobs, args.jobs, args.model)
    with multiprocessing.Pool(args.jobs) as pool:
        results = pool.map(analyze, [task for _, _, task in tasks], chunksize=4)
    rows = [{'kind': kind, 'who': who, **result} for (kind, who, _), result in zip(tasks, results)]
    path = os.path.join(args.out, 'results.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump({'edits': edits, 'model': args.model, 'rows': rows}, file)
    print(f'quality: {len(rows)} measured - {path}')
    return report(path)


# The gap measure's clips (seconds), its layers (the feature encoder's - the DTW alignment's - and the transformer's), and
# the trim's silence (dB below the loudest 20 ms frame) and margin (s)
GAP_SECONDS = (3.0, 9.0)
GAP_LAYERS = {'conv': None, 'L3': 3, 'L6': 6, 'L12': 12}
GAP_MEASURES = ['fdL12', 'fdL6', 'pairedL3', 'pairedL12']
TRIM_DB = 40
TRIM_MARGIN = 0.05


# Samples trimmed of leading and trailing silence (20 ms frames TRIM_DB below the loudest), TRIM_MARGIN kept
def trim(samples, rate):
    import numpy  # pylint: disable=import-outside-toplevel
    size = rate // 50
    energy = numpy.array([numpy.mean(samples[index:index + size] ** 2) + 1e-12 for index in range(0, len(samples), size)])
    loud = numpy.nonzero(10 * numpy.log10(energy / energy.max()) > -TRIM_DB)[0]
    margin = int(TRIM_MARGIN * rate)
    return samples[max(0, loud[0] * size - margin):min(len(samples), (loud[-1] + 1) * size + margin)]


# A WAV file's features - {layer: [frames, dims] float64}: trimmed, at 16 kHz, normalized as the recognizer expects
def gap_features(model, device, path):
    import numpy  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from .wav import read_wav  # pylint: disable=import-outside-toplevel
    samples, rate = read_wav(path, dtype='float32')
    samples = soxr.resample(trim(samples.mean(axis=1), rate), rate, 16000).astype(numpy.float32)
    samples = (samples - samples.mean()) / numpy.sqrt(samples.var() + 1e-7)
    with torch.no_grad():
        outputs = model.wav2vec2(torch.from_numpy(samples)[None].to(device), output_hidden_states=True)
    return {name: (outputs.extract_features if index is None else outputs.hidden_states[index])[0].cpu().double().numpy()
            for name, index in GAP_LAYERS.items()}


# Rows scaled to unit length
def unit_rows(rows):
    import numpy  # pylint: disable=import-outside-toplevel
    return rows / (numpy.linalg.norm(rows, axis=1, keepdims=True) + 1e-9)


# The DTW alignment of two feature sequences by cosine distance - (first's frames, second's frames) along the path
def dtw_path(first, second):
    import numpy  # pylint: disable=import-outside-toplevel
    cost = 1 - unit_rows(first) @ unit_rows(second).T
    rows, columns = cost.shape
    total = numpy.full((rows + 1, columns + 1), numpy.inf)
    total[0, 0] = 0
    # (an anti-diagonal at a time - a cell's predecessors are on the two before it)
    for diagonal in range(2, rows + columns + 1):
        row = numpy.arange(max(1, diagonal - columns), min(rows, diagonal - 1) + 1)
        column = diagonal - row
        total[row, column] = cost[row - 1, column - 1] + numpy.minimum(
            numpy.minimum(total[row - 1, column], total[row, column - 1]), total[row - 1, column - 1])
    path = []
    row, column = rows, columns
    while row > 0 and column > 0:
        path.append((row - 1, column - 1))
        step = int(numpy.argmin((total[row - 1, column - 1], total[row - 1, column], total[row, column - 1])))
        row, column = (row - 1, column - 1) if step == 0 else (row - 1, column) if step == 1 else (row, column - 1)
    first_frames, second_frames = zip(*path[::-1])
    return numpy.array(first_frames), numpy.array(second_frames)


# A clip's paired distance to its source clip in a layer - the mean cosine distance along their DTW alignment
def paired_distance(features, source, layer):
    import numpy  # pylint: disable=import-outside-toplevel
    first, second = dtw_path(features['conv'], source['conv'])
    return float(numpy.mean(1 - numpy.sum(unit_rows(features[layer][first]) * unit_rows(source[layer][second]), axis=1)))


# The Frechet distance of two sets of frames [count, dims]
def frechet(first, second):
    import numpy  # pylint: disable=import-outside-toplevel
    cov_first, cov_second = numpy.cov(first, rowvar=False), numpy.cov(second, rowvar=False)
    values, vectors = numpy.linalg.eigh(cov_first)
    root = (vectors * numpy.sqrt(numpy.clip(values, 0, None))) @ vectors.T
    product = numpy.linalg.eigvalsh(root @ cov_second @ root)
    return float(numpy.sum((first.mean(0) - second.mean(0)) ** 2) + numpy.trace(cov_first) + numpy.trace(cov_second)
                 - 2 * numpy.sum(numpy.sqrt(numpy.clip(product, 0, None))))


# A voice's gap measures - its synthesized clips' features against their source clips' ([(synth, source)])
def gap_measures(pairs):
    import numpy  # pylint: disable=import-outside-toplevel
    result = {}
    for layer in ('L12', 'L6'):
        result[f'fd{layer}'] = frechet(numpy.concatenate([synth[layer] for synth, _ in pairs]),
                                       numpy.concatenate([source[layer] for _, source in pairs]))
    for layer in ('L3', 'L12'):
        result[f'paired{layer}'] = sum(paired_distance(synth, source, layer) for synth, source in pairs) / len(pairs)
    return result


def gap_command(args):
    from . import ctc, tts  # pylint: disable=import-outside-toplevel
    from .wav import read_wav  # pylint: disable=import-outside-toplevel
    settings = tts.load_settings()
    directory = tts.corpus_dir(settings)
    clips = {}
    for clip in sorted(tts.selection(settings), key=lambda clip: clip['id']):
        path = tts.wav_path(directory, clip)
        if clip['split'] == 'test' and len(clips.get(clip['key'], [])) < args.per_voice and os.path.exists(path):
            samples, rate = read_wav(path)
            if GAP_SECONDS[0] <= len(samples) / rate <= GAP_SECONDS[1]:
                clips.setdefault(clip['key'], []).append({**clip, 'wav': path})
    os.makedirs(os.path.join(args.out, 'gap'), exist_ok=True)
    jobs = [{'text': clip['text'], 'voice': key, 'wav': os.path.join(args.out, 'gap', f'{key}-{clip["id"]}.wav')}
            for key, voice_clips in clips.items() for clip in voice_clips]
    render(jobs, args.jobs, args.model)
    model, _, device = ctc.load_ctc()
    voices = {}
    for key, voice_clips in clips.items():
        pairs = [(gap_features(model, device, os.path.join(args.out, 'gap', f'{key}-{clip["id"]}.wav')),
                  gap_features(model, device, clip['wav'])) for clip in voice_clips]
        voices[key] = gap_measures(pairs)
    path = os.path.join(args.out, 'gap.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump({'model': args.model, 'perVoice': args.per_voice, 'voices': voices}, file, indent=1)
    print(f'quality gap: {sum(len(voice_clips) for voice_clips in clips.values())} clips - {path}')
    print(f'{"voice":16}' + ''.join(f'{name:>12}' for name in GAP_MEASURES))
    for key, values in voices.items():
        print(f'{key:16}' + ''.join(f'{values[name]:12.4g}' for name in GAP_MEASURES))
    print(f'{"mean":16}' + ''.join(f'{sum(values[name] for values in voices.values()) / len(voices):12.4g}' for name in GAP_MEASURES))
    return 0


# A measure's mean over rows (None for none)
def mean(rows, name):
    values = [row[name] for row in rows if row.get(name) is not None]
    return sum(values) / len(values) if values else None


# Print each voice's synthesized measures (its held-out texts, then the panel texts) less its source clips'
def report(path):
    with open(path, encoding='utf-8') as file:
        results = json.load(file)
    groups = {}
    for row in results['rows']:
        groups.setdefault((row['kind'], row['who']), []).append(row)
    source = {who: {name: mean(rows, name) for name in MEASURES} for (kind, who), rows in groups.items() if kind == 'source'}

    def cells(values):
        return ''.join(f'{value:+10.2f}' if value is not None else '       n/a' for value in values)
    header = ''.join(f'{name[:9]:>10}' for name in MEASURES)
    for kind, title in (('synth', 'Voices synthesized (their held-out texts), less their source clips'),
                        ('voice', 'Voices (the panel texts), less their source clips')):
        print(f'{title}\n{"voice":16}{header}')
        for (row_kind, who), rows in sorted(groups.items()):
            if row_kind == kind:
                deltas = [None if mean(rows, name) is None or source.get(who, {}).get(name) is None else
                          mean(rows, name) - source[who][name] for name in MEASURES]
                print(f'{who:16}{cells(deltas)}')
        print()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='picovox_train.quality', description='Voice quality measures against their source clips')
    commands = parser.add_subparsers(dest='command', required=True)
    measure_parser = commands.add_parser('measure', help='render and measure the voices and their source clips')
    measure_parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'quality'), help='the output directory')
    measure_parser.add_argument('--per-voice', type=int, default=16, help='held-out clips per voice')
    measure_parser.add_argument('--edits', help='frame edits for every rendering (tools/renderLines.js\'s, JSON)')
    measure_parser.add_argument('--model', help='a voice model module to render with (default: the package\'s)')
    measure_parser.add_argument('-j', '--jobs', type=int, default=8, help='Node renderers and analysis workers')
    gap_parser = commands.add_parser('gap', help='the voices\' distance from their source in wav2vec2 features')
    gap_parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'quality'), help='the output directory')
    gap_parser.add_argument('--per-voice', type=int, default=40, help='held-out clips per voice')
    gap_parser.add_argument('--model', help='a voice model module to render with (default: the package\'s)')
    gap_parser.add_argument('-j', '--jobs', type=int, default=8, help='Node renderers')
    report_parser = commands.add_parser('report', help='print a measurement\'s results')
    report_parser.add_argument('results', help='a measurement\'s results.json')
    args = parser.parse_args(argv)
    if args.command == 'measure':
        return measure_command(args)
    if args.command == 'gap':
        return gap_command(args)
    return report(args.results)


if __name__ == '__main__':
    sys.exit(main())
