# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Voice quality measures - each of the package's voices, synthesized, measured against its source: the training
speech's clips of its engine voice (tts.py - the voices' tinniness and warble, 2026-10-23):

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality measure --out build/quality \\
        [--per-voice 16] [--edits '{"add": {"shelf": -5}}'] [--model build/train/voiceModel.js] [-j 8]
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality report build/quality/results.json

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
    report_parser = commands.add_parser('report', help='print a measurement\'s results')
    report_parser.add_argument('results', help='a measurement\'s results.json')
    args = parser.parse_args(argv)
    if args.command == 'measure':
        return measure_command(args)
    return report(args.results)


if __name__ == '__main__':
    sys.exit(main())
