# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Voice quality measures - each of the package's voices, and each of their readers synthesized, measured against the
readers' own recordings (the voices' tinniness and warble, 2026-10-23):

    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality measure --out build/quality \\
        [--per-reader 16] [--edits '{"add": {"shelf": -5}}'] [-j 8]
    PYTHONPATH=train/src build/venv-train/bin/python -B -m picovox_train.quality report build/quality/results.json

measure renders each voice over the panel texts (tools/panel-texts.json) and each of the voices' readers over their
held-out utterances (train/data/multi-split.json's test, then val, then training utterances to --per-reader) with the
package (tools/renderLines.js - --edits are frame edits, for an experiment), and measures them and the readers'
recordings (downloads/corpora) in voiced frames (Praat's pitch):

- bands - third-octave levels (dB re the voiced power from 90 Hz to 95% of the source's bandwidth), and their summaries:
  tilt (dB/octave, 300 Hz to 5 kHz), alpha (50 Hz-1 kHz less 1-5 kHz), hammarberg (the 0-2 kHz peak less the 2-5 kHz),
  presence (2-5 kHz), and air (5-8 kHz)
- hnr - Praat's harmonics-to-noise ratio (cross-correlation), and hnr2k - of the 2-5 kHz band
- crest2k - the 2-5 kHz band's envelope crest factor (30 ms windows) - how pulse-like the upper harmonics are
- f0Flutter - the pitch's 4-20 Hz movement (cents RMS) and f0Wobble - its 1.5-6 Hz movement, in voiced runs of 250 ms+

report prints each reader's synthesized measures less its recordings', and each voice's less its readers' (weighted by
the voice's blend). The measures are guides, not verdicts: listeners heard the voices as tinny, and lowering the source
shelf fixed it while moving presence further below the recordings' (the energy isn't the cue), and phase dispersion cut
crest2k below the recordings' without helping - blind listening decides (tools/voicelab/). Formant and
band modulation and spectral peakiness were measured too and dropped - they didn't track what listeners heard.
"""

import argparse
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys

from . import ROOT


# The measures' names (report's columns), and the third-octave band centers (Hz)
MEASURES = ['tilt', 'alpha', 'hammarberg', 'presence', 'air', 'hnr', 'hnr2k', 'crest2k', 'f0Flutter', 'f0Wobble']
BAND_CENTERS = [100 * 2 ** (index / 3) for index in range(22)]

# The package's audio bandwidth (Hz - its 24 kHz output)
VOICE_BANDWIDTH = 12000

# The analysis frames (samples) and Praat's pitch step (s)
FFT_SIZE = 1024
FFT_HOP = 256
PITCH_STEP = 0.005


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
    result = {'path': path, **spectral_balance(samples, rate, times, voiced, bandwidth)}

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


# The voices - [(key, sex, blend {reader key: share})] - from lib/synthesis.js's voice table
def voice_table():
    with open(os.path.join(ROOT, 'lib', 'synthesis.js'), encoding='utf-8') as file:
        source = file.read()
    voices = []
    entry = r"\{'key': '([a-z]+)', 'label': '[^']*', 'sex': '(male|female)', "
    for key, sex, speaker, blend in re.findall(entry + r"(?:'speaker': '([^']+)'|'rate': [0-9.]+,\s*'blend': (\{[^}]*\}))", source):
        voices.append((key, sex, {speaker: 1.0} if speaker else {reader: float(share) for reader, share in
                                                                re.findall(r"'([^']+)': ([0-9.]+)", blend)}))
    return voices


# The render jobs (tools/renderLines.js's items) and analysis tasks [(kind, who, task)] of a measurement: each voice over
# the panel texts, each reader synthesized over per_reader of its utterances, and those utterances' recordings
def plan(out_dir, per_reader, edits, speakers, split, utterances, panel_texts, voices):
    jobs, tasks = [], []
    for key, _, blend in voices:
        readers = [speakers[reader] for reader in blend]
        floor, ceiling = min(reader['pitchFloor'] for reader in readers), max(reader['pitchCeiling'] for reader in readers)
        for text in panel_texts:
            wav = os.path.join(out_dir, 'voices', f'{key}-{text["id"]}.wav')
            jobs.append({'text': text['text'], 'voice': key, 'wav': wav, **({'edits': edits} if edits else {})})
            tasks.append(('voice', key, (wav, floor, ceiling, VOICE_BANDWIDTH)))
    for reader in sorted({reader for _, _, blend in voices for reader in blend}):
        info = speakers[reader]
        held_out = [sentence for sentence in split.get(reader, []) if sentence in utterances[reader]]
        others = [sentence for sentence in utterances[reader] if sentence not in held_out]
        for sentence in (held_out + others[::max(1, len(others) // per_reader)])[:per_reader]:
            utterance = utterances[reader][sentence]
            wav = os.path.join(out_dir, 'readers', f'{reader}-{sentence}.wav')
            jobs.append({'text': utterance['text'], 'speaker': reader, 'wav': wav, **({'edits': edits} if edits else {})})
            settings = (info['pitchFloor'], info['pitchCeiling'], info['bandwidth'])
            tasks.append(('synth', reader, (wav, *settings)))
            tasks.append(('human', reader, (utterance['wav'], *settings)))
    return jobs, tasks


# Render jobs with the package (tools/renderLines.js), in processes Node processes at once
def render(jobs, processes):
    groups = [jobs[index::processes] for index in range(processes)]
    children = [subprocess.Popen(['node', os.path.join('tools', 'renderLines.js')], cwd=ROOT,  # pylint: disable=consider-using-with
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True) for group in groups if group]
    for child, group in zip(children, [group for group in groups if group]):
        child.stdin.write(''.join(json.dumps(job) + '\n' for job in group))
        child.stdin.close()
    failed = [child.wait() for child in children]
    if any(failed):
        raise RuntimeError('tools/renderLines.js failed')


def measure_command(args):
    from . import multi  # pylint: disable=import-outside-toplevel
    with open(os.path.join(ROOT, 'train', 'data', 'speakers.json'), encoding='utf-8') as file:
        speakers = {speaker['key']: speaker for speaker in json.load(file)['speakers']}
    with open(os.path.join(ROOT, 'train', 'data', 'multi-split.json'), encoding='utf-8') as file:
        split_file = json.load(file)
    split = {key: list(split_file['test'].get(key, [])) + list(split_file['val'].get(key, []))
             for key in set(split_file['test']) | set(split_file['val'])}
    with open(os.path.join(ROOT, 'tools', 'panel-texts.json'), encoding='utf-8') as file:
        panel_texts = json.load(file)['texts']
    voices = voice_table()
    readers = {reader for _, _, blend in voices for reader in blend}
    utterances = {key: {utterance['sentence']: utterance for utterance in info['utterances']}
                  for key, info in multi.candidate_speakers().items() if key in readers}
    edits = json.loads(args.edits) if args.edits else None
    jobs, tasks = plan(args.out, args.per_reader, edits, speakers, split, utterances, panel_texts, voices)
    for directory in ('voices', 'readers'):
        os.makedirs(os.path.join(args.out, directory), exist_ok=True)
    render(jobs, args.jobs)
    with multiprocessing.Pool(args.jobs) as pool:
        results = pool.map(analyze, [task for _, _, task in tasks], chunksize=4)
    rows = [{'kind': kind, 'who': who, **result} for (kind, who, _), result in zip(tasks, results)]
    path = os.path.join(args.out, 'results.json')
    with open(path, 'w', encoding='utf-8') as file:
        json.dump({'edits': edits, 'blends': {key: blend for key, _, blend in voices}, 'rows': rows}, file)
    print(f'quality: {len(rows)} measured - {path}')
    return report(path)


# A measure's mean over rows (None for none)
def mean(rows, name):
    values = [row[name] for row in rows if row.get(name) is not None]
    return sum(values) / len(values) if values else None


# Print the readers' synthesized measures less their recordings', and the voices' less their readers' (blend-weighted)
def report(path):
    with open(path, encoding='utf-8') as file:
        results = json.load(file)
    groups = {}
    for row in results['rows']:
        groups.setdefault((row['kind'], row['who']), []).append(row)
    human = {who: {name: mean(rows, name) for name in MEASURES} for (kind, who), rows in groups.items() if kind == 'human'}

    def cells(values):
        return ''.join(f'{value:+10.2f}' if value is not None else '       n/a' for value in values)
    header = ''.join(f'{name[:9]:>10}' for name in MEASURES)
    print(f'Readers synthesized, less their recordings\n{"reader":16}{header}')
    for (kind, who), rows in sorted(groups.items()):
        if kind == 'synth':
            deltas = [None if mean(rows, name) is None or human[who][name] is None else mean(rows, name) - human[who][name]
                      for name in MEASURES]
            print(f'{who:16}{cells(deltas)}')
    print(f'\nVoices (the panel texts), less their readers\' recordings (blend-weighted)\n{"voice":16}{header}')
    for (kind, who), rows in groups.items():
        if kind == 'voice':
            blend = results['blends'][who]
            deltas = []
            for name in MEASURES:
                parts = [(share, human.get(reader, {}).get(name)) for reader, share in blend.items()]
                value = mean(rows, name)
                if value is None or any(part is None for _, part in parts):
                    deltas.append(None)
                else:
                    deltas.append(value - sum(share * part for share, part in parts) / sum(share for share, _ in parts))
            print(f'{who:16}{cells(deltas)}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='picovox_train.quality', description='Voice quality measures against the readers\' recordings')
    commands = parser.add_subparsers(dest='command', required=True)
    measure_parser = commands.add_parser('measure', help='render and measure the voices and their readers')
    measure_parser.add_argument('--out', default=os.path.join(ROOT, 'build', 'quality'), help='the output directory')
    measure_parser.add_argument('--per-reader', type=int, default=16, help='utterances per reader')
    measure_parser.add_argument('--edits', help='frame edits for every rendering (tools/renderLines.js\'s, JSON)')
    measure_parser.add_argument('-j', '--jobs', type=int, default=8, help='Node renderers and analysis workers')
    report_parser = commands.add_parser('report', help='print a measurement\'s results')
    report_parser.add_argument('results', help='a measurement\'s results.json')
    args = parser.parse_args(argv)
    if args.command == 'measure':
        return measure_command(args)
    return report(args.results)


if __name__ == '__main__':
    sys.exit(main())
