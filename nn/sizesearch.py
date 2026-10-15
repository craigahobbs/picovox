# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The acoustic model's minimal-size search (paper section 9.2) - a ladder of model widths and depths, by encoder type
(self-attention, all-convolution) and weight precision (8, 6, 5, 4 bits; post-training quantization, and
quantization-aware training for the candidates), each trained as a proxy run of a fixed length, measured, and placed on
the Pareto front of size against quality:

    build/venv-nn/bin/python -B -m nn.sizesearch plan                 # the ladder, the runs, and the GPU-time estimate
    caffeinate -dims build/venv-nn/bin/python -B -m nn.sizesearch run # everything, one GPU run at a time (resumable)
    build/venv-nn/bin/python -B -m nn.sizesearch report               # pareto.json and pareto.svg from the results so far

The method, fixed before searching:

1. The ladder (LADDER): widths (dim = decoderDim) 64, 96, 128, 160 x depths (encoder, decoder layers) (2, 3), (3, 4),
   (3, 6) x encoder (attention, conv) - 24 proxies, from 0.1M to 1.4M parameters; the current model (128, (3, 6),
   attention) is one of them. --seeds trains each with several seeds (default 1).
2. The proxies - nn/trainmulti.py from scratch, readers only (no emotional data, no critic), PROXY_STEPS steps (--steps),
   the default learning rate schedule; one at a time.
3. The measures (nn/archeval.py) of each proxy's export at 8, 6, 5, and 4 bits (post-training quantization; 4 and 5 bits
   with per-group scales of GROUP): the validation spectral loss of the exported weights (VAL_COUNT reader utterances),
   and on the listener panel's texts for the six sample speakers (three female, three male; the browser path at 16 kHz)
   the mean UTMOS, Whisper's WER, and the ECAPA speaker similarity to the speakers' recordings.
4. Quantization-aware training (stage 2) - for the candidates (the points within the tolerance at 8 bits, and the next
   smaller proxy of each encoder), a QAT fine-tune (QAT_STEPS steps) at 6, 5, and 4 bits, measured the same way.
5. The size - the generated JavaScript module's bytes (nn/package.py; gzip bytes too, the served size).
6. The tolerance (TOLERANCE): the selected point is the smallest module whose mean UTMOS is within 0.05 of the best
   point's, whose WER is no more than 1.5 points above the best WER, and whose speaker similarity is no more than 0.03
   below the best - the means over seeds when there are several.

Outputs, under build/nn/sizesearch/<name>/: plan.json, points/<point>.json (each measured point), pareto.json (every
point, the Pareto front of module gzip bytes against UTMOS, the tolerance, and the selection), pareto.svg, and the log
(search.log). Training runs are build/nn/runs/ss-<name>-<proxy>/ (and -qat<bits>).
"""

import argparse
import concurrent.futures
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time

from .archeval import ARCH_DIR, evaluate


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_DIR = os.path.join(ROOT, 'build', 'nn', 'sizesearch')
RUNS_DIR = os.path.join(ROOT, 'build', 'nn', 'runs')
PYTHON = sys.executable

# The ladder - widths (dim = decoderDim; ffnMultiple 2, two attention heads) and depths (encoder, decoder layers), by
# encoder
WIDTHS = (64, 96, 128, 160)
DEPTHS = ((2, 3), (3, 4), (3, 6))
ENCODERS = ('attention', 'conv')
LADDER = [{'dim': width, 'encoderLayers': depth[0], 'decoderLayers': depth[1], 'encoder': encoder}
          for encoder in ENCODERS for depth in DEPTHS for width in WIDTHS]

# The precisions measured - (bits, group): 8 and 6 bits per row, 5 and 4 bits per group of GROUP weights (64: at 4 bits
# on multi-a step 15000, post-training, as good as groups of 32 and 6% smaller; per row is worse - build/nn/arch/qat.log)
GROUP = 64
PRECISIONS = ((8, 0), (6, 0), (5, GROUP), (4, GROUP))
QAT_PRECISIONS = ((6, 0), (5, GROUP), (4, GROUP))

# The proxy runs' length, the QAT fine-tunes', and the validation utterances of the measures
PROXY_STEPS = 8000
QAT_STEPS = 1000
VAL_COUNT = 96

# The tolerance - UTMOS within mos of the best, WER at most wer points above the best, similarity at most similarity below
TOLERANCE = {'mos': 0.05, 'wer': 1.5, 'similarity': 0.03}

# Seconds per training step (the optimized trainer, nn/trainmulti.py, on an M3 Max's GPU, alone): measured warm
# 0.142 s and on fresh shapes 0.249 s for the current model's configuration (paper section 9.3) - the proxies' synthesizer
# dominates a step, so width matters little; the estimate uses their mean
SECONDS_PER_STEP = 0.2
# Validation (480 utterances at the end of a run) and the evaluation's GPU validation per precision (seconds)
SECONDS_PER_VALIDATION = 25
SECONDS_PER_EVAL_VALIDATION = 8


# A proxy's name
def proxy_name(proxy, seed=1):
    encoder = 'attn' if proxy['encoder'] == 'attention' else 'conv'
    return f'{encoder}-d{proxy["dim"]}-e{proxy["encoderLayers"]}-l{proxy["decoderLayers"]}' + (f'-s{seed}' if seed != 1 else '')


# A proxy's model config (nn/model.py MultiModel overrides)
def proxy_config(proxy):
    config = {'dim': proxy['dim'], 'decoderDim': proxy['dim'], 'encoderLayers': proxy['encoderLayers'],
              'decoderLayers': proxy['decoderLayers']}
    if proxy['encoder'] != 'attention':
        config['encoder'] = proxy['encoder']
    return config


# A proxy's parameter count (without building a model's training state)
def proxy_parameters(proxy):
    from .model import MultiModel  # pylint: disable=import-outside-toplevel
    model = MultiModel({'speakers': 328, 'vocab': 78, **proxy_config(proxy)})
    total = sum(parameter.numel() for parameter in model.parameters())
    return total, total - model.speaker.weight.numel() - model.eq_free.numel()


# The search's plan - the runs and the estimated GPU time
def make_plan(args):
    proxies = []
    for proxy in LADDER:
        for seed in range(1, args.seeds + 1):
            total, core = proxy_parameters(proxy)
            proxies.append({**proxy, 'seed': seed, 'name': proxy_name(proxy, seed), 'parameters': total, 'coreParameters': core})
    proxy_steps = len(proxies) * args.steps
    # Stage 2 - the QAT candidates; about a third of the proxies, by experience, x the QAT precisions
    qat_runs = math.ceil(len(proxies) / 3) * len(QAT_PRECISIONS)
    qat_steps = qat_runs * args.qat_steps
    evaluations = len(proxies) * len(PRECISIONS) + qat_runs
    seconds_per_step = args.seconds_per_step
    gpu_seconds = (proxy_steps + qat_steps) * seconds_per_step + (len(proxies) + qat_runs) * SECONDS_PER_VALIDATION + \
        evaluations * SECONDS_PER_EVAL_VALIDATION
    return {
        'name': args.name, 'steps': args.steps, 'qatSteps': args.qat_steps, 'seeds': args.seeds, 'precisions': PRECISIONS,
        'qatPrecisions': QAT_PRECISIONS, 'tolerance': TOLERANCE, 'valCount': args.val_count, 'texts': args.texts,
        'proxies': proxies, 'proxySteps': proxy_steps, 'qatRunsEstimated': qat_runs, 'qatStepsEstimated': qat_steps,
        'evaluationsEstimated': evaluations,
        'estimate': {'secondsPerStep': seconds_per_step, 'gpuHours': round(gpu_seconds / 3600, 1),
                     'gpuHoursRange': [round((gpu_seconds + (proxy_steps + qat_steps) * (rate - seconds_per_step)) / 3600, 1)
                                       for rate in (0.142, 0.249)],
                     'cpuEvaluationHours': round(evaluations * 150 / 3600, 1),
                     'note': 'GPU time: training steps x s/step (0.142 warm - 0.249 fresh shapes, alone on the GPU) plus '
                             'validation; the panel scoring (CPU, ~2.5 min per evaluation) overlaps the next proxy\'s training'}
    }


_LOG_LOCK = threading.Lock()


def _log(directory, message):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}'
    with _LOG_LOCK:
        print(line, file=sys.stderr, flush=True)
        with open(os.path.join(directory, 'search.log'), 'a', encoding='utf-8') as file:
            file.write(line + '\n')


# Train a run (nn/trainmulti.py) unless its final checkpoint exists - the checkpoint path
def train(run, steps, extra, directory, train_args=''):
    checkpoint = os.path.join(RUNS_DIR, run, f'step-{steps}.pt')
    if os.path.exists(checkpoint):
        return checkpoint
    os.makedirs(os.path.join(RUNS_DIR, run), exist_ok=True)
    command = [PYTHON, '-B', '-m', 'nn.trainmulti', '--name', run, '--steps', str(steps), '--val-every', str(steps),
               '--keep-every', str(steps), *extra, *shlex.split(train_args)]
    _log(directory, f'train {run}: {" ".join(command[3:])}')
    start = time.time()
    with open(os.path.join(RUNS_DIR, run, 'train.log'), 'a', encoding='utf-8') as log:
        subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    _log(directory, f'train {run}: done in {time.time() - start:.0f} s')
    return checkpoint


# Measure a checkpoint at a precision unless measured - the point's result (qat marks a QAT fine-tune's point)
def measure(directory, point, checkpoint, bits, group, args, qat=False):
    path = os.path.join(directory, 'points', f'{point}.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as file:
            return json.load(file)
    result = evaluate(checkpoint, point, os.path.join(directory, 'eval'), bits, group, args.device, args.val_count, args.texts,
                      keep_wavs=False)
    result = {key: value for key, value in result.items() if key != 'transcripts'}
    result['qat'] = qat
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=1)
    _log(directory, f'measured {point}: UTMOS {result["panel"]["mos"]:.3f}, WER {result["panel"]["wer"]:.2f}, '
                    f'similarity {result["panel"]["similarity"]:.3f}, val spectral {result["val"]["spectral"]:.4f}, '
                    f'module {result["size"]["moduleBytes"]} bytes ({result["size"]["moduleGzipBytes"]} gzip)')
    return result


# The points' tolerance and selection - (the best values, the points within tolerance, the selected point)
def select(points):
    best = {'mos': max(point['panel']['mos'] for point in points), 'wer': min(point['panel']['wer'] for point in points),
            'similarity': max(point['panel']['similarity'] for point in points)}
    within = [point for point in points if point['panel']['mos'] >= best['mos'] - TOLERANCE['mos'] and
              point['panel']['wer'] <= best['wer'] + TOLERANCE['wer'] and
              point['panel']['similarity'] >= best['similarity'] - TOLERANCE['similarity']]
    selected = min(within, key=lambda point: point['size']['moduleGzipBytes']) if within else None
    return best, within, selected


# The Pareto front - the points no other point beats on both size (module gzip bytes, smaller) and UTMOS (higher)
def pareto_front(points):
    ordered = sorted(points, key=lambda point: (point['size']['moduleGzipBytes'], -point['panel']['mos']))
    front = []
    for point in ordered:
        if not front or point['panel']['mos'] > front[-1]['panel']['mos']:
            front.append(point)
    return front


# Merge the seeds of each (proxy, precision) - means of the measures
def merge_seeds(points):
    groups = {}
    for point in points:
        groups.setdefault(point['point'], []).append(point)
    merged = []
    for key, group in groups.items():
        first = dict(group[0])
        for section in ('panel',):
            first[section] = {name: sum(point[section][name] for point in group) / len(group) for name in group[0][section]}
        first['val'] = {name: sum(point['val'][name] for point in group) / len(group) for name in ('spectral', 'score')}
        first['seeds'] = len(group)
        first['point'] = key
        merged.append(first)
    return merged


# The report - pareto.json and pareto.svg from the measured points
def report(directory):
    files = sorted(os.listdir(os.path.join(directory, 'points'))) if os.path.isdir(os.path.join(directory, 'points')) else []
    points = []
    for name in files:
        with open(os.path.join(directory, 'points', name), encoding='utf-8') as file:
            point = json.load(file)
        point['point'] = point['name'].replace('-s2', '').replace('-s3', '')
        points.append(point)
    if not points:
        return None
    merged = merge_seeds(points)
    best, within, selected = select(merged)
    front = pareto_front(merged)
    out = {
        'tolerance': TOLERANCE, 'best': best, 'selected': selected['point'] if selected else None,
        'within': [point['point'] for point in within], 'front': [point['point'] for point in front],
        'points': [{'point': point['point'], 'encoder': point['config'].get('encoder', 'attention'), 'dim': point['config']['dim'],
                    'encoderLayers': point['config']['encoderLayers'], 'decoderLayers': point['config']['decoderLayers'],
                    'bits': point['bits'], 'group': point['group'], 'qat': point.get('qat', False), 'seeds': point['seeds'],
                    'parameters': point['parameters'], 'moduleBytes': point['size']['moduleBytes'],
                    'moduleGzipBytes': point['size']['moduleGzipBytes'], 'mos': round(point['panel']['mos'], 4),
                    'wer': round(point['panel']['wer'], 2), 'similarity': round(point['panel']['similarity'], 4),
                    'valSpectral': round(point['val']['spectral'], 5)} for point in merged]
    }
    with open(os.path.join(directory, 'pareto.json'), 'w', encoding='utf-8') as file:
        json.dump(out, file, indent=1)
    with open(os.path.join(directory, 'pareto.svg'), 'w', encoding='utf-8') as file:
        file.write(pareto_svg(out))
    return out


# The Pareto chart - UTMOS against the module's gzip size (log scale), a mark per point (shape by encoder, color by bits),
# the front, the tolerance line, and the selected point
def pareto_svg(report_data):
    points = report_data['points']
    width, height, margin = 760, 480, 64
    sizes = [point['moduleGzipBytes'] / 1024 for point in points]
    scores = [point['mos'] for point in points]
    low_x, high_x = math.log10(min(sizes) * 0.85), math.log10(max(sizes) * 1.15)
    low_y = min(min(scores), report_data['best']['mos'] - 0.1) - 0.05
    high_y = max(scores) + 0.05

    def x_at(kilobytes):
        return margin + (math.log10(kilobytes) - low_x) / (high_x - low_x) * (width - 2 * margin)

    def y_at(mos):
        return height - margin - (mos - low_y) / (high_y - low_y) * (height - 2 * margin)
    colors = {8: '#2a6fdb', 6: '#14a37f', 5: '#e0a019', 4: '#d64545'}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="15">'
             'Acoustic model size search - UTMOS against module size</text>']
    # Axes and ticks
    parts.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#444"/>')
    parts.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#444"/>')
    for kilobytes in (50, 100, 200, 300, 500, 700, 1000, 1500, 2000):
        if low_x <= math.log10(kilobytes) <= high_x:
            parts.append(f'<line x1="{x_at(kilobytes):.1f}" y1="{height - margin}" x2="{x_at(kilobytes):.1f}" '
                         f'y2="{height - margin + 5}" stroke="#444"/>')
            parts.append(f'<text x="{x_at(kilobytes):.1f}" y="{height - margin + 18}" text-anchor="middle">{kilobytes}</text>')
    tick = math.floor(low_y * 10) / 10
    while tick <= high_y:
        if tick >= low_y:
            parts.append(f'<text x="{margin - 8}" y="{y_at(tick) + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
            parts.append(f'<line x1="{margin}" y1="{y_at(tick):.1f}" x2="{width - margin}" y2="{y_at(tick):.1f}" stroke="#eee"/>')
        tick = round(tick + 0.1, 2)
    parts.append(f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle">generated module, gzip KB (log scale)</text>')
    parts.append(f'<text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})">'
                 'UTMOS (panel, six speakers)</text>')
    # The tolerance line
    line_y = y_at(report_data['best']['mos'] - report_data['tolerance']['mos'])
    parts.append(f'<line x1="{margin}" y1="{line_y:.1f}" x2="{width - margin}" y2="{line_y:.1f}" stroke="#999" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{width - margin}" y="{line_y - 5:.1f}" text-anchor="end" fill="#666">'
                 f'best - {report_data["tolerance"]["mos"]}</text>')
    # The front
    by_name = {point['point']: point for point in points}
    front = [by_name[name] for name in report_data['front']]
    path = ' '.join(f'{x_at(point["moduleGzipBytes"] / 1024):.1f},{y_at(point["mos"]):.1f}' for point in front)
    parts.append(f'<polyline points="{path}" fill="none" stroke="#333" stroke-width="1.5"/>')
    # The points
    for point in points:
        cx, cy = x_at(point['moduleGzipBytes'] / 1024), y_at(point['mos'])
        color = colors.get(point['bits'], '#777')
        fill = color if point['qat'] else '#ffffff'
        title = f'<title>{point["point"]}: {point["parameters"]} parameters, {point["moduleGzipBytes"]} gzip bytes, ' \
            f'UTMOS {point["mos"]}, ' \
            f'WER {point["wer"]}, similarity {point["similarity"]}</title>'
        if point['encoder'] == 'conv':
            parts.append(f'<rect x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" fill="{fill}" stroke="{color}" '
                         'stroke-width="2">'
                         f'{title}</rect>')
        else:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}" stroke="{color}" stroke-width="2">{title}</circle>')
    if report_data['selected']:
        point = by_name[report_data['selected']]
        cx, cy = x_at(point['moduleGzipBytes'] / 1024), y_at(point['mos'])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="#000" stroke-width="1.5"/>')
        parts.append(f'<text x="{cx + 14:.1f}" y="{cy + 4:.1f}">selected: {point["point"]}</text>')
    # The legend
    legend_y = margin + 6
    for index, (bits, color) in enumerate(colors.items()):
        parts.append(f'<circle cx="{width - margin - 150}" cy="{legend_y + 16 * index}" r="5" fill="{color}" stroke="{color}"/>')
        parts.append(f'<text x="{width - margin - 140}" y="{legend_y + 16 * index + 4}">{bits}-bit (hollow: post-training)</text>')
    parts.append(f'<circle cx="{width - margin - 150}" cy="{legend_y + 72}" r="5" fill="none" stroke="#333" stroke-width="2"/>')
    parts.append(f'<text x="{width - margin - 140}" y="{legend_y + 76}">attention encoder</text>')
    parts.append(f'<rect x="{width - margin - 154.5}" y="{legend_y + 83.5}" width="9" height="9" fill="none" stroke="#333" '
                 'stroke-width="2"/>')
    parts.append(f'<text x="{width - margin - 140}" y="{legend_y + 92}">all-convolution encoder</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


# Run the search - every proxy (stage 1), the QAT candidates (stage 2), and the report; resumable
def run(args, directory):
    plan = make_plan(args)
    with open(os.path.join(directory, 'plan.json'), 'w', encoding='utf-8') as file:
        json.dump(plan, file, indent=1)
    _log(directory, f'start: {len(plan["proxies"])} proxies x {args.steps} steps, estimated {plan["estimate"]["gpuHours"]} GPU hours')
    proxies = [proxy for proxy in plan['proxies'] if not args.only or proxy['name'] in args.only.split(',')]
    # (each proxy's measures run on a thread while the next proxy trains - the listeners on the CPU)
    measuring = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending = []

    def measure_all(proxy, checkpoint, index):
        for bits, group in PRECISIONS:
            measure(directory, f'{proxy["name"]}-b{bits}', checkpoint, bits, group, args)
        _log(directory, f'stage 1: {index}/{len(proxies)} proxies measured')
        report(directory)
    for index, proxy in enumerate(proxies, 1):
        run_name = f'ss-{args.name}-{proxy["name"]}'
        extra = ['--config', json.dumps({key: value for key, value in proxy_config(proxy).items() if key != 'encoder'}),
                 '--encoder', proxy['encoder'], '--seed', str(proxy['seed']), '--device', args.device]
        checkpoint = train(run_name, args.steps, extra, directory, args.train_args)
        _log(directory, f'stage 1: {index}/{len(proxies)} proxies trained')
        pending.append(measuring.submit(measure_all, proxy, checkpoint, index))
    for future in pending:
        future.result()
    measuring.shutdown()

    # Stage 2 - QAT fine-tunes of the candidates: the 8-bit points within tolerance, and the next smaller proxy of each
    # encoder
    data = report(directory)
    eight = [point for point in data['points'] if point['bits'] == 8 and not point['qat']]
    _, within, _ = select([{'panel': {'mos': point['mos'], 'wer': point['wer'], 'similarity': point['similarity']},
                            'size': {'moduleGzipBytes': point['moduleGzipBytes']}, 'point': point['point']} for point in eight])
    candidates = {point['point'][:-3] for point in within}
    for encoder in ENCODERS:
        sized = sorted((point for point in eight if point['encoder'] == encoder), key=lambda point: point['parameters'])
        names = [point['point'][:-3] for point in sized]
        inside = [ix for ix, name in enumerate(names) if name in candidates]
        if inside and inside[0] > 0:
            candidates.add(names[inside[0] - 1])
    _log(directory, f'stage 2: QAT candidates {sorted(candidates)}')
    for proxy in proxies:
        if proxy['name'] not in candidates:
            continue
        base = os.path.join(RUNS_DIR, f'ss-{args.name}-{proxy["name"]}', f'step-{args.steps}.pt')
        for bits, group in QAT_PRECISIONS:
            run_name = f'ss-{args.name}-{proxy["name"]}-qat{bits}'
            extra = ['--init', base, '--qat-bits', str(bits), '--qat-group', str(group), '--lr', str(args.qat_lr), '--warmup', '50',
                     '--guide-steps', '0.001', '--config', json.dumps({key: value for key, value in proxy_config(proxy).items()
                                                                        if key != 'encoder'}),
                     '--encoder', proxy['encoder'], '--seed', str(proxy['seed']), '--device', args.device]
            checkpoint = train(run_name, args.qat_steps, extra, directory, args.train_args)
            measure(directory, f'{proxy["name"]}-b{bits}-qat', checkpoint, bits, group, args, qat=True)
    data = report(directory)
    _log(directory, f'done: selected {data["selected"]} (best UTMOS {data["best"]["mos"]:.3f}); front {data["front"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m nn.sizesearch', description='The acoustic model\'s minimal-size search')
    parser.add_argument('command', choices=('plan', 'run', 'report'))
    parser.add_argument('--name', default='s1', help='the search\'s name (build/nn/sizesearch/<name>/)')
    parser.add_argument('--steps', type=int, default=PROXY_STEPS, help='each proxy run\'s steps')
    parser.add_argument('--qat-steps', type=int, default=QAT_STEPS)
    parser.add_argument('--qat-lr', type=float, default=1e-4)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--val-count', type=int, default=VAL_COUNT)
    parser.add_argument('--texts', type=int, default=16, help='panel texts per speaker')
    parser.add_argument('--device', default='mps', help='training\'s and validation\'s device (the listeners run on the CPU)')
    parser.add_argument('--seconds-per-step', type=float, default=SECONDS_PER_STEP, help='for the estimate')
    parser.add_argument('--only', default=None, help='only these proxies (names, comma-separated)')
    parser.add_argument('--train-args', default='', help='more nn/trainmulti.py arguments for every run (a string)')
    args = parser.parse_args(argv)
    directory = os.path.join(SEARCH_DIR, args.name)
    os.makedirs(directory, exist_ok=True)
    if args.command == 'plan':
        plan = make_plan(args)
        summary = {key: value for key, value in plan.items() if key != 'proxies'}
        summary['ladder'] = [f'{proxy["name"]}: {proxy["parameters"]} parameters ({proxy["coreParameters"]} core)'
                             for proxy in plan['proxies']]
        json.dump(summary, sys.stdout, indent=1)
        print()
    elif args.command == 'run':
        run(args, directory)
    else:
        json.dump(report(directory), sys.stdout, indent=1)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
