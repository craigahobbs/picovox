# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The listener-model panel - neural listeners that score WAV files, one JSON object per file (JSON lines) on stdout.

    build/venv-listeners/bin/python -I train/src/picovox_train/listeners.py [--no-mos] [--no-embedding] [--no-emotion]
        [--device auto|cpu|mps] [--files-from list.txt|-] file.wav ...

Each line is {"file", "seconds", "mos", "embedding", "emotion"}:

- mos - UTMOS22 strong (the UTokyo-SaruLab VoiceMOS Challenge 2022 system, through torch.hub tarepan/SpeechMOS), a
  predicted mean opinion score of naturalness from 1 (bad) to 5 (excellent).
- embedding - the 192-dimension ECAPA-TDNN speaker embedding (speechbrain/spkrec-ecapa-voxceleb); compare two with
  cosine similarity (same speaker is typically above 0.5-0.6).
- emotion - emotion2vec+ base (emotion2vec/emotion2vec_plus_base) class probabilities: angry, disgusted, fearful,
  happy, neutral, other, sad, surprised, unknown.

Audio of any sample rate is mixed to mono and resampled to 16 kHz. Files longer than --chunk-seconds are scored in
chunks (MOS and emotion averaged by duration, embeddings averaged). All models are loaded once per run. Model files
are cached under data/listeners/ (like the corpora, kept by make clean); after the first run, no network access is
needed. On Apple silicon the models run on the GPU (MPS) by default - about 3x faster than the CPU, with the same scores
to four decimals.
"""

import argparse
import contextlib
import io
import json
import logging
import math
import os
import sys
import warnings


# (the repository root - train/src/picovox_train/ is three levels below it)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE_DIR = os.path.join(ROOT, 'data', 'listeners')
SAMPLE_RATE = 16000

UTMOS_REPO = 'tarepan/SpeechMOS:v1.2.0'
UTMOS_REPO_DIR = 'tarepan_SpeechMOS_v1.2.0'
ECAPA_REPO = 'speechbrain/spkrec-ecapa-voxceleb'
EMOTION_REPO = 'emotion2vec/emotion2vec_plus_base'

# The model caches, set before the model libraries are imported
os.environ.setdefault('TORCH_HOME', os.path.join(CACHE_DIR, 'torch'))
os.environ.setdefault('HF_HOME', os.path.join(CACHE_DIR, 'hf'))
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(CACHE_DIR, 'modelscope'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# Parse the command line
def parse_args(argv):
    parser = argparse.ArgumentParser(prog='listeners.py', description='Score WAV files with the listener-model panel')
    parser.add_argument('files', nargs='*', help='WAV files')
    parser.add_argument('--files-from', metavar='PATH',
                        help='read more WAV paths, one per line, from a file (- for stdin)')
    parser.add_argument('--no-mos', action='store_true', help='skip the UTMOS naturalness score')
    parser.add_argument('--no-embedding', action='store_true', help='skip the ECAPA speaker embedding')
    parser.add_argument('--no-emotion', action='store_true', help='skip the emotion2vec+ emotion probabilities')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps'],
                        help='the torch device (default auto - mps on Apple silicon, else cpu)')
    parser.add_argument('--chunk-seconds', type=float, default=20, help='score longer files in chunks of this length')
    parser.add_argument('--emotion-model', default=EMOTION_REPO,
                        help=f'the emotion2vec+ model (default {EMOTION_REPO})')
    args = parser.parse_args(argv)
    if args.files_from:
        with (sys.stdin if args.files_from == '-' else open(args.files_from, encoding='utf-8')) as list_file:
            args.files.extend(line.strip() for line in list_file if line.strip())
    if not args.files:
        parser.error('no WAV files')
    return args


# Quiet the model libraries' logging, warnings, and progress output (stdout is reserved for the JSON lines)
def quiet_libraries():
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    os.environ.setdefault('TQDM_DISABLE', '1')


# Run a function with its stdout output sent to stderr
def to_stderr(function, *args, **kwargs):
    with contextlib.redirect_stdout(sys.stderr):
        return function(*args, **kwargs)


# Download a Hugging Face model snapshot, or find it in the cache without network access - returns its local directory
# (the listener models' libraries - huggingface_hub, speechbrain, funasr - are build/venv-listeners's)
def hf_snapshot(repo):
    import huggingface_hub  # pylint: disable=import-outside-toplevel,import-error
    try:
        return huggingface_hub.snapshot_download(repo, local_files_only=True)
    except Exception:  # pylint: disable=broad-exception-caught
        return to_stderr(huggingface_hub.snapshot_download, repo)


# Load UTMOS22 strong (torch.hub) - from the hub cache when present, so no network access is needed
def load_utmos(device):
    import torch  # pylint: disable=import-outside-toplevel
    repo_dir = os.path.join(torch.hub.get_dir(), UTMOS_REPO_DIR)
    if os.path.isfile(os.path.join(repo_dir, 'hubconf.py')):
        model = to_stderr(torch.hub.load, repo_dir, 'utmos22_strong', source='local', progress=False)
    else:
        model = to_stderr(torch.hub.load, UTMOS_REPO, 'utmos22_strong', trust_repo=True, progress=False)
    return model.to(device).eval()


# Load the ECAPA-TDNN speaker encoder (speechbrain)
def load_ecapa(device):
    from speechbrain.inference.speaker import EncoderClassifier  # pylint: disable=import-outside-toplevel,import-error
    source = hf_snapshot(ECAPA_REPO)
    savedir = os.path.join(CACHE_DIR, 'spkrec-ecapa-voxceleb')
    return to_stderr(EncoderClassifier.from_hparams, source=source, savedir=savedir, run_opts={'device': device})


# Load the emotion2vec+ emotion recognizer (funasr) from its local snapshot
def load_emotion(model_repo, device):
    from funasr import AutoModel  # pylint: disable=import-outside-toplevel,import-error
    model_dir = hf_snapshot(model_repo)
    return to_stderr(AutoModel, model=model_dir, disable_update=True, device=device, log_level='ERROR')


# Read a WAV file as 16 kHz mono float32 - returns (samples, original duration in seconds)
def read_audio(path):
    import numpy  # pylint: disable=import-outside-toplevel
    import soundfile  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    samples, sample_rate = soundfile.read(path, dtype='float32', always_2d=True)
    samples = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        samples = soxr.resample(samples, sample_rate, SAMPLE_RATE, quality='HQ').astype(numpy.float32)
    return samples, len(samples) / SAMPLE_RATE


# Split samples into chunks of at most chunk_seconds (equal lengths, so none is a short tail)
def chunks(samples, chunk_seconds):
    count = max(1, math.ceil(len(samples) / (chunk_seconds * SAMPLE_RATE)))
    size = math.ceil(len(samples) / count)
    return [samples[ix:ix + size] for ix in range(0, len(samples), size)] or [samples]


# The emotion2vec+ label of a class (its labels are "<Chinese>/<English>")
def emotion_label(label):
    english = label.split('/')[-1]
    return 'unknown' if english == '<unk>' else english


# Score one file with the loaded models - returns its result object
def score_file(path, models, args):
    import numpy  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    samples, seconds = read_audio(path)
    result = {'file': path, 'seconds': round(seconds, 3)}
    parts = chunks(samples, args.chunk_seconds)
    weights = numpy.array([len(part) for part in parts], dtype=numpy.float64)
    weights /= weights.sum()
    batches = [torch.from_numpy(part).unsqueeze(0).to(args.device) for part in parts]
    with torch.inference_mode():
        if 'mos' in models:
            scores = [models['mos'](batch, SAMPLE_RATE).item() for batch in batches]
            result['mos'] = round(float(numpy.dot(weights, scores)), 4)
        if 'embedding' in models:
            embeddings = [models['embedding'].encode_batch(batch).squeeze().cpu().numpy() for batch in batches]
            embedding = numpy.average(numpy.stack(embeddings), axis=0, weights=weights)
            result['embedding'] = [round(float(value), 5) for value in embedding]
        if 'emotion' in models:
            probabilities = {}
            for weight, part in zip(weights, parts):
                output = to_stderr(models['emotion'].generate, part, granularity='utterance', extract_embedding=False,
                                   disable_pbar=True)[0]
                for label, score in zip(output['labels'], output['scores']):
                    name = emotion_label(label)
                    probabilities[name] = probabilities.get(name, 0) + weight * float(score)
            result['emotion'] = {name: round(score, 5) for name, score in probabilities.items()}
    return result


# Load the requested models and score every file, writing JSON lines to stdout
def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    quiet_libraries()
    import torch  # pylint: disable=import-outside-toplevel
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    models = {}
    if not args.no_mos:
        models['mos'] = load_utmos(args.device)
    if not args.no_embedding:
        models['embedding'] = load_ecapa(args.device)
    if not args.no_emotion:
        models['emotion'] = load_emotion(args.emotion_model, args.device)
    output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    for path in args.files:
        output.write(json.dumps(score_file(path, models, args)) + '\n')
    output.flush()


if __name__ == '__main__':
    main()
