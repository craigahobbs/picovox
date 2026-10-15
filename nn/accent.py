# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
English accent identification of the corpus speakers - the multi-speaker pool's accent filter (nn/multi.py).

Each speaker's accent is classified with the CommonAccent ECAPA-TDNN model (SpeechBrain,
Jzuluaga/accent-id-commonaccent_ecapa - 16 English accents of Common Voice: us, england, canada, australia, indian,
scotland, ireland, african, malaysia, newzealand, southatlandtic, bermuda, philippines, hongkong, wales, singapore) on
a few of the speaker's utterances (16 kHz); the per-utterance posteriors are averaged. It runs in the listener-model
environment (build/venv-listeners), on the CPU by default (inference only - a few minutes for 300 speakers):

    build/venv-listeners/bin/python -B -m nn.accent [--count 8] [--corpus libritts-train-clean-100 ...]

It writes build/nn/multi/accent.json - {speaker key: {corpus, speaker, utterances, posterior: {accent: p}, top,
votes: {accent: count}}}; speaker keys are "arctic-<name>" and "libritts-<id>". Speakers already classified are kept
(the file is updated incrementally), so a re-run only classifies new speakers.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
import warnings

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA_DIR = os.path.join(ROOT, 'data', 'corpora')
LIBRITTS_DIR = os.path.join(CORPORA_DIR, 'libritts_r', 'LibriTTS_R')
ARCTIC_DIR = os.path.join(CORPORA_DIR, 'arctic')
OUT_PATH = os.path.join(ROOT, 'build', 'nn', 'multi', 'accent.json')
CACHE_DIR = os.path.join(ROOT, 'build', 'listeners')
ACCENT_REPO = 'Jzuluaga/accent-id-commonaccent_ecapa'

# The model caches, set before the model libraries are imported (shared with tools/listeners.py)
os.environ.setdefault('TORCH_HOME', os.path.join(CACHE_DIR, 'torch'))
os.environ.setdefault('HF_HOME', os.path.join(CACHE_DIR, 'hf'))
os.environ.setdefault('TQDM_DISABLE', '1')

# The corpora classified - name: (kind, LibriTTS subset directory)
CORPORA = {
    'arctic': ('arctic', None),
    'libritts-dev-clean': ('libritts', 'dev-clean'),
    'libritts-train-clean-100': ('libritts', 'train-clean-100')
}

# The scale of the classifier's additive angular margin softmax (the CommonAccent recipe's, SpeechBrain's default)
AAM_SCALE = 30.0

# The utterances chosen per speaker - between these lengths (seconds), by a hash of the utterance id
MIN_SECONDS = 3.0
MAX_SECONDS = 15.0


# The speakers of a corpus - [(key, corpus, speaker, [wav paths])]
def corpus_speakers(corpus):
    kind, subset = CORPORA[corpus]
    speakers = []
    if kind == 'arctic':
        if not os.path.isdir(ARCTIC_DIR):
            return speakers
        for directory in sorted(os.listdir(ARCTIC_DIR)):
            wav_dir = os.path.join(ARCTIC_DIR, directory, 'wav')
            if os.path.isdir(wav_dir):
                name = directory.split('_')[2]
                speakers.append((f'arctic-{name}', corpus, name,
                                 [os.path.join(wav_dir, item) for item in sorted(os.listdir(wav_dir)) if item.endswith('.wav')]))
        return speakers
    base = os.path.join(LIBRITTS_DIR, subset)
    if not os.path.isdir(base):
        return speakers
    for speaker in sorted(os.listdir(base), key=lambda item: int(item) if item.isdigit() else 0):
        speaker_dir = os.path.join(base, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        wavs = []
        for dirpath, _dirnames, filenames in os.walk(speaker_dir):
            wavs.extend(os.path.join(dirpath, name) for name in filenames if name.endswith('.wav'))
        speakers.append((f'libritts-{speaker}', corpus, speaker, sorted(wavs)))
    return speakers


# Choose a speaker's utterances - up to count, between MIN_SECONDS and MAX_SECONDS, by a hash of the file name
def choose_utterances(wavs, count):
    import soundfile  # pylint: disable=import-outside-toplevel
    ordered = sorted(wavs, key=lambda path: hashlib.sha1(os.path.basename(path).encode('utf-8')).hexdigest())
    chosen = []
    for path in ordered:
        if MIN_SECONDS <= soundfile.info(path).duration <= MAX_SECONDS:
            chosen.append(path)
            if len(chosen) == count:
                break
    return chosen or ordered[:count]


# Load the accent classifier
def load_classifier(device):
    from speechbrain.inference.classifiers import EncoderClassifier  # pylint: disable=import-outside-toplevel
    import huggingface_hub  # pylint: disable=import-outside-toplevel
    with contextlib.redirect_stdout(sys.stderr):
        try:
            source = huggingface_hub.snapshot_download(ACCENT_REPO, local_files_only=True)
        except Exception:  # pylint: disable=broad-exception-caught
            source = huggingface_hub.snapshot_download(ACCENT_REPO)
        return EncoderClassifier.from_hparams(source=source, savedir=os.path.join(CACHE_DIR, 'accent-id-commonaccent_ecapa'),
                                              run_opts={'device': device})


# The accent posteriors of a wav file - (posterior array, labels)
def classify(classifier, path):
    import soundfile  # pylint: disable=import-outside-toplevel
    import soxr  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    samples, rate = soundfile.read(path, dtype='float32')
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        samples = soxr.resample(samples, rate, 16000, quality='HQ').astype(np.float32)
    with torch.no_grad():
        out_prob, _score, _index, _labels = classifier.classify_batch(torch.from_numpy(samples)[None, :])
    # The ECAPA classifier's outputs are cosine similarities to the class weights, trained with an additive angular
    # margin softmax at scale AAM_SCALE - the posterior is their scaled softmax
    logits = AAM_SCALE * out_prob[0].cpu().numpy().astype(np.float64)
    posterior = np.exp(logits - logits.max())
    return posterior / posterior.sum()


# Classify the speakers of the chosen corpora, updating build/nn/multi/accent.json
def main(argv=None):
    parser = argparse.ArgumentParser(prog='nn.accent', description='Classify the corpus speakers\' English accents')
    parser.add_argument('--corpus', nargs='+', default=list(CORPORA), choices=list(CORPORA))
    parser.add_argument('-n', '--count', type=int, default=8, help='utterances per speaker')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    parser.add_argument('--redo', action='store_true', help='reclassify speakers already in the results')
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    results = {}
    if os.path.exists(OUT_PATH) and not args.redo:
        with open(OUT_PATH, encoding='utf-8') as file:
            results = json.load(file)
    classifier = load_classifier(args.device)
    labels = [classifier.hparams.label_encoder.decode_ndim(index) for index in range(len(classifier.hparams.label_encoder))]
    start = time.time()
    for corpus in args.corpus:
        for key, corpus_name, speaker, wavs in corpus_speakers(corpus):
            if key in results or not wavs:
                continue
            paths = choose_utterances(wavs, args.count)
            posteriors = np.array([classify(classifier, path) for path in paths])
            mean = posteriors.mean(axis=0)
            votes = {}
            for row in posteriors:
                votes[labels[int(np.argmax(row))]] = votes.get(labels[int(np.argmax(row))], 0) + 1
            results[key] = {
                'corpus': corpus_name,
                'speaker': speaker,
                'utterances': [os.path.relpath(path, ROOT) for path in paths],
                'posterior': {label: round(float(value), 4) for label, value in sorted(zip(labels, mean), key=lambda item: -item[1])},
                'top': labels[int(np.argmax(mean))],
                'votes': dict(sorted(votes.items(), key=lambda item: -item[1]))
            }
            print(f'{key:16s} {results[key]["top"]:12s} us {results[key]["posterior"].get("us", 0):.3f} '
                  f'votes {results[key]["votes"]} ({time.time() - start:.0f} s)', file=sys.stderr, flush=True)
            os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
            with open(OUT_PATH + '.tmp', 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=1)
                file.write('\n')
            os.replace(OUT_PATH + '.tmp', OUT_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
