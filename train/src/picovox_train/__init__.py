# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The neural voice's training pipeline - a small network that maps our front end's phoneme tokens to durations and
interpretable source-filter parameters, trained end to end through a differentiable synthesizer against the training
speech - a local TTS model's, generated at train time and measured with our own tools - and the G2P network that
pronounces every word. make train (the Makefile) runs it from the recipe, train/data/train.json.

The front end and the G2P:

- frontend.py - the token vocabulary (vocab.json) and a Python port of lib/lexicon.js textToPhrases, every word
  pronounced by the app's G2P network (g2pmodel.py - lib/g2p.js's inference, bit for bit, on lib/models/g2pModel.js)
- g2p.py - the G2P network's training, the recipe in train/data/train.json; heteronyms.py - its relabeled
  words from the training speech's measured readings (train/data/g2p-relabel.json); g2paudit.py - the dictionary's
  label errors, by cross-validated disagreement and the training speech's audio (train/data/g2p-fixes.json)
- context.py - the heteronyms by context: the G2P's reading input and ambiguity head, and the context network
  (picovox_train.context export writes lib/models/g2pModel.js); contextlabels.py - their labels (train/data/context-labels.json);
  readings.py - the dictionary's variants clustered into readings, how two readings are told apart in the audio, the
  measured words and the corpus's coverage of them, and the shared phone helpers
- lib/tokens.js - the same tokens in JavaScript, from lib/lexicon.js, for browser-side inference

The data:

- tts.py - the training speech: book sentences (the heteronym coverage, then the distillation's books) spoken by a
  local TTS model through Ollama (the recipe's tts section - its model pinned by digest, a voice for each of the
  package's voices), cached in downloads/tts/<model>-<digest>/, CTC-aligned, checked, and built in data.py's format
  (build/nn/data/tts/)
- ctc.py - our speech measurement: a wav2vec2 phoneme recognizer's CTC posteriors - the forced alignment, the greedy
  phone checks, and the candidate vote between pronunciations; measure.py - the training speech's heteronym tokens'
  readings, by the CTC vote or the vowels' formants, with each method's controls
- data.py - the training data's featurization and format (tokens, durations from the alignment - align.py - 24 kHz
  audio, and Praat's auxiliary tracks), and its loader (CorpusData)
- wav.py - WAV files (16-bit PCM) read with the standard library, exactly as libsndfile reads them
- books.py - the pinned book texts (train/data/books.json): the Project Gutenberg books the training speech, the
  heteronym labels, and the distillation read, each fetched once and verified by its SHA-256 - a revised text is never
  used
- corpora.py - the pinned downloads: the CMU Pronouncing Dictionary and the listener panel's Whisper model

The synthesizer:

- params.py - the parameter set (32 interpretable source-filter parameters per 5 ms frame, their units, ranges, and
  network normalization), and the channel equalizer; lib/params.js is generated from it
  (python -m picovox_train.params --write-js)
- dsynth.py - the differentiable synthesizer (PyTorch, batched, any sample rate) - harmonic voicing through the
  tract's analytic response, and STFT-filtered noise; lf.py (the LF pulse's harmonic table) and rng.py (the
  counter-based noise shared with JavaScript)
- spectral.py - the STFT, log-mel spectra, and the log-spectral distance
- lib/render.js - the same synthesizer in JavaScript (browser and Node, streaming); lib/lf.js, lib/rng.js,
  lib/fft.js

The network:

- model.py's MultiModel and trainmulti.py - the multi-speaker model: a speaker table (a row per voice) conditioning
  the network (FiLM), trained on the training speech with an adversarial spectrogram critic (discriminator.py)
  against the over-smoothing; the losses are losses.py's
- optim.py - the training step's options for trainmulti.py and distill.py: the lean gradient clip and AdamW step,
  Muon for the hidden matrices (batched Newton-Schulz - the student's optimizer), and the warmup-stable-decay schedule
- quant.py - the export's 8-bit weight quantization (per row), shared by export.py and lib/model.js
- distill.py - knowledge distillation of the multi-speaker model into small students (train/data/train.json's distill)
- export.py - export to <voice>.bin and .json (8-bit weights - quant.py); package.py - an export
  as a generated JavaScript module (lib/models/voiceModel.js), loaded by lib/package.js (tools/packagecheck.js checks
  it); lib/model.js - the forward pass in JavaScript; lib/transforms.js - the voice and style transforms;
  tools/speak.js - text to WAV (Node)

The pipeline's tools:

- panel.py - the listener panel's scoring (UTMOS, Whisper's WER, speaker similarity) of an exported model, shared by
  every evaluation
- smoke.py - every training entry point for a few steps on the CPU (the pipeline's smoke test)
- quality.py - voice quality measures (spectral balance, band harmonicity, upper-band crest, pitch flutter and wobble)
  of the package's voices synthesized, against their source clips (the training speech) - guides for listening
  (tools/voicelab/)
- stepcheck.py - the trainer's step-0 losses on fixed batches, exactly, to check a change leaves training unchanged
- gpulock.py - a command run holding the GPU lock (make train's GPU stages one at a time)
- trained.py - make train's trained models: its report (each model's size and scores, and whether it differs from the
  committed module) and make train-install (the trained modules that differ into lib/models/)

(The rule-based synthesizer's bridge and comparisons, the single-speaker model and trainer, the spectral correction
head, the all-convolution encoder, the size search, the finished experiments' evaluations, and the training on human
recordings - CMU ARCTIC, LibriTTS-R and its MFA alignments, CREMA-D, the speaker pool, before 1.7 - are in git history.)
"""

import json
import os


# The repository root (train/src/picovox_train/ is three levels below it) - the data, the corpora, the build outputs,
# and the package's modules (lib/) the pipeline reads
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# make train's recipe - the training settings of every stage
RECIPE_PATH = os.path.join(ROOT, 'train', 'data', 'train.json')

# Read a JSON file - the training's data and settings (the tests give objects for some - tests/helpers.py json_files)
def read_json(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


# The package's source directory (train/src) - the PYTHONPATH of the pipeline's Python subprocesses
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
