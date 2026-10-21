# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The neural voice's training pipeline - a small network that maps our front end's phoneme tokens to durations and
interpretable source-filter parameters, trained end to end through a differentiable synthesizer against recordings,
and the G2P network that pronounces every word. make train (the Makefile) runs it from the recipe, data/nn/train.json.

The front end and the G2P:

- frontend.py - the token vocabulary (vocab.json) and a Python port of lib/lexicon.js textToPhrases, every word
  pronounced by the app's G2P network (g2pmodel.py - lib/g2p.js's inference, bit for bit, on lib/models/g2pModel.js)
- g2p.py - the G2P network's training and export, the recipe in data/nn/train.json; heteronyms.py - its relabeled
  words from readers' usage (data/nn/g2p-relabel.json); g2paudit.py - the dictionary's label errors, by cross-validated
  disagreement and the readers' audio (data/nn/g2p-fixes.json)
- context.py - the heteronyms by context: the G2P's reading input and ambiguity head, and the context network
  (picovox_train.context export writes lib/models/g2pModel.js); contextlabels.py - their labels (data/nn/context-labels.json);
  readings.py - a heteronym token's reading from the forced alignment, for the voice data
- style.py - the style network (the package's style "auto" - a quotation's angry, shouting, or fearful words): the pinned
  books' quotations, a teacher's labels (data/nn/style-labels.json), the network's training (pretrained on the
  attributions' words, distilled from the teacher), and its export (lib/models/styleModel.js)
- lib/tokens.js - the same tokens in JavaScript, from lib/lexicon.js, for browser-side inference

The data:

- data.py - the training data's featurization and format (tokens, durations reconciled with the corpus's phone
  labels - align.py - 24 kHz audio, and Praat's auxiliary tracks), and its loader (ArcticData)
- multi.py - the American reader pool (LibriTTS-R readers and the CMU ARCTIC speakers)
  in data.py's format, build/nn/data/multi/, its speaker table data/nn/speakers.json
- books.py - the pinned book texts (data/nn/books.json): the Project Gutenberg books the heteronym labels and the
  distillation read, each fetched once and verified by its SHA-256 - a revised text is never used
- cremad.py - emotional speech for the style modifiers: CREMA-D fetched, force-aligned (a wav2vec2 phoneme CTC
  recognizer), and built under build/nn/data/cremad/ in data.py's format with emotion and intensity labels; the
  listener panel's emotion recognition of it (data/nn/cremad.json)

The synthesizer:

- params.py - the parameter set (32 interpretable source-filter parameters per 5 ms frame, their units, ranges, and
  network normalization), the channel equalizer, and the parameter file format; lib/params.js is generated from it
  (python -m picovox_train.params --write-js)
- dsynth.py - the differentiable synthesizer (PyTorch, batched, any sample rate) - harmonic voicing through the
  tract's analytic response, and STFT-filtered noise; lf.py (the LF pulse's harmonic table) and rng.py (the
  counter-based noise shared with JavaScript)
- spectral.py - the STFT, log-mel spectra, and the log-spectral distance
- lib/render.js - the same synthesizer in JavaScript (browser and Node, streaming); lib/lf.js, lib/rng.js,
  lib/fft.js

The network:

- model.py's MultiModel and trainmulti.py - the multi-speaker, multi-style model: a speaker table (blendable
  voices) and a style table (emotions scaled by intensity) conditioning the network (FiLM), trained on the American
  reader pool and CREMA-D with an adversarial spectrogram critic (discriminator.py) against the over-smoothing; the
  losses are losses.py's
- optim.py - the training step's options for trainmulti.py and distill.py (all off by default): the lean gradient
  clip and AdamW step, Muon for the hidden matrices (batched Newton-Schulz), the flat EMA of the weights, the
  warmup-stable-decay schedule, and zero-initialized residual projections
- quant.py - weight quantization at 8-4 bits, per row or per group, shared by export.py (--bits, --group),
  quantization-aware training (trainmulti.py --qat-bits), and lib/model.js
- distill.py - knowledge distillation of the multi-speaker model into small students (data/nn/train.json's distill)
- export.py - export to <voice>.bin and .json (8-bit, or quant.py's packed formats); package.py - an export
  as a generated JavaScript module (lib/models/voiceModel.js), loaded by lib/package.js (nn/js/packagecheck.js checks
  it); lib/model.js - the forward pass in JavaScript; lib/transforms.js - the voice and style transforms;
  nn/js/voices.js - the designed voices (data/nn/voices.json; nn/js/voicescheck.js checks them); nn/js/speak.js - text
  to WAV (Node)
- synthesize.py - inference in PyTorch

The pipeline's tools:

- panel.py - the listener panel's scoring (UTMOS, Whisper's WER, speaker similarity) of an exported model, shared by
  every evaluation
- smoke.py - every training entry point for a few steps on the CPU (the pipeline's smoke test)
- stepcheck.py - the trainer's step-0 losses on fixed batches, exactly, to check a change leaves training unchanged
- gpulock.py - a command run holding the GPU lock (make train's GPU stages one at a time)

(The rule-based synthesizer's bridge and comparisons, the single-speaker model and trainer, the spectral correction
head, the all-convolution encoder, the size search, and the finished experiments' evaluations are in git history.)
"""

import os


# The repository root (train/src/picovox_train/ is three levels below it) - the data, the corpora, the build outputs,
# and the package's modules (lib/) the pipeline reads
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# The package's source directory (train/src) - the PYTHONPATH of the pipeline's Python subprocesses
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
