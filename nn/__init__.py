# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The neural voice's training pipeline - a small network that maps our front end's phoneme tokens to durations and
interpretable source-filter parameters, trained end to end through a differentiable synthesizer against recordings,
and the G2P network that pronounces every word. make train (the Makefile) runs it from the recipe, data/nn/train.json.

The front end and the G2P:

- nn/frontend.py - the token vocabulary (nn/vocab.json) and a Python port of lib/lexicon.js textToPhrases, every word
  pronounced by the app's G2P network (nn/g2pmodel.py - lib/g2p.js's inference, bit for bit, on lib/models/g2pModel.js)
- nn/g2p.py - the G2P network's training and export, the recipe in data/nn/train.json; nn/heteronyms.py - its relabeled
  words from readers' usage (data/nn/g2p-relabel.json)
- nn/context.py - the heteronyms by context: the G2P's reading input and ambiguity head, and the context network
  (nn.context export writes lib/models/g2pModel.js); nn/contextlabels.py - their labels (data/nn/context-labels.json);
  nn/readings.py - a heteronym token's reading from the forced alignment, for the voice data
- nn/js/tokens.js - the same tokens in JavaScript, from lib/lexicon.js, for browser-side inference

The data:

- nn/data.py - the training data's featurization and format (tokens, durations reconciled with the corpus's phone
  labels - nn/align.py - 24 kHz audio, and Praat's auxiliary tracks), and its loader (ArcticData)
- nn/multi.py - the American reader pool (LibriTTS-R readers - accents by nn/accent.py - and the CMU ARCTIC speakers)
  in nn/data.py's format, build/nn/data/multi/, its speaker table data/nn/speakers.json
- nn/cremad.py - emotional speech for the style modifiers: CREMA-D fetched, force-aligned (a wav2vec2 phoneme CTC
  recognizer), and built under build/nn/data/cremad/ in nn/data.py's format with emotion and intensity labels; the
  listener panel's emotion recognition of it (data/nn/cremad.json)

The synthesizer:

- nn/params.py - the parameter set (32 interpretable source-filter parameters per 5 ms frame, their units, ranges, and
  network normalization), the channel equalizer, and the parameter file format; nn/js/params.js is generated from it
  (python -m nn.params --write-js)
- nn/dsynth.py - the differentiable synthesizer (PyTorch, batched, any sample rate) - harmonic voicing through the
  tract's analytic response, and STFT-filtered noise; nn/lf.py (the LF pulse's harmonic table) and nn/rng.py (the
  counter-based noise shared with JavaScript)
- nn/spectral.py - the STFT, log-mel spectra, and the log-spectral distance
- nn/js/render.js - the same synthesizer in JavaScript (browser and Node, streaming); nn/js/lf.js, nn/js/rng.js,
  nn/js/fft.js

The network:

- nn/model.py's MultiModel and nn/trainmulti.py - the multi-speaker, multi-style model: a speaker table (blendable
  voices) and a style table (emotions scaled by intensity) conditioning the network (FiLM), trained on the American
  reader pool and CREMA-D with an adversarial spectrogram critic (nn/discriminator.py) against the over-smoothing; the
  losses are nn/losses.py's
- nn/quant.py - weight quantization at 8-4 bits, per row or per group, shared by nn/export.py (--bits, --group),
  quantization-aware training (nn/trainmulti.py --qat-bits), and nn/js/model.js
- nn/distill.py - knowledge distillation of the multi-speaker model into small students (data/nn/train.json's distill)
- nn/export.py - export to <voice>.bin and .json (8-bit, or nn/quant.py's packed formats); nn/package.py - an export
  as a generated JavaScript module (lib/models/voiceModel.js), loaded by nn/js/package.js (nn/js/packagecheck.js checks
  it); nn/js/model.js - the forward pass in JavaScript; nn/js/transforms.js - the voice and style transforms;
  nn/js/voices.js - the designed voices (data/nn/voices.json; nn/js/voicescheck.js checks them); nn/js/speak.js - text
  to WAV (Node)
- nn/synthesize.py - inference in PyTorch; nn/parity.py - nn/js/model.js against PyTorch, the quantization, and speed

The pipeline's tools:

- nn/panel.py - the listener panel's scoring (UTMOS, Whisper's WER, speaker similarity) of an exported model, shared by
  every evaluation
- nn/smoke.py - every training entry point for a few steps on the CPU (the pipeline's smoke test)
- nn/stepcheck.py - the trainer's step-0 losses on fixed batches, exactly, to check a change leaves training unchanged
- nn/gpulock.py - a command run holding the GPU lock (make train's GPU stages one at a time)

(The rule-based synthesizer's bridge and comparisons, the single-speaker model and trainer, the spectral correction
head, the all-convolution encoder, the size search, and the finished experiments' evaluations are in git history.)
"""
