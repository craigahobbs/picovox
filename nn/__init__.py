# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The neural-network experiment - a small network that maps our front end's phoneme tokens to durations and
interpretable source-filter parameters, trained end to end through a differentiable synthesizer against recordings.

- nn/frontend.py - the token vocabulary (nn/vocab.json) and a Python port of lib/lexicon.js textToPhrases, every word
  pronounced by the app's G2P network (nn/g2pmodel.py - lib/g2p.js's inference, bit for bit, on lib/models/g2pModel.js)
- nn/g2p.py - the G2P network's training and export (lib/models/g2pModel.js), the recipe in data/nn/train.json;
  nn/heteronyms.py - its relabeled words from readers' usage (data/nn/g2p-relabel.json)
- nn/data.py - the training data (CMU ARCTIC slt and bdl) under build/nn/data/, its loader, and its statistics
- nn/js/tokens.js - the same tokens in JavaScript, from lib/lexicon.js, for browser-side inference
- nn/cremad.py - emotional speech for the style modifiers: CREMA-D fetched, force-aligned (a wav2vec2 phoneme CTC
  recognizer), and built under build/nn/data/cremad/ in nn/data.py's format with emotion and intensity labels; the
  listener panel's emotion recognition of it (data/nn/cremad.json)

The synthesizer:

- nn/params.py - the parameter set (32 interpretable source-filter parameters per 5 ms frame, their units, ranges, and
  network normalization), the voice and style transforms, and the parameter file format; nn/js/params.js is generated
  from it (python -m nn.params --write-js)
- nn/dsynth.py - the differentiable synthesizer (PyTorch, batched, any sample rate) - harmonic voicing through the
  tract's analytic response, and STFT-filtered noise; nn/lf.py (the LF pulse's harmonic table) and nn/rng.py (the
  counter-based noise shared with JavaScript)
- nn/spectral.py - the multi-resolution STFT loss, log-mel spectra, and the log-spectral distance
- nn/js/render.js - the same synthesizer in JavaScript (browser and Node, streaming); nn/js/lf.js, nn/js/rng.js,
  nn/js/fft.js; nn/js/rendercli.js renders a parameter file to WAV
- nn/js/bridge.js - lib/speak.js's Klatt frames to the parameter set (the rule voice as a prior and a baseline);
  nn/js/speakparams.js speaks text to parameter files and WAVs
- nn/equiv.py - the Python and JavaScript synthesizers compared; nn/bridge.py - the rule voice through the parameter
  set compared with lib/klatt.js; nn/fit.py - analysis by synthesis, parameter trajectories fit to recordings (the
  synthesizer's ceiling)

The network:

- nn/model.py - the model: tokens to durations to parameter frames (non-autoregressive, ~0.75M parameters)
- nn/train.py - training end to end through nn/dsynth.py (checkpoints under build/nn/runs/)
- nn/model.py's MultiModel and nn/trainmulti.py - the multi-speaker, multi-style model (~0.95M parameters): a speaker
  table (blendable voices) and a style table (emotions scaled by intensity) conditioning the network (FiLM), trained
  on the American reader pool and CREMA-D with an adversarial spectrogram critic against the over-smoothing;
  nn/export.py writes data/nn/multi.*, which nn/js/model.js speaks with a speaker blend and a style
- nn/evalmulti.py - the multi-speaker model's evaluation and samples (build/nn/samples-multi/)
- nn/export.py - 8-bit export to data/nn/<voice>.bin and .json; nn/js/model.js - the forward pass in JavaScript;
  nn/js/transforms.js - the voice and style transforms; nn/js/speak.js - text to WAV with a neural voice (Node)
- nn/synthesize.py - inference in PyTorch; nn/parity.py - nn/js/model.js against PyTorch, the quantization, and speed
- nn/evaluate.py - the neural voices against the rule voices, the recordings, and the analysis-by-synthesis ceiling;
  nn/samples.py - listening samples and the transforms demonstration (build/nn/samples/)

Architecture options and model size (each off by default):

- nn/model.py's options - the spectral correction head (correctionBands; nn/dsynth.py and nn/js/render.js apply it)
  and the all-convolution encoder (encoder 'conv'); nn/trainmulti.py --correction-bands, --encoder
- nn/quant.py - weight quantization at 8-4 bits, per row or per group, shared by nn/export.py (--bits, --group),
  quantization-aware training (nn/trainmulti.py --qat-bits), and nn/js/model.js
- nn/package.py - an export as a generated JavaScript module, loaded by nn/js/package.js (nn/js/packagecheck.js checks it)
- nn/archeval.py - a checkpoint at an export precision: validation, the panel (UTMOS, WER, speaker similarity), size;
  nn/sizesearch.py - the minimal-size search over widths, depths, encoders, and bit widths (Pareto front)
- nn/stepcheck.py - the trainer's step-0 losses on fixed batches, exactly, to check a change leaves training unchanged

Shrinking and checking:

- nn/distill.py - knowledge distillation of the multi-speaker model into small students (data/nn/distill.json)
- nn/smoke.py - every training entry point for a few steps on the CPU (the pipeline's smoke test)
"""
