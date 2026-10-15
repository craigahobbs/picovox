# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The neural-network experiment - a small network that maps our front end's phoneme tokens to durations and
interpretable source-filter parameters, trained end to end through a differentiable synthesizer against recordings.

- nn/frontend.py - the token vocabulary (nn/vocab.json) and a Python port of lib/lexicon.js textToPhrases
- nn/data.py - the training data (CMU ARCTIC slt and bdl) under build/nn/data/, its loader, and its statistics
- nn/js/tokens.js - the same tokens in JavaScript, from lib/lexicon.js, for browser-side inference

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
- nn/export.py - 8-bit export to data/nn/<voice>.bin and .json; nn/js/model.js - the forward pass in JavaScript;
  nn/js/transforms.js - the voice and style transforms; nn/js/speak.js - text to WAV with a neural voice (Node)
- nn/synthesize.py - inference in PyTorch; nn/parity.py - nn/js/model.js against PyTorch, the quantization, and speed
- nn/evaluate.py - the neural voices against the rule voices, the recordings, and the analysis-by-synthesis ceiling;
  nn/samples.py - listening samples and the transforms demonstration (build/nn/samples/)
"""
