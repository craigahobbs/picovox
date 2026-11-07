# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The perceptual loss - the rendered crops against the recordings in a pretrained speech model's features: the pinned
wav2vec2 phoneme recognizer's (ctc.py), frozen, its transformer cut after the deepest layer used. Each crop is
resampled to 16 kHz (differentiably - wav.resample, a windowed-sinc filter) and normalized as the recognizer expects; the
loss is the mean over LAYERS of each frame's cosine distance between the two crops' features (the recording's without
gradients) - the same features, on lower layers, that ranked the per-clip synthesizer fit far closer to the source than
the network (the paper's Section 13.9), and not the layer the gap is judged on (quality.py's fdL12), so the loss can't
simply learn the measure.

trainmulti.py adds it to the training step's loss (--perceptual-weight - the recipe's 5, on the first
--perceptual-crops crops of each step; --perceptual-precision bfloat16 runs the recognizer under autocast, as the critic's
convolutions run - the features compared in float32).
"""

import torch

from . import ctc
from .data import SAMPLE_RATE
from .wav import resample


# The layers whose features the loss compares (the transformer's, 1-based)
LAYERS = (3, 6)


class PerceptualLoss:
    """The perceptual loss on a device - __call__(audio, target) of crops [batch, samples] at the corpus's rate
    (data.py SAMPLE_RATE)"""

    def __init__(self, device, layers=LAYERS, dtype=None):
        model, _, _ = ctc.load_ctc(device)
        self.layers = tuple(layers)
        self.dtype = dtype
        self.device_type = torch.device(device).type
        self.encoder = model.wav2vec2
        # (the transformer cut after the deepest layer used - the rest is never run)
        self.encoder.encoder.layers = self.encoder.encoder.layers[:max(self.layers)]
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    # A batch's features - [layer features [batch, frames, dims]] (float32; the recognizer under autocast in dtype, if any)
    def features(self, audio):
        audio = resample(audio, SAMPLE_RATE, ctc.CTC_SAMPLE_RATE)
        audio = (audio - audio.mean(1, keepdim=True)) / (audio.var(1, keepdim=True) + 1e-7).sqrt()
        with torch.autocast(self.device_type, dtype=self.dtype or torch.bfloat16, enabled=self.dtype is not None):
            hidden = self.encoder(audio, output_hidden_states=True).hidden_states
        return [hidden[layer].float() for layer in self.layers]

    def __call__(self, audio, target):
        with torch.no_grad():
            targets = self.features(target)
        distances = [1 - torch.nn.functional.cosine_similarity(rendered, wanted, dim=-1).mean()
                     for rendered, wanted in zip(self.features(audio), targets)]
        return sum(distances) / len(distances)
