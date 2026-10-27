# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
Inference in PyTorch - text to parameter frames with an exported voice's weights (export.py's quantized_model), the
same steps as lib/model.js: tokens (frontend.py), split into sentences (each with its own <bos>), rare tokens remapped
(the export manifest's remap), durations, frames. The export's parity check (tests/test_export.py) compares
lib/model.js with it.

The voice is a speaker (a key or an id of the speaker table), and its style an emotion (model.py's STYLE_NAMES, or
neutral) at an intensity (0-1).
"""

import numpy
import torch

from . import params
from .distill import sentences
from .export import load_model, quantized_model, token_remap, training_token_counts
from .frontend import tokenize
from .model import STYLE_NAMES


class Voice:
    """An exported model ready for inference - the model (its weights quantized as exported), its remap, and its channel
    equalizer"""

    def __init__(self, checkpoint, device='cpu', speaker=0, emotion=None, intensity=1.0):
        model, state = load_model(checkpoint, device)
        self.model = quantized_model(model).eval()
        counts, self.vocab = training_token_counts(state)
        remap = token_remap(self.vocab, counts)
        self.remap = {self.vocab.index(source): self.vocab.index(target) for source, target in remap.items()}
        self.device = device
        self.speaker = speaker_id(state, speaker)
        self.speaker_input = torch.tensor([self.speaker], device=device)
        self.emotion = torch.tensor([STYLE_NAMES.index(emotion) if emotion in STYLE_NAMES else -1], device=device)
        self.intensity = torch.tensor([float(intensity)], device=device)
        with torch.no_grad():
            self.eq = self.model.eq(self.speaker_input)[0].cpu().numpy()

    def frames(self, text):
        """Text to parameter frames [T, COUNT] (numpy, the end frame included) and the sentences' token ids and
        durations"""
        ids = tokenize(text)['ids']
        chunks = []
        parts = []
        for sentence in sentences(ids, self.vocab):
            mapped = [self.remap.get(token_id, token_id) for token_id in sentence]
            tokens = torch.tensor([mapped], device=self.device)
            with torch.no_grad():
                result = self.model.infer(tokens, speaker=self.speaker_input, emotion=self.emotion, intensity=self.intensity)
            parts.append({'ids': sentence, 'durations': result['durations'][0].cpu().numpy().tolist()})
            chunks.append(result['values'][0, :-1].cpu().numpy())
        if not chunks:
            return numpy.zeros((0, params.COUNT)), parts
        values = numpy.concatenate(chunks)
        return numpy.concatenate([values, values[-1:]]), parts


# A speaker's row in a checkpoint's speaker table - from an id (int or numeric string) or a key
def speaker_id(state, speaker):
    if isinstance(speaker, int) or (isinstance(speaker, str) and speaker.isdigit()):
        return int(speaker)
    return state['speakers'].index(speaker)
