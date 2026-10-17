# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The network - a small non-autoregressive (FastSpeech/LightSpeech-like) model from our front end's phoneme tokens
(vocab.json) to per-token durations and then to frame-by-frame interpretable synthesizer parameters (params.py),
which dsynth.py (training) and lib/render.js (the browser) render.

    tokens [B, N] --embedding--> encoder (conv-FFN + self-attention blocks) --> token features [B, N, D]
        --> duration predictor: log(duration + 1) per token, and for the boundary tokens (word boundaries, phrase
            ends) a pause probability and a paused duration
        --> length regulator (each token's features repeated for its duration in 10 ms frames, plus the frame's
            position within its token)
        --> decoder (conv-FFN blocks) --> parameter head: two 5 ms parameter frames per 10 ms frame, each parameter a
            sigmoid mapped to its range by params.py's denormalize

The outputs stay interpretable - F0, the voicing and noise levels, Rd, the shelf, the formants and bandwidths, the nasal
pole and zero, the frication spectrum - so the voice and style transforms of params.py apply to them. Jitter and
shimmer are constants (the spectral loss gives jitter no gradient), and a static per-speaker channel equalizer
(params.py's EQ, 24 bands) models the recording channel.

F0 gets no useful gradient through the synthesizer (dsynth.py), so the F0 column is trained by regression on the
recordings' Praat F0 (trainmulti.py, losses.py), and training renders with the recorded F0 (teacher forcing, as
FastPitch does).

Conditioning: MultiModel's condition vector - a speaker embedding (a row, or a blend of rows, of the speaker table)
and a style embedding (an emotion row of the style table, scaled by its intensity) - is added to the token features,
the duration predictor's input, and the decoder's input, and modulates every conv-FFN block (FiLM). (The first,
single-speaker model and its trainer, nn/train.py, are in git history.)

No absolute positional encoding - order comes from the convolutions - so any length works; synthesize.py and
lib/model.js speak long texts a sentence at a time, like the training data.

Every layer is one lib/model.js implements: embedding, linear, depthwise 1-D convolution ('same' padding), layer
norm, SiLU, tanh, and multi-head self-attention with a key padding mask. (The spectral correction head and the
all-convolution encoder, architecture options that weren't adopted, are in git history.)
"""

import math

import torch

from . import params


# The model's hyperparameters (config 'kind': 'multi')
MULTI_CONFIG = {
    'vocab': 78,
    'dim': 128,                # token feature width
    'encoderLayers': 3,
    'heads': 2,
    'encoderKernel': 5,        # depthwise convolution kernel in the encoder's FFN
    'ffnMultiple': 2,
    'durationKernel': 3,
    'durationDropout': 0.2,    # training only
    'decoderDim': 128,
    'decoderLayers': 6,
    'decoderKernel': 9,        # depthwise convolution kernel in the decoder (10 ms frames)
    'subframes': 2,            # parameter frames (5 ms) per decoder frame (10 ms)
    'speakers': 328,           # the speaker table's rows (data/nn/speakers.json ids, then the style corpus's actors)
    'styleDim': 8,             # the style embedding's width
    'jitter': 0.005,
    'shimmer': 0.02,
    'positionFeatures': 5,
    'kind': 'multi',
    'speakerDim': 16,          # the speaker embedding's width
    'styles': 5                # the emotions with a style row (neutral is the zero vector) - STYLE_NAMES
}


# Token classes (vocab.json): 0 <pad>, 1 <unk>, 2 <bos>, 3 "_" (word boundary), 4-8 phrase ends, 9+ phones
PAD, UNK, BOS, WORD = 0, 1, 2, 3
PHRASE_ENDS = (4, 5, 6, 7, 8)
FIRST_PHONE = 9

# The parameters the network predicts (all but jitter and shimmer, which are constants)
FIXED_NAMES = ('jitter', 'shimmer')


# Whether tokens are boundary tokens - word boundaries and phrase ends (their durations are pauses, often zero)
def is_boundary(tokens):
    return (tokens >= WORD) & (tokens < FIRST_PHONE)


class LayerNorm(torch.nn.LayerNorm):
    """Layer norm over the channels of [B, T, C]"""


class DepthwiseConv(torch.nn.Module):
    """A depthwise 1-D convolution over time of [B, T, C] with 'same' padding"""

    def __init__(self, channels, kernel):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class ConvFFN(torch.nn.Module):
    """A pre-norm feed-forward block with a depthwise convolution - x + W2 SiLU(DW(SiLU(W1 LN(x))))"""

    def __init__(self, dim, hidden, kernel, dropout):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.inner = torch.nn.Linear(dim, hidden)
        self.conv = DepthwiseConv(hidden, kernel)
        self.outer = torch.nn.Linear(hidden, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        hidden = torch.nn.functional.silu(self.inner(self.norm(x))) * mask
        hidden = torch.nn.functional.silu(self.conv(hidden))
        return x + self.dropout(self.outer(hidden)) * mask


class SelfAttention(torch.nn.Module):
    """Pre-norm multi-head self-attention with a key padding mask"""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm = LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.out = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = query @ key.transpose(-1, -2) / math.sqrt(dim // self.heads)
        scores = scores.masked_fill(~mask[:, None, None, :, 0].bool(), -1e9)
        attended = (torch.softmax(scores, dim=-1) @ value).transpose(1, 2).reshape(batch, length, dim)
        return x + self.dropout(self.out(attended)) * mask


# The number of trainable parameters of a model
def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


#
# The multi-speaker, multi-style model
#

# The style rows of the multi-speaker model's style table, in order - neutral is the absence of style (the zero vector)
STYLE_NAMES = ('angry', 'disgusted', 'fearful', 'happy', 'sad')


class FiLMConvFFN(ConvFFN):
    """A conv-FFN block whose pre-norm output is modulated by the condition (FiLM) - x + W2 SiLU(DW(SiLU(W1 (LN(x) (1 +
    gamma) + beta)))), [gamma, beta] = Wc c (zero at initialization, so the block starts unconditioned)"""

    def __init__(self, dim, hidden, kernel, dropout, condition_dim):
        super().__init__(dim, hidden, kernel, dropout)
        self.film = torch.nn.Linear(condition_dim, 2 * dim)
        torch.nn.init.zeros_(self.film.weight)
        torch.nn.init.zeros_(self.film.bias)

    def forward(self, x, mask, condition=None):
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        hidden = self.norm(x) * (1 + gamma) + beta
        hidden = torch.nn.functional.silu(self.inner(hidden)) * mask
        hidden = torch.nn.functional.silu(self.conv(hidden))
        return x + self.dropout(self.outer(hidden)) * mask


class MultiModel(torch.nn.Module):
    """The multi-speaker, multi-style model - the tokens-to-durations-to-frames network, conditioned on a condition
    vector c = [speaker embedding (speakerDim), style embedding (styleDim)]:

    - the speaker embedding - a row of the speaker table, or any blend (a convex combination of rows) or raw vector
    - the style embedding - intensity x a row of the style table (STYLE_NAMES; neutral is zero), or any blend of rows

    c is added (projected) to the token embeddings, the duration predictor's input, and the decoder's input, and
    modulates every conv-FFN block (FiLM). The channel equalizer is a table per speaker (blended in dB at inference)."""

    def __init__(self, config=None, priors=None, dropout=0.1):
        super().__init__()
        self.config = {**MULTI_CONFIG, **(config or {})}
        cfg = self.config
        dim, decoder_dim = cfg['dim'], cfg['decoderDim']
        condition_dim = cfg['speakerDim'] + cfg['styleDim']
        self.embedding = torch.nn.Embedding(cfg['vocab'], dim, padding_idx=PAD)
        self.speaker = torch.nn.Embedding(cfg['speakers'], cfg['speakerDim'])
        torch.nn.init.normal_(self.speaker.weight, std=0.3)
        self.style = torch.nn.Parameter(torch.randn(cfg['styles'], cfg['styleDim']) * 0.3)
        self.condition_encoder = torch.nn.Linear(condition_dim, dim)
        layers = []
        for _ in range(cfg['encoderLayers']):
            layers.append(FiLMConvFFN(dim, dim * cfg['ffnMultiple'], cfg['encoderKernel'], dropout, condition_dim))
            layers.append(SelfAttention(dim, cfg['heads'], dropout))
        self.encoder = torch.nn.ModuleList(layers)
        self.encoder_norm = LayerNorm(dim)

        # The duration predictor - the condition added, two depthwise-separable convolution layers, then [log(d + 1),
        # pause logit, log(paused d + 1)]
        self.condition_duration = torch.nn.Linear(condition_dim, dim)
        self.duration_layers = torch.nn.ModuleList()
        for _ in range(2):
            self.duration_layers.append(torch.nn.ModuleDict({
                'conv': DepthwiseConv(dim, cfg['durationKernel']), 'linear': torch.nn.Linear(dim, dim), 'norm': LayerNorm(dim)
            }))
        self.duration_out = torch.nn.Linear(dim, 3)
        self.duration_dropout = torch.nn.Dropout(cfg['durationDropout'])

        # The decoder
        self.decoder_in = torch.nn.Linear(dim, decoder_dim)
        self.position_in = torch.nn.Linear(cfg['positionFeatures'], decoder_dim)
        self.condition_decoder = torch.nn.Linear(condition_dim, decoder_dim)
        self.decoder = torch.nn.ModuleList([
            FiLMConvFFN(decoder_dim, decoder_dim * cfg['ffnMultiple'], cfg['decoderKernel'], dropout, condition_dim)
            for _ in range(cfg['decoderLayers'])
        ])
        self.decoder_norm = LayerNorm(decoder_dim)
        self.head = torch.nn.Linear(decoder_dim, cfg['subframes'] * params.COUNT)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            prior = torch.tensor([float((priors or {}).get(name, params.DEFAULTS[name])) for name in params.NAMES])
            unit = params.normalize(prior[None])[0].clamp(0.02, 0.98)
            self.head.bias.copy_(torch.log(unit / (1 - unit)).repeat(cfg['subframes']))

        # The channel equalizer per speaker, and the constant jitter and shimmer
        self.eq_free = torch.nn.Parameter(torch.zeros(cfg['speakers'], params.EQ_BAND_COUNT))
        fixed = torch.zeros(params.COUNT)
        fixed_mask = torch.zeros(params.COUNT)
        for name in FIXED_NAMES:
            fixed[params.INDEX[name]] = cfg[name]
            fixed_mask[params.INDEX[name]] = 1
        self.register_buffer('fixed', fixed, persistent=False)
        self.register_buffer('fixed_mask', fixed_mask, persistent=False)

    def speaker_vector(self, speaker):
        """Speaker embeddings [B, speakerDim] of speaker ids [B], or of blends - weights [B, speakers] (rows summing to
        1)"""
        if speaker.dim() == 2 and speaker.is_floating_point():
            return speaker @ self.speaker.weight
        return self.speaker(speaker)

    def style_vector(self, emotion, intensity):
        """Style embeddings [B, styleDim] of emotion ids [B] (STYLE_NAMES indexes; -1 is neutral) and intensities [B]
        (0-1), or of blends - weights [B, styles] (intensity included)"""
        if emotion.dim() == 2:
            return emotion @ self.style
        rows = self.style[emotion.clamp(min=0)]
        return rows * ((emotion >= 0).float() * intensity)[:, None]

    def condition(self, speaker, emotion=None, intensity=None, speaker_vector=None):
        """The condition vectors [B, 1, C]"""
        speaker_vectors = self.speaker_vector(speaker) if speaker_vector is None else speaker_vector
        batch = speaker_vectors.shape[0]
        if emotion is None:
            style = speaker_vectors.new_zeros(batch, self.config['styleDim'])
        else:
            if intensity is None:
                intensity = torch.ones(batch, device=speaker_vectors.device)
            style = self.style_vector(emotion, intensity)
        return torch.cat([speaker_vectors, style], dim=-1)[:, None, :]

    def encode(self, tokens, condition):
        """Token features [B, N, D] and the token mask [B, N, 1]"""
        mask = (tokens != PAD).float()[..., None]
        x = (self.embedding(tokens) + self.condition_encoder(condition)) * mask
        for layer in self.encoder:
            x = layer(x, mask, condition) if isinstance(layer, FiLMConvFFN) else layer(x, mask)
        return self.encoder_norm(x) * mask, mask

    def durations(self, features, mask, condition):
        """The duration predictor's outputs [B, N, 3] - log(d + 1), the pause logit, log(paused d + 1)"""
        x = (features + self.condition_duration(condition)) * mask
        for layer in self.duration_layers:
            x = self.duration_dropout(layer['norm'](torch.nn.functional.silu(layer['linear'](layer['conv'](x * mask))))) * mask
        return self.duration_out(x)

    # Durations (frames, integer) from the duration predictor's outputs, for inference
    @staticmethod
    def predicted_durations(outputs, tokens, pause_threshold=0.5, rate=1.0):
        log_duration, pause_logit, log_paused = outputs.unbind(-1)
        phone = torch.expm1(log_duration).clamp(min=1.0) * rate
        paused = torch.expm1(log_paused).clamp(min=1.0) * rate
        boundary = is_boundary(tokens) | (tokens == BOS)
        pause = torch.sigmoid(pause_logit) > pause_threshold
        durations = torch.where(boundary, torch.where(pause, paused, torch.zeros_like(paused)), phone)
        durations = durations * (tokens != PAD)
        # Round the running sum, so rounding errors don't accumulate
        ends = torch.round(torch.cumsum(durations, dim=-1))
        starts = torch.cat([torch.zeros_like(ends[:, :1]), ends[:, :-1]], dim=-1)
        return (ends - starts).long()

    @staticmethod
    def regulate(features, durations, frame_count=None):
        """The length regulator - token features [B, N, D] repeated for their durations [B, N] (10 ms frames) to
        [B, F, D], with the position features [B, F, 5] and the frame mask [B, F, 1]; frame_count is F when the caller
        knows it (max(1, the longest total duration) - reading it from the device synchronizes)"""
        batch = features.shape[0]
        lengths = durations.sum(dim=1)
        frame_count = max(1, int(lengths.max())) if frame_count is None else frame_count
        ends = torch.cumsum(durations, dim=1)
        frames = torch.arange(frame_count, device=features.device)
        # Each frame's token - the first token whose end is after it
        token_index = torch.searchsorted(ends, frames[None].expand(batch, -1).contiguous(), right=True)
        token_index = token_index.clamp(max=features.shape[1] - 1)
        regulated = torch.gather(features, 1, token_index[..., None].expand(-1, -1, features.shape[2]))
        duration = torch.gather(durations, 1, token_index).float().clamp(min=1)
        start = torch.gather(ends - durations, 1, token_index).float()
        offset = frames[None].float() - start
        positions = torch.stack([
            (offset + 0.5) / duration, 1 - (offset + 0.5) / duration, torch.log1p(duration) / 3,
            torch.clamp(offset, max=20) / 20, torch.clamp(duration - 1 - offset, max=20) / 20
        ], dim=-1)
        mask = (frames[None] < lengths[:, None]).float()[..., None]
        return regulated * mask, positions * mask, mask

    def parameters_from_unit(self, unit):
        """Normalized frames [B, T, COUNT] to parameter values, with the fixed parameters (jitter, shimmer)"""
        values = params.denormalize(unit)
        return values * (1 - self.fixed_mask) + self.fixed * self.fixed_mask

    def decode(self, regulated, positions, mask, condition):
        """Decoder frames [B, F, D] (10 ms) to normalized parameter frames [B, F * subframes, COUNT]"""
        x = (self.decoder_in(regulated) + self.position_in(positions) + self.condition_decoder(condition)) * mask
        for layer in self.decoder:
            x = layer(x, mask, condition)
        hidden = self.decoder_norm(x)
        logits = self.head(hidden)
        batch, frames, _ = logits.shape
        return torch.sigmoid(logits.reshape(batch, frames * self.config['subframes'], params.COUNT))

    def eq(self, speaker):
        """The channel equalizer gains (dB) [B, EQ_BAND_COUNT] of speaker ids [B] or blend weights [B, speakers]"""
        table = params.EQ_RANGE_DB * torch.tanh(self.eq_free)
        if speaker.dim() == 2 and speaker.is_floating_point():
            return speaker @ table
        return table[speaker]

    def forward(self, tokens, durations, speaker=None, emotion=None, intensity=None, speaker_vector=None, frame_count=None):
        """Training forward pass (teacher-forced durations [B, N], 10 ms frames) - {duration (the predictor's outputs
        [B, N, 3]), unit (normalized parameter frames [B, 2F, COUNT]), frameMask [B, F, 1]}; frame_count is F if known
        (regulate)"""
        if speaker is None and speaker_vector is None:
            speaker = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        condition = self.condition(speaker, emotion, intensity, speaker_vector)
        features, mask = self.encode(tokens, condition)
        duration_outputs = self.durations(features, mask, condition)
        regulated, positions, frame_mask = self.regulate(features, durations, frame_count)
        unit = self.decode(regulated, positions, frame_mask, condition)
        return {'duration': duration_outputs, 'unit': unit, 'frameMask': frame_mask}

    @torch.no_grad()
    def infer(self, tokens, speaker=None, emotion=None, intensity=None, speaker_vector=None, pause_threshold=0.5, rate=1.0):
        """Inference - {durations [B, N] (10 ms frames), values (parameter frames [B, 2F + 1, COUNT]), frames}"""
        if speaker is None and speaker_vector is None:
            speaker = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        condition = self.condition(speaker, emotion, intensity, speaker_vector)
        features, mask = self.encode(tokens, condition)
        durations = self.predicted_durations(self.durations(features, mask, condition), tokens, pause_threshold, rate)
        regulated, positions, frame_mask = self.regulate(features, durations)
        values = self.parameters_from_unit(self.decode(regulated, positions, frame_mask, condition))
        values = torch.cat([values, values[:, -1:]], dim=1)
        return {'durations': durations, 'values': values, 'frames': durations.sum(dim=1) * self.config['subframes'] + 1}


# A model of a config - the multi-speaker model (config 'kind': 'multi'; the single-speaker model and its trainer,
# nn/train.py, are in git history)
def build_model(config, priors=None, dropout=0.1):
    if (config or {}).get('kind') != 'multi':
        raise ValueError('only the multi-speaker model (config kind "multi") is supported')
    return MultiModel(config, priors=priors, dropout=dropout)
