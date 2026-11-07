# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The wav2vec2 phoneme recognizer's network (ctc.py's facebook/wav2vec2-lv-60-espeak-cv-ft - wav2vec2 large, the
"stable layer norm" variant, with a CTC head), in plain PyTorch, loaded from the pinned checkpoint (corpora.py -
downloads/wav2vec2/): the feature encoder (seven strided convolutions, each layer-normed and GELU'd - a 20 ms frame
of 16 kHz samples), the feature projection, the positional convolution (weight-normed, grouped), 24 pre-norm
transformer layers, the final layer norm, and the CTC head. The modules and their parameters are named as the
checkpoint names them, so it loads as is.

    model = load(corpora.WAV2VEC2_DIR, device)           # a Wav2Vec2ForCTC, in eval mode
    model(samples, attention_mask).logits                # [batch, frames, vocab] - the CTC posteriors (ctc.py)
    model.wav2vec2(samples, output_hidden_states=True).hidden_states   # 25 [batch, frames, 1024] - the features
    model.output_lengths(sample_counts)                  # each clip's frames (perceptual.py, quality.py)

hidden_states[0] is the transformer's input (the projected features with their positions) and hidden_states[k] (1 to
24) layer k's output - last_hidden_state is the final layer norm of the last (as transformers' Wav2Vec2Model gives them,
bit for bit); samples are normalized (zero mean, unit
variance) by the caller, as the recognizer expects. A batch's attention_mask (1 for a clip's samples, 0 for the
padding) zeroes the padded frames and keeps the attention from them, so a clip's features are the same alone or padded
in a batch.
"""

import json
import os
import types

import torch

from . import read_json


class ConvLayer(torch.nn.Module):
    """A feature encoder layer - a strided convolution, a layer norm over its channels, GELU"""

    def __init__(self, in_channels, out_channels, kernel, stride):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel, stride=stride)
        self.layer_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, hidden):
        hidden = self.conv(hidden)
        hidden = self.layer_norm(hidden.transpose(-2, -1)).transpose(-2, -1)
        return torch.nn.functional.gelu(hidden)


class FeatureEncoder(torch.nn.Module):
    """The feature encoder - the convolution layers over the samples [batch, samples] to features [batch, channels, frames]"""

    def __init__(self, config):
        super().__init__()
        channels = [1] + list(config['conv_dim'])
        self.conv_layers = torch.nn.ModuleList([
            ConvLayer(channels[index], channels[index + 1], kernel, stride)
            for index, (kernel, stride) in enumerate(zip(config['conv_kernel'], config['conv_stride']))
        ])

    def forward(self, samples):
        hidden = samples[:, None]
        for layer in self.conv_layers:
            hidden = layer(hidden)
        return hidden


class FeatureProjection(torch.nn.Module):
    """The features' layer norm and projection to the transformer's width"""

    def __init__(self, config):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(config['conv_dim'][-1], eps=config['layer_norm_eps'])
        self.projection = torch.nn.Linear(config['conv_dim'][-1], config['hidden_size'])

    def forward(self, features):
        return self.projection(self.layer_norm(features))


class PositionalConvEmbedding(torch.nn.Module):
    """The positional convolution - a wide grouped convolution over the frames (weight-normed), its last output
    dropped (an even kernel's 'same' padding), GELU"""

    def __init__(self, config):
        super().__init__()
        kernel = config['num_conv_pos_embeddings']
        conv = torch.nn.Conv1d(config['hidden_size'], config['hidden_size'], kernel, padding=kernel // 2,
                               groups=config['num_conv_pos_embedding_groups'])
        self.conv = torch.nn.utils.parametrizations.weight_norm(conv, name='weight', dim=2)
        self.remove = 1 if kernel % 2 == 0 else 0

    def forward(self, hidden):
        hidden = self.conv(hidden.transpose(1, 2))
        hidden = hidden[:, :, :hidden.shape[-1] - self.remove]
        return torch.nn.functional.gelu(hidden).transpose(1, 2)


class Attention(torch.nn.Module):
    """Multi-head self-attention - the queries, keys, and values projected, scaled dot-product attention (keys under
    the mask left out), the output projected"""

    def __init__(self, config):
        super().__init__()
        self.heads = config['num_attention_heads']
        width = config['hidden_size']
        self.k_proj = torch.nn.Linear(width, width)
        self.v_proj = torch.nn.Linear(width, width)
        self.q_proj = torch.nn.Linear(width, width)
        self.out_proj = torch.nn.Linear(width, width)

    def forward(self, hidden, mask):
        batch, frames, width = hidden.shape
        shape = (batch, frames, self.heads, width // self.heads)
        query = self.q_proj(hidden).view(shape).transpose(1, 2)
        key = self.k_proj(hidden).view(shape).transpose(1, 2)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, frames, width))


class FeedForward(torch.nn.Module):
    """The transformer layer's feed-forward - a GELU between two linear layers"""

    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = torch.nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = torch.nn.Linear(config['intermediate_size'], config['hidden_size'])

    def forward(self, hidden):
        return self.output_dense(torch.nn.functional.gelu(self.intermediate_dense(hidden)))


class EncoderLayer(torch.nn.Module):
    """A pre-norm transformer layer - x + attention(norm(x)), then + feed_forward(norm(...))"""

    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.layer_norm = torch.nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = torch.nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])

    def forward(self, hidden, mask):
        hidden = hidden + self.attention(self.layer_norm(hidden), mask)
        return hidden + self.feed_forward(self.final_layer_norm(hidden))


class Encoder(torch.nn.Module):
    """The transformer - the positions added, the layers, the final layer norm"""

    def __init__(self, config):
        super().__init__()
        self.pos_conv_embed = PositionalConvEmbedding(config)
        self.layer_norm = torch.nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.layers = torch.nn.ModuleList([EncoderLayer(config) for _ in range(config['num_hidden_layers'])])

    # The layers' hidden states - [the input, each layer's output] - and the last one's layer norm; frame_mask
    # [batch, frames] is True at a clip's frames (None - every frame)
    def forward(self, hidden, frame_mask=None):
        mask = None
        if frame_mask is not None:
            hidden = hidden * frame_mask[..., None]
            mask = frame_mask[:, None, None, :]
        hidden = hidden + self.pos_conv_embed(hidden)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden, mask)
            states.append(hidden)
        return states, self.layer_norm(hidden)


class Wav2Vec2(torch.nn.Module):
    """The wav2vec2 model - samples to features"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureEncoder(config)
        self.feature_projection = FeatureProjection(config)
        self.encoder = Encoder(config)

    # Each clip's frames for its samples (an int tensor, or an int) - the convolutions' output lengths
    def output_lengths(self, lengths):
        for kernel, stride in zip(self.config['conv_kernel'], self.config['conv_stride']):
            lengths = torch.div(lengths - kernel, stride, rounding_mode='floor') + 1
        return lengths

    def forward(self, samples, attention_mask=None, output_hidden_states=False):
        """samples [batch, samples], attention_mask [batch, samples] (1 at a clip's samples, 0 at the padding) -
        {last_hidden_state [batch, frames, width], hidden_states (with output_hidden_states - the encoder's)}"""
        features = self.feature_extractor(samples).transpose(1, 2)
        frame_mask = None
        if attention_mask is not None:
            lengths = self.output_lengths(attention_mask.sum(-1))
            frame_mask = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        states, last = self.encoder(self.feature_projection(features), frame_mask)
        return types.SimpleNamespace(last_hidden_state=last, hidden_states=tuple(states) if output_hidden_states else None)


class Wav2Vec2ForCTC(torch.nn.Module):
    """The recognizer - the wav2vec2 model and its CTC head"""

    def __init__(self, config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2(config)
        self.lm_head = torch.nn.Linear(config['hidden_size'], config['vocab_size'])

    def output_lengths(self, lengths):
        return self.wav2vec2.output_lengths(lengths)

    def forward(self, samples, attention_mask=None):
        return types.SimpleNamespace(logits=self.lm_head(self.wav2vec2(samples, attention_mask).last_hidden_state))


# The checkpoint's parameter names as this model names them - the positional convolution's weight norm (weight_g and
# weight_v - torch's parametrization names them original0 and original1); the pre-training's masked_spec_embed is left out
def state_dict_names(state):
    renamed = {}
    for name, tensor in state.items():
        if name == 'wav2vec2.masked_spec_embed':
            continue
        name = name.replace('pos_conv_embed.conv.weight_g', 'pos_conv_embed.conv.parametrizations.weight.original0')
        name = name.replace('pos_conv_embed.conv.weight_v', 'pos_conv_embed.conv.parametrizations.weight.original1')
        renamed[name] = tensor
    return renamed


# Load the recognizer from its directory (config.json and pytorch_model.bin - corpora.py's pinned files) onto a device,
# in eval mode
def load(directory, device='cpu'):
    config = read_json(os.path.join(directory, 'config.json'))
    model = Wav2Vec2ForCTC(config)
    state = torch.load(os.path.join(directory, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict_names(state))
    return model.to(device).eval()


# The recognizer's vocabulary - {token: id} (vocab.json)
def vocabulary(directory):
    with open(os.path.join(directory, 'vocab.json'), encoding='utf-8') as file:
        return json.load(file)
