# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import json
import os
import tempfile
import unittest

import torch

from picovox_train import wav2vec2


# A tiny recognizer configuration - the real one's layout (seven convolutions, the stable layer norm transformer) at a
# fraction of the width
CONFIG = {
    'conv_dim': [8, 8, 8, 8, 8, 8, 8], 'conv_kernel': [10, 3, 3, 3, 3, 2, 2], 'conv_stride': [5, 2, 2, 2, 2, 2, 2],
    'hidden_size': 16, 'num_hidden_layers': 3, 'num_attention_heads': 2, 'intermediate_size': 32,
    'num_conv_pos_embeddings': 8, 'num_conv_pos_embedding_groups': 4, 'layer_norm_eps': 1e-5, 'vocab_size': 7
}


class TestWav2Vec2(unittest.TestCase):

    def test_output_lengths(self):
        model = wav2vec2.Wav2Vec2ForCTC(CONFIG)
        # (a 20 ms frame of 16 kHz samples - 320 samples a frame, the convolutions' receptive field 400)
        self.assertEqual(model.output_lengths(torch.tensor([16000, 400, 399, 16320])).tolist(), [49, 1, 0, 50])
        self.assertEqual(model.output_lengths(16000), 49)
        torch.manual_seed(1)
        self.assertEqual(model(torch.randn(2, 16000)).logits.shape, (2, 49, CONFIG['vocab_size']))

    def test_hidden_states(self):
        # The transformer's input and each layer's output - num_hidden_layers + 1 states - and the last one layer-normed
        torch.manual_seed(2)
        model = wav2vec2.Wav2Vec2(CONFIG).eval()
        outputs = model(torch.randn(1, 8000), output_hidden_states=True)
        self.assertEqual(len(outputs.hidden_states), CONFIG['num_hidden_layers'] + 1)
        self.assertEqual(outputs.hidden_states[0].shape, (1, 24, CONFIG['hidden_size']))
        self.assertTrue(torch.equal(outputs.last_hidden_state, model.encoder.layer_norm(outputs.hidden_states[-1])))
        self.assertTrue(torch.equal(outputs.hidden_states[-1], model.encoder.layers[-1](outputs.hidden_states[-2], None)))
        self.assertIsNone(model(torch.randn(1, 8000)).hidden_states)

    def test_attention_mask(self):
        # A clip padded in a batch gets the features it gets alone (the padded frames zero), while without the mask the
        # padding reaches it
        torch.manual_seed(3)
        model = wav2vec2.Wav2Vec2ForCTC(CONFIG).eval()
        clip = torch.randn(1, 6400)
        padded = torch.cat([clip, torch.zeros(1, 3200)], dim=1)
        mask = torch.cat([torch.ones(1, 6400, dtype=torch.long), torch.zeros(1, 3200, dtype=torch.long)], dim=1)
        with torch.no_grad():
            alone = model.wav2vec2(clip, output_hidden_states=True)
            masked = model.wav2vec2(padded, attention_mask=mask, output_hidden_states=True)
            unmasked = model.wav2vec2(padded, output_hidden_states=True)
        frames = alone.last_hidden_state.shape[1]
        for own, batched in zip(alone.hidden_states, masked.hidden_states):
            self.assertTrue(torch.allclose(own, batched[:, :frames], atol=1e-5))
        # (the padded frames' input is zero before the positions are added, and the final layer norm sees their stream)
        self.assertTrue(torch.equal(masked.hidden_states[0][:, frames:], model.wav2vec2.encoder.pos_conv_embed(
            torch.cat([model.wav2vec2.feature_projection(model.wav2vec2.feature_extractor(padded).transpose(1, 2))[:, :frames],
                       torch.zeros(1, masked.hidden_states[0].shape[1] - frames, CONFIG['hidden_size'])], dim=1))[:, frames:]))
        self.assertFalse(torch.allclose(alone.last_hidden_state, unmasked.last_hidden_state[:, :frames], atol=1e-3))
        self.assertEqual(model(padded, attention_mask=mask).logits.shape, (1, masked.last_hidden_state.shape[1], CONFIG['vocab_size']))

    def test_load(self):
        # A checkpoint as the Hub's names it (the positional convolution's weight_g and weight_v, the pre-training's
        # masked_spec_embed) loads, and the loaded model is the saved one
        torch.manual_seed(4)
        model = wav2vec2.Wav2Vec2ForCTC(CONFIG)
        state = model.state_dict()
        saved = {}
        for name, tensor in state.items():
            name = name.replace('parametrizations.weight.original0', 'weight_g').replace('parametrizations.weight.original1', 'weight_v')
            saved[name] = tensor
        saved['wav2vec2.masked_spec_embed'] = torch.zeros(CONFIG['hidden_size'])
        self.assertIn('wav2vec2.encoder.pos_conv_embed.conv.weight_g', saved)
        with tempfile.TemporaryDirectory() as directory:
            torch.save(saved, os.path.join(directory, 'pytorch_model.bin'))
            with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as file:
                json.dump({**CONFIG, 'architectures': ['Wav2Vec2ForCTC']}, file)
            with open(os.path.join(directory, 'vocab.json'), 'w', encoding='utf-8') as file:
                json.dump({'<pad>': 0, 'a': 1}, file)
            loaded = wav2vec2.load(directory)
            self.assertEqual(wav2vec2.vocabulary(directory), {'<pad>': 0, 'a': 1})
        self.assertFalse(loaded.training)
        self.assertEqual(set(loaded.state_dict()), set(state))
        self.assertTrue(all(torch.equal(tensor, state[name]) for name, tensor in loaded.state_dict().items()))
        samples = torch.randn(1, 4000)
        with torch.no_grad():
            self.assertTrue(torch.equal(loaded(samples).logits, model.eval()(samples).logits))
