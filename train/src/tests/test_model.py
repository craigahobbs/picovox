# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import os
import tempfile
import unittest

import torch

from picovox_train import model, params

from .helpers import tiny_config


# A batch of two token sequences (the second padded) and their durations (10 ms frames)
def token_batch():
    tokens = torch.tensor([[model.BOS, 20, 30, model.WORD, 25, 5], [model.BOS, 40, 41, 5, model.PAD, model.PAD]])
    durations = torch.tensor([[1, 3, 4, 0, 2, 3], [0, 2, 2, 1, 0, 0]])
    return tokens, durations


class TestModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_is_boundary(self):
        tokens = torch.tensor([model.PAD, model.BOS, model.WORD, 4, 8, model.FIRST_PHONE])
        self.assertEqual(model.is_boundary(tokens).tolist(), [False, False, True, True, True, False])

    def test_layers(self):
        mask = torch.ones(2, 5, 1)
        x = torch.randn(2, 5, 8)
        self.assertEqual(model.DepthwiseConv(8, 3)(x).shape, (2, 5, 8))
        self.assertEqual(model.ConvFFN(8, 16, 3, 0.0)(x, mask).shape, (2, 5, 8))
        attention = model.SelfAttention(8, 2, 0.0)
        # A padded key changes nothing for the others
        padded = torch.cat([x, torch.randn(2, 1, 8)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(2, 1, 1)], dim=1)
        self.assertTrue(torch.allclose(attention(padded, padded_mask)[:, :5], attention(x, mask), atol=1e-6))

    def test_film(self):
        block = model.FiLMConvFFN(8, 16, 3, 0.0, 4)
        x = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5, 1)
        # Zero at initialization - unconditioned
        plain = model.ConvFFN(8, 16, 3, 0.0)
        plain.load_state_dict({key: value for key, value in block.state_dict().items() if not key.startswith('film')})
        self.assertTrue(torch.allclose(block(x, mask, torch.randn(1, 1, 4)), plain(x, mask)))

    def test_parameter_count(self):
        self.assertEqual(model.parameter_count(torch.nn.Linear(3, 2)), 8)

    def test_save_checkpoint(self):
        # Written beside its path, then moved there - replacing the last
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'last.pt')
            model.save_checkpoint({'step': 1}, path)
            model.save_checkpoint({'step': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False), {'step': 2})
            self.assertEqual(os.listdir(directory), ['last.pt'])

    def test_multi_model(self):
        network = model.MultiModel(tiny_config(), priors={'f0': 150})
        self.assertEqual(network.config['dim'], 16)
        self.assertEqual(network.config['kind'], 'multi')
        # The head starts at the priors
        frames = network.parameters_from_unit(torch.sigmoid(network.head.bias[:params.COUNT])[None, None]).detach()
        self.assertAlmostEqual(float(frames[0, 0, params.INDEX['f0']]), 150, delta=1)
        self.assertAlmostEqual(float(frames[0, 0, params.INDEX['jitter']]), 0.005, places=6)
        self.assertAlmostEqual(float(frames[0, 0, params.INDEX['shimmer']]), 0.02, places=6)
        # The default config
        self.assertEqual(model.MultiModel().config['dim'], 128)

    def test_forward(self):
        network = model.MultiModel(tiny_config()).eval()
        tokens, durations = token_batch()
        outputs = network(tokens, durations)
        self.assertEqual(outputs['duration'].shape, (2, 6, 3))
        self.assertEqual(outputs['unit'].shape, (2, 26, params.COUNT))
        self.assertEqual(outputs['frameMask'].shape, (2, 13, 1))
        self.assertEqual(outputs['frameMask'][1, :, 0].sum().item(), 5)
        self.assertTrue(((outputs['unit'] > 0) & (outputs['unit'] < 1)).all())
        # A known frame count, the speaker and style, and a speaker vector
        speaker = torch.tensor([1, 2])
        styled = network(tokens, durations, speaker=speaker, emotion=torch.tensor([0, -1]), intensity=torch.tensor([1.0, 0.5]),
                         frame_count=16)
        self.assertEqual(styled['unit'].shape, (2, 32, params.COUNT))
        vector = network(tokens, durations, speaker_vector=network.speaker_vector(speaker), emotion=torch.tensor([0, -1]),
                         intensity=torch.tensor([1.0, 0.5]), frame_count=16)
        self.assertTrue(torch.allclose(vector['unit'], styled['unit']))
        # Speaker 0 by default
        self.assertTrue(torch.allclose(network(tokens, durations, speaker=torch.tensor([0, 0]))['unit'], outputs['unit']))

    def test_condition(self):
        network = model.MultiModel(tiny_config())
        speaker = torch.tensor([1, 3])
        # Neutral (-1) is the zero style; no emotion is the zero style; intensity defaults to 1
        neutral = network.condition(speaker, torch.tensor([-1, -1]), torch.tensor([1.0, 1.0]))
        self.assertEqual(neutral.shape, (2, 1, 8))
        self.assertTrue(torch.equal(neutral, network.condition(speaker)))
        full = network.condition(speaker, torch.tensor([2, 2]))
        self.assertTrue(torch.allclose(full[:, 0, 4:], network.style[2].expand(2, -1)))
        half = network.condition(speaker, torch.tensor([2, 2]), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.allclose(half[:, 0, 4:], 0.5 * network.style[2].expand(2, -1)))
        # Blends - speaker weights and style weights
        weights = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
        self.assertTrue(torch.allclose(network.speaker_vector(weights), (network.speaker.weight[1] + network.speaker.weight[3])[None] / 2))
        style_weights = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.25]])
        self.assertTrue(torch.allclose(network.style_vector(style_weights, None), (network.style[1] + 0.25 * network.style[4])[None]))
        # The equalizer of ids and blends
        with torch.no_grad():
            network.eq_free.normal_()
        table = params.EQ_RANGE_DB * torch.tanh(network.eq_free)
        self.assertTrue(torch.allclose(network.eq(speaker), table[speaker]))
        self.assertTrue(torch.allclose(network.eq(weights), (table[1] + table[3])[None] / 2))

    def test_predicted_durations(self):
        tokens = torch.tensor([[model.BOS, 20, model.WORD, 21, 5, model.PAD]])
        outputs = torch.zeros(1, 6, 3)
        outputs[..., 0] = torch.log1p(torch.tensor([2.0, 2.4, 9.0, 0.2, 3.0, 7.0]))
        outputs[..., 1] = torch.tensor([5.0, 0.0, -5.0, 0.0, 5.0, 0.0])
        outputs[..., 2] = torch.log1p(torch.tensor([4.0, 0.0, 6.0, 0.0, 2.6, 0.0]))
        durations = model.MultiModel.predicted_durations(outputs, tokens)
        # <bos> paused (4), the phone 2.4, the word boundary unpaused, a phone clamped to 1, the period paused (2.6), padding 0
        self.assertEqual(durations.tolist(), [[4, 2, 0, 1, 3, 0]])
        self.assertEqual(durations.dtype, torch.long)
        slower = model.MultiModel.predicted_durations(outputs, tokens, pause_threshold=0.999, rate=2.0)
        self.assertEqual(slower.tolist(), [[0, 5, 0, 2, 0, 0]])

    def test_regulate(self):
        features = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        durations = torch.tensor([[2, 0, 3]])
        regulated, positions, mask = model.MultiModel.regulate(features, durations)
        self.assertEqual(regulated[0, :, 0].tolist(), [0.0, 0.0, 4.0, 4.0, 4.0])
        self.assertEqual(positions.shape, (1, 5, 5))
        self.assertAlmostEqual(float(positions[0, 0, 0]), 0.25)
        self.assertAlmostEqual(float(positions[0, 4, 1]), 1 - 2.5 / 3, places=6)
        self.assertEqual(mask[0, :, 0].tolist(), [1.0] * 5)
        # A longer frame count is masked past the durations
        regulated, _, mask = model.MultiModel.regulate(features, durations, frame_count=7)
        self.assertEqual(mask[0, :, 0].tolist(), [1.0] * 5 + [0.0] * 2)
        self.assertEqual(regulated[0, 6].tolist(), [0.0, 0.0])
        # All-zero durations give one frame
        _, _, mask = model.MultiModel.regulate(features, torch.zeros(1, 3, dtype=torch.long))
        self.assertEqual(mask.shape, (1, 1, 1))

    def test_infer(self):
        network = model.MultiModel(tiny_config()).eval()
        tokens, _ = token_batch()
        result = network.infer(tokens)
        self.assertEqual(result['durations'].shape, (2, 6))
        frame_count = result['values'].shape[1]
        self.assertEqual(frame_count, 2 * int(result['durations'].sum(dim=1).max().clamp(min=1)) + 1)
        self.assertEqual(result['frames'].tolist(), (result['durations'].sum(dim=1) * 2 + 1).tolist())
        self.assertTrue(torch.equal(result['values'][:, -1], result['values'][:, -2]))
        self.assertFalse(result['values'].requires_grad)
        styled = network.infer(tokens, speaker=torch.tensor([1, 1]), emotion=torch.tensor([3, 3]), intensity=torch.tensor([1.0, 1.0]))
        self.assertEqual(styled['values'].shape[-1], params.COUNT)
        vector = network.infer(tokens, speaker_vector=network.speaker_vector(torch.tensor([1, 1])), emotion=torch.tensor([3, 3]),
                               intensity=torch.tensor([1.0, 1.0]))
        self.assertTrue(torch.allclose(vector['values'], styled['values']))

    def test_build_model(self):
        self.assertIsInstance(model.build_model(tiny_config(), dropout=0.0), model.MultiModel)
        with self.assertRaises(ValueError):
            model.build_model({'kind': 'single'})
        with self.assertRaises(ValueError):
            model.build_model(None)
