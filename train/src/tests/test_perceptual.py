# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import types
import unittest
import unittest.mock

import torch

from picovox_train import perceptual


class FakeEncoder(torch.nn.Module):
    """A fake wav2vec2 - each 20 ms frame through a linear layer, then its transformer's layers (linear too), each
    layer's output a hidden state (the first, the projected frames)"""

    def __init__(self, layers=24):
        super().__init__()
        torch.manual_seed(1)
        self.project = torch.nn.Linear(320, 8)
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(layers)])
        self.inputs = []

    def forward(self, audio, output_hidden_states):
        assert output_hidden_states
        self.inputs.append(audio.detach().clone())
        hidden = [torch.tanh(self.project(audio[:, :audio.shape[1] // 320 * 320].reshape(audio.shape[0], -1, 320)))]
        for layer in self.encoder.layers:
            hidden.append(torch.tanh(layer(hidden[-1])))
        return types.SimpleNamespace(hidden_states=tuple(hidden))


def fake_loss(layers=perceptual.LAYERS, dtype=None):
    encoder = FakeEncoder()
    with unittest.mock.patch('picovox_train.ctc.load_ctc', return_value=(types.SimpleNamespace(wav2vec2=encoder), {}, 'cpu')):
        return perceptual.PerceptualLoss('cpu', layers, dtype), encoder


class TestPerceptual(unittest.TestCase):

    def test_frozen_and_cut(self):
        # The recognizer frozen, in eval mode, and its transformer cut after the deepest layer used
        _, encoder = fake_loss((2, 5))
        self.assertEqual(len(encoder.encoder.layers), 5)
        self.assertFalse(encoder.training)
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))

    def test_loss(self):
        # A crop against itself - 0; against another - positive, and its gradient reaches the rendered crop (not the target)
        loss, encoder = fake_loss()
        generator = torch.Generator().manual_seed(2)
        target = torch.randn(2, 24000, generator=generator)
        self.assertAlmostEqual(float(loss(target.clone(), target)), 0.0, places=5)
        audio = torch.randn(2, 24000, generator=generator, requires_grad=True)
        value = loss(audio, target)
        self.assertGreater(float(value.detach()), 0.01)
        value.backward()
        self.assertGreater(float(audio.grad.abs().sum()), 0.0)

        # (resampled to 16 kHz and normalized - zero mean, unit variance)
        rendered = encoder.inputs[-1]
        self.assertEqual(rendered.shape, (2, 16000))
        self.assertTrue(torch.allclose(rendered.mean(1), torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.allclose(rendered.std(1), torch.ones(2), atol=1e-3))

    def test_bfloat16(self):
        # The recognizer under autocast in bfloat16 - the features compared in float32, the loss near the float32 loss's
        loss, _ = fake_loss()
        half, _ = fake_loss(dtype=torch.bfloat16)
        generator = torch.Generator().manual_seed(3)
        target = torch.randn(2, 24000, generator=generator)
        audio = torch.randn(2, 24000, generator=generator, requires_grad=True)
        features = half.features(target)
        self.assertEqual({feature.dtype for feature in features}, {torch.float32})
        value = half(audio, target)
        self.assertEqual(value.dtype, torch.float32)
        self.assertAlmostEqual(float(value.detach()), float(loss(audio, target).detach()), places=2)
        value.backward()
        self.assertGreater(float(audio.grad.abs().sum()), 0.0)
