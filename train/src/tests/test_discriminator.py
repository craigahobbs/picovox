# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# pylint: disable=missing-class-docstring, missing-function-docstring, missing-module-docstring

import unittest
import unittest.mock

import torch

from picovox_train import discriminator


class TestDiscriminator(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_strided_conv_function(self):
        # The polyphase input gradient and the weight and bias gradients equal torch's convolution's
        for width in (16, 17):
            x = torch.randn(2, 3, 5, width, dtype=torch.float64, requires_grad=True)
            weight = torch.randn(3, 3, 3, 9, dtype=torch.float64, requires_grad=True)
            bias = torch.randn(3, dtype=torch.float64, requires_grad=True)
            output = discriminator._StridedConvFunction.apply(x, weight, bias)  # pylint: disable=protected-access
            expected = torch.nn.functional.conv2d(x, weight, bias, stride=(1, 2), padding=(1, 4))
            self.assertTrue(torch.allclose(output, expected))
            grad = torch.randn_like(output)
            gradients = torch.autograd.grad(output, (x, weight, bias), grad)
            expected_gradients = torch.autograd.grad(expected, (x, weight, bias), grad)
            for value, expected_value in zip(gradients, expected_gradients):
                self.assertTrue(torch.allclose(value, expected_value))
        # Only the input's gradient, or only the weights'
        x = torch.randn(1, 2, 4, 12, requires_grad=True)
        weight = torch.randn(2, 2, 3, 9)
        bias = torch.randn(2)
        self.assertEqual(torch.autograd.grad(discriminator._StridedConvFunction.apply(x, weight, bias).sum(), x)[0].shape,  # pylint: disable=protected-access
                         x.shape)
        weight.requires_grad_(True)
        x = x.detach()
        self.assertEqual(torch.autograd.grad(discriminator._StridedConvFunction.apply(x, weight, bias).sum(), weight)[0].shape,  # pylint: disable=protected-access
                         weight.shape)

    def test_strided_conv_function_autocast(self):
        x = torch.randn(1, 2, 4, 12)
        weight = torch.randn(2, 2, 3, 9)
        bias = torch.randn(2)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = discriminator._StridedConvFunction.apply(x, weight, bias)  # pylint: disable=protected-access
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_strided_conv2d(self):
        conv = discriminator.StridedConv2d(4)
        x = torch.randn(1, 4, 5, 20)
        self.assertTrue(torch.allclose(conv(x), torch.nn.functional.conv2d(x, conv.weight, conv.bias, stride=(1, 2), padding=(1, 4))))
        # On MPS, the faster input gradient
        mps_input = unittest.mock.Mock()
        mps_input.device.type = 'mps'
        with unittest.mock.patch.object(discriminator._StridedConvFunction, 'apply', return_value='applied') as apply:  # pylint: disable=protected-access
            self.assertEqual(conv(mps_input), 'applied')
        apply.assert_called_once_with(mps_input, conv.weight, conv.bias)

    def test_spectrogram_discriminator(self):
        model = discriminator.SpectrogramDiscriminator(16000, 256, 32, 7000.0, channels=4)
        audio = torch.randn(2, 4000)
        spectrogram = model.spectrogram(audio)
        self.assertEqual(spectrogram.shape, (2, 1, 63, 32))
        scores, features = model(audio)
        self.assertEqual(len(features), 5)
        self.assertEqual(scores.shape, (2, 1, 63, 4))
        self.assertEqual(features[1].shape, (2, 4, 63, 16))

    def test_critic(self):
        critic = discriminator.Critic(16000, 7000.0, channels=4)
        audio = torch.randn(2, 4000)
        outputs = critic(audio)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0][0].dtype, torch.float32)
        # Autocast precision - the scores float32, the features in the precision
        low = discriminator.Critic(16000, 7000.0, dtype=torch.bfloat16, channels=4)
        low.load_state_dict(critic.state_dict())
        low_outputs = low(audio)
        self.assertEqual(low_outputs[0][0].dtype, torch.float32)
        self.assertEqual(low_outputs[0][1][0].dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(low_outputs[0][0], outputs[0][0], atol=0.1))

    def test_losses(self):
        real = [(torch.ones(2, 3), [torch.ones(2, 2)]), (torch.ones(1, 4), [torch.zeros(3)])]
        fake = [(torch.zeros(2, 3), [torch.ones(2, 2)]), (torch.zeros(1, 4), [torch.ones(3)])]
        self.assertEqual(float(discriminator.discriminator_loss(real, fake)), 0.0)
        self.assertEqual(float(discriminator.discriminator_loss(fake, real)), 2.0)
        adversarial, matching = discriminator.generator_losses(real, fake)
        self.assertEqual(float(adversarial), 1.0)
        self.assertEqual(float(matching), 0.5)
