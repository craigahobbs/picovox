# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The adversarial critic for training (trainmulti.py --adversarial) - spectrogram discriminators that judge the
synthesizer's renderings of the network's parameters against the recordings. A deterministic network trained by
spectral regression over-smooths: where a formant's position is uncertain, a broad peak minimizes the expected error,
so the predicted bandwidths come out wide and the voice muffled. The critic penalizes what no recording looks like.
Its gradient reaches only the interpretable parameters (through dsynth.py), so they stay interpretable - the critic
is used in training only, never exported.

Each discriminator is a small 2-D convolution stack over a log-mel spectrogram (time x mel) at one resolution; the
losses are least-squares GAN losses and feature matching (the L1 distance of the discriminators' intermediate
activations, real against rendered).
"""

import torch

from . import spectral


# (an autograd Function needs only forward and backward, not the other abstract methods)
class _StridedConvFunction(torch.autograd.Function):  # pylint: disable=abstract-method
    """A (3, 9) convolution of stride (1, 2) and padding (1, 4) whose input gradient is two stride-1 convolutions of the
    output gradient - the kernel's even and odd columns (its polyphase components), flipped - interleaved: on MPS the
    strided convolution's own input gradient takes about 2.7 times its forward. The forward is the convolution itself
    (in the autocast precision, as autocast runs it)."""

    @staticmethod
    def forward(ctx, x, weight, bias):  # pylint: disable=arguments-differ
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        x, weight, bias = x.to(dtype), weight.to(dtype), bias.to(dtype)
        ctx.save_for_backward(x, weight)
        with torch.autocast(x.device.type, enabled=False):
            return torch.nn.functional.conv2d(x, weight, bias, stride=(1, 2), padding=(1, 4))

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            # grad_x[..., 2p + r] = the output gradient correlated with the kernel's columns r, r + 2, ... (time and
            # columns flipped, input and output channels swapped)
            flipped = weight.flip(2).transpose(0, 1)
            even = torch.nn.functional.conv2d(grad, flipped[..., 0::2].flip(-1), padding=(1, 2))
            odd = torch.nn.functional.conv2d(torch.nn.functional.pad(grad, (1, 2)), flipped[..., 1::2].flip(-1), padding=(1, 0))
            grad_x = grad.new_empty(grad.shape[0], weight.shape[1], grad.shape[2], x.shape[3])
            grad_x[..., 0::2] = even[..., :(x.shape[3] + 1) // 2]
            grad_x[..., 1::2] = odd[..., :x.shape[3] // 2]
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            _, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
                grad, x, weight, [weight.shape[0]], (1, 2), (1, 4), (1, 1), False, (0, 0), 1,
                (False, ctx.needs_input_grad[1], ctx.needs_input_grad[2]))
        return grad_x, grad_weight, grad_bias


class StridedConv2d(torch.nn.Conv2d):
    """The discriminators' (3, 9) convolution of stride (1, 2) - on MPS with the faster input gradient
    (_StridedConvFunction); elsewhere torch's"""

    def __init__(self, channels):
        super().__init__(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))

    def forward(self, input):
        if input.device.type == 'mps':
            return _StridedConvFunction.apply(input, self.weight, self.bias)
        return super().forward(input)


class SpectrogramDiscriminator(torch.nn.Module):
    """A log-mel spectrogram discriminator at one resolution"""

    def __init__(self, sample_rate, fft_size, mels, max_hz, channels=32, eps=0.0):
        super().__init__()
        self.fft_size = fft_size
        self.eps = eps
        bank = spectral.mel_filterbank(sample_rate, fft_size, mels, high=max_hz)
        self.register_buffer('bank', torch.tensor(bank, dtype=torch.float32), persistent=False)
        leaky = torch.nn.LeakyReLU(0.2)
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(1, channels, (3, 9), padding=(1, 4)), leaky),
            torch.nn.Sequential(StridedConv2d(channels), leaky),
            torch.nn.Sequential(StridedConv2d(channels), leaky),
            torch.nn.Sequential(StridedConv2d(channels), leaky),
            torch.nn.Sequential(torch.nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)), leaky)
        ])
        self.out = torch.nn.Conv2d(channels, 1, (3, 3), padding=(1, 1))

    def spectrogram(self, audio):
        """Audio [B, N] to the discriminator's input, its log-mel spectrogram [B, 1, T, M] - it doesn't depend on the
        discriminator's weights, so a training step computes it once per signal for all the critic's passes"""
        magnitude = spectral.stft_magnitude(audio, self.fft_size, self.fft_size // 4, eps=self.eps)
        return torch.log(magnitude ** 2 @ self.bank.T + 1e-5)[:, None]

    def score(self, spectrogram):
        """A log-mel spectrogram [B, 1, T, M] to (scores [B, 1, T, M'], intermediate features)"""
        x = spectrogram
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return self.out(x), features

    def forward(self, audio):
        """Audio [B, N] to (scores [B, 1, T, M'], intermediate features)"""
        return self.score(self.spectrogram(audio))


class Critic(torch.nn.Module):
    """Spectrogram discriminators at several resolutions; dtype (torch.bfloat16 or torch.float16) runs their
    convolutions in that precision (autocast - the spectrograms, the weights, and the scores stay float32; the
    intermediate features stay in that precision until generator_losses uses them); channels is each convolution's width"""

    def __init__(self, sample_rate, max_hz, eps=0.0, dtype=None, channels=32):
        super().__init__()
        self.dtype = dtype
        self.discriminators = torch.nn.ModuleList([
            SpectrogramDiscriminator(sample_rate, 1024, 80, max_hz, channels=channels, eps=eps),
            SpectrogramDiscriminator(sample_rate, 512, 64, max_hz, channels=channels, eps=eps),
            SpectrogramDiscriminator(sample_rate, 256, 32, max_hz, channels=channels, eps=eps)
        ])

    def spectrograms(self, audio):
        """Each discriminator's input spectrogram of audio [B, N]"""
        return [discriminator.spectrogram(audio) for discriminator in self.discriminators]

    def score(self, spectrograms):
        """Each discriminator's (scores, features) of its input spectrogram (spectrograms())"""
        if self.dtype is None:
            return [discriminator.score(spectrogram) for discriminator, spectrogram in zip(self.discriminators, spectrograms)]
        device_type = spectrograms[0].device.type
        with torch.autocast(device_type, dtype=self.dtype):
            outputs = [discriminator.score(spectrogram) for discriminator, spectrogram in zip(self.discriminators, spectrograms)]
        return [(scores.float(), features) for scores, features in outputs]

    def forward(self, audio):
        return self.score(self.spectrograms(audio))


# The discriminators' loss (least squares) - real toward 1, rendered toward 0
def discriminator_loss(real_outputs, fake_outputs):
    loss = 0
    for (real, _), (fake, _) in zip(real_outputs, fake_outputs):
        loss = loss + ((real - 1) ** 2).mean() + (fake ** 2).mean()
    return loss / len(real_outputs)


# The generator's losses - adversarial (rendered toward 1) and feature matching
def generator_losses(real_outputs, fake_outputs):
    adversarial = 0
    matching = 0
    for (_, real_features), (fake, fake_features) in zip(real_outputs, fake_outputs):
        adversarial = adversarial + ((fake - 1) ** 2).mean()
        for real_feature, fake_feature in zip(real_features, fake_features):
            matching = matching + (real_feature.detach().float() - fake_feature.float()).abs().mean() / len(real_features)
    return adversarial / len(real_outputs), matching / len(real_outputs)
