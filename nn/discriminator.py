# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

"""
The adversarial critic for training (optional, nn/train.py --adversarial) - spectrogram discriminators that judge the
synthesizer's renderings of the network's parameters against the recordings. A deterministic network trained by
spectral regression over-smooths: where a formant's position is uncertain, a broad peak minimizes the expected error,
so the predicted bandwidths come out wide and the voice muffled. The critic penalizes what no recording looks like.
Its gradient reaches only the interpretable parameters (through nn/dsynth.py), so they stay interpretable - the critic
is used in training only, never exported.

Each discriminator is a small 2-D convolution stack over a log-mel spectrogram (time x mel) at one resolution; the
losses are least-squares GAN losses and feature matching (the L1 distance of the discriminators' intermediate
activations, real against rendered).
"""

import torch

from . import spectral


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
            torch.nn.Sequential(torch.nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)), leaky),
            torch.nn.Sequential(torch.nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)), leaky),
            torch.nn.Sequential(torch.nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4)), leaky),
            torch.nn.Sequential(torch.nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)), leaky)
        ])
        self.out = torch.nn.Conv2d(channels, 1, (3, 3), padding=(1, 1))

    def forward(self, audio):
        """Audio [B, N] to (scores [B, 1, T, M'], intermediate features)"""
        magnitude = spectral.stft_magnitude(audio, self.fft_size, self.fft_size // 4, eps=self.eps)
        x = torch.log(magnitude ** 2 @ self.bank.T + 1e-5)[:, None]
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return self.out(x), features


class Critic(torch.nn.Module):
    """Spectrogram discriminators at several resolutions"""

    def __init__(self, sample_rate, max_hz, eps=0.0):
        super().__init__()
        self.discriminators = torch.nn.ModuleList([
            SpectrogramDiscriminator(sample_rate, 1024, 80, max_hz, eps=eps),
            SpectrogramDiscriminator(sample_rate, 512, 64, max_hz, eps=eps),
            SpectrogramDiscriminator(sample_rate, 256, 32, max_hz, eps=eps)
        ])

    def forward(self, audio):
        return [discriminator(audio) for discriminator in self.discriminators]


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
            matching = matching + (real_feature.detach() - fake_feature).abs().mean() / len(real_features)
    return adversarial / len(real_outputs), matching / len(real_outputs)
