import torch.nn as nn
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator


class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)

    def forward(self, x):
        return self.MRD(x), self.MPD(x)
