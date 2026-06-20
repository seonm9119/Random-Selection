from simclr.models import Projector
from torch import nn

class Simclr(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = Projector(args)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def encode(self, x):
        return self.encoder.model(x, out='h')