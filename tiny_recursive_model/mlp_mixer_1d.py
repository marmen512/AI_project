from functools import partial

from torch import nn
from torch.nn import Module, LayerNorm
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim, bias = False)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, dim_hidden, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim_hidden, dim),
        nn.Dropout(dropout)
    )

def MLPMixer1D(*, dim, depth, seq_len, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(seq_len, int(expansion_factor * dim), dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, int(expansion_factor_token * dim), dropout, chan_last))
        ) for _ in range(depth)],
        LayerNorm(dim, bias = False)
    )

# quick test

if __name__ == '__main__':

    import torch
    tokens = torch.randn(1, 1024, 512)
    mixer = MLPMixer1D(dim = 512, depth = 4, seq_len = 1024)

    assert mixer(tokens).shape == tokens.shape
