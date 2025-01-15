from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import einsum, rearrange, repeat, reduce, pack, unpack

def identity(x):
    return x

class SimVQ(Module):
    def __init__(
        self, 
        dim, 
        codebook_size,
        init_fn: Callable = identity,
    ):
        super().__init__()
        codebook = torch.randn(codebook_size, dim)
        codebook = init_fn(codebook)

        self.codebook_to_codes = nn.Linear(dim, dim, bias = False)
        self.register_buffer('codebook', codebook)

    def forward(self, x):

        implicit_codebook = self.codebook_to_codes(self.codebook)

        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim = -1)
        
        quantized = get_at('[c] d, b n -> b n d', implicit_codebook, indices)

        commit_loss = (F.pairwise_distance(x, quantized.detach())**2).mean()

        # straight through

        quantized = x + (quantized - x).detach()

        return quantized, indices, commit_loss
        

        