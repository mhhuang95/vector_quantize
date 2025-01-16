from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

# from einx import get_at
from einops import einsum, rearrange, repeat, reduce, pack, unpack

def identity(x):
    return x

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def safe_div(num, den, eps=1e-6):
    return num / den.clamp(min=eps)

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out
    
    return packed, inverse

def l2norm(x, dim=-1):
    return F.normalize(x, dim=dim)

def efficient_rotation_trick_transform(u, q, e):
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim = 1).detach()
    return (
        e - 
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) + 
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )


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
        
        # quantized = get_at('[c] d, b n -> b n d', implicit_codebook, indices)
        quantized = F.embedding(indices, implicit_codebook)

        commit_loss = (F.pairwise_distance(x, quantized.detach())**2).mean()

        # straight through

        # quantized = x + (quantized - x).detach()

        x, inverse = pack_one(x, '* d')
        quantized, _ = pack_one(quantized, '* d')

        norm_x = x.norm(dim=-1, keepdim = True)
        norm_quantize = quantized.norm(dim = -1, keepdim = True)

        rot_quantize = efficient_rotation_trick_transform(
            safe_div(x, norm_x),
            safe_div(quantized, norm_quantize),
            x
        ).squeeze()

        quantized = rot_quantize * safe_div(norm_quantize, norm_x).detach()

        x, quantized = inverse(x), inverse(quantized)


        return quantized, indices, commit_loss
        

        