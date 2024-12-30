import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize import VectorQuantize

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        n_embed,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(codebook_size=n_embed, **kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_output = 0
        residual = x
        
        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indeces, loss = layer(residual)
            residual = residual - quantized
            quantized_output = quantized_output + quantized

            all_indices.append(indeces)
            all_losses.append(loss)

        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))

        return quantized_output, all_indices, all_losses
