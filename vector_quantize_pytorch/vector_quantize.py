
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def kmeans(x, num_clusters, num_iters=10):
    samples = rearrange(x, '... d -> (...) d')
    num_samples, dim = samples.shape
    dtype = x.dtype
    device = x.device

    if num_samples >= num_clusters:
        indices = torch.randperm(num_samples, device=device)[:num_clusters]
    else:
        indices = torch.randint(0, num_samples, num_clusters, device=device)

    centers = samples[indices]

    for _ in range(num_iters):
        dists = samples.pow(2).sum(1, keepdim=True) - 2 * samples @ centers.t() + centers.t().pow(2).sum(0, keepdim=True)
        buckets = dists.argmin(dim=-1)

        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)

        new_centers = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_centers.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_centers = new_centers / bins[..., None]
        centers = torch.where(zero_mask[..., None], centers, new_centers)
    return rearrange(centers, 'n d -> d n')

class VectorQuantize(nn.Module):
    def __init__ (self, dim, n_embed, decay=0.1, commitment=1., eps=1e-5, kmeans_init=False, kmeans_iters=10):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.commitment = commitment
        self.eps = eps

        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(dim, n_embed)

        embed = torch.randn(dim, n_embed)
        self.kmeans_iters = kmeans_iters
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())
    
    def init_embed_(self, data):
        embed = kmeans(data, self.n_embed, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, input):
        if not self.initted:
            self.init_embed_(input)

        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        commot_loss = 0
        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

            commit_loss = F.mse_loss(quantize.detach(), input) * self.commitment

            # straight through for gradient transfer
            quantize = input + (quantize - input).detach()
        return quantize, embed_ind, commit_loss
