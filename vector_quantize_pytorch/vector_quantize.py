
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def orthogonal_loss_fn(t):
    n = t.shape[0]
    normed_codes = F.normalize(t, p = 2, dim = -1)
    print(normed_codes.shape)
    cosine_sim = einsum(normed_codes, normed_codes, 'i d, j d -> i j')
    print(cosine_sim)
    return (cosine_sim ** 2).sum() / (n ** 2) - (1 / n)

def kmeans(x, num_clusters, num_iters=10, use_cosine_sim = False):
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
        if use_cosine_sim:
            dists = samples @ centers.t()
            buckets = dists.max(dim = -1).indices
        else:
            dists = samples.pow(2).sum(1, keepdim=True) - 2 * samples @ centers.t() + centers.t().pow(2).sum(0, keepdim=True)
            buckets = dists.argmin(dim=-1)

        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)

        new_centers = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_centers.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_centers = new_centers / bins[..., None]

        if use_cosine_sim:
            new_centers = F.normalize(new_centers, dim = -1)

        centers = torch.where(zero_mask[..., None], centers, new_centers)
    return centers


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5
    ):
        super().__init__()
        self.decay = decay

        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())


    def init_embed_(self, data):
        embed = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed.t()
    
        if not self.initted:
            self.init_embed_(flatten)
        
        dist = -(flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim = -1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy(embed_normalized)
        
        return quantize, embed_ind



class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = F.normalize(torch.randn(codebook_size, dim), dim = -1)
        else:
            embed = torch.zeros(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)

    def init_embed_(self, data):
        embed = kmeans(data, self.codebook_size, self.kmeans_iters, use_cosine_sim=True)
        self.embed.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = F.normalize(flatten, dim = -1)
        embed = F.normalize(self.embed, dim = -1)
    
        if not self.initted:
            self.init_embed_(flatten)
        
        dist = flatten @ embed.t()
        embed_ind = dist.max(dim = -1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(0)
            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = F.normalize(embed_normalized, dim = -1)
            embed_normalized = torch.where(zero_mask[..., None], embed, embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)
        
        return quantize, embed_ind


class VectorQuantize(nn.Module):
    def __init__ (self, dim, codebook_size, codebook_dim = None, decay=0.1, commitment=1., eps=1e-5, kmeans_init=False, kmeans_iters=10, use_cosine_sim=False, orthogonal_reg_weight = 0.,):
        super().__init__()

        if not codebook_dim:
            codebook_dim = dim
            require_projection = False
        else:
            require_projection = True
        self.projection_in = nn.Linear(dim, codebook_dim) if require_projection else nn.Identity()
        self.projection_out = nn.Linear(codebook_dim, dim) if require_projection else nn.Identity()

        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment = commitment
        self.eps = eps
        has_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_orthogonal_loss = has_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight

        klass = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        self._codebook = klass(
            dim = codebook_dim,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps,
        )
    
    @property
    def codebook(self):
        return self._codebook.codebook
    
    def forward(self, x):
        dtype = x.dtype
        x = self.projection_in(x)

        quantize, embed_ind = self._codebook(x)
        
        loss = 0
        if self.training:

            loss = F.mse_loss(quantize.detach(), x) * self.commitment

            if self.has_orthogonal_loss:
                codebook = self._codebook.embed
                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

            # straight through for gradient transfer
            quantize = x + (quantize - x).detach()
        
        quantize = self.projection_out(quantize)
        return quantize, embed_ind, loss
