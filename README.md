## VQVAE Implementation

VQVAE model from https://arxiv.org/abs/1711.00937. VQ could be used as a clustering algorithm.


## Residual VQ

RQVAE was intorduced in this paper https://arxiv.org/abs/2107.03312 to recursively quantize the residuals of the embedding. RQVAE could be viewd as a hierarchical clustering algorithm.

```python
 vq = ResidualVQ(
        num_quantizers=2,
        n_embed = 512,
        dim = 256,
        decay = 0.8,
        commitment = 1.,
        kmeans_init=True, 
        kmeans_iters=2,
        use_cosine_sim = True,
        orthogonal_reg_weight = 0.5,
    )
```

## Kmeans Initialization

Initialize the codebook with kmeans centroid as proposed in this paper https://arxiv.org/abs/2107.03312

```python
import torch
from vector_quantize_pytorch import VectorQuantize
vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 4,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
)
x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```


## Cosine Similarity VQ

this paper https://openreview.net/forum?id=pfNyExj7z2 proposed to use cosine similarity to replace the l2 distance in nearest neighbor computation. They claim that cosine similarity leads to codebook usage improvement.

```python
import torch
from vector_quantize_pytorch import VectorQuantize
vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 4,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
    use_cosine_sim = True,
)
x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```

## Orthogonal Regularizer

this paper https://arxiv.org/abs/2112.00384 introduces an orthogonal regularizer into the loss and claimed significant performance improvement on VQGAN

```python
import torch
from vector_quantize_pytorch import VectorQuantize
vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 4,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
    use_cosine_sim = True,
    orthogonal_reg_weight = 0.5,
)
x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```


## Shared codebook in RQVAE

Shared codebook was proposed in this paper https://arxiv.org/pdf/2203.01941 to reduce the parameter and increase the codebook utilization.


```python
import torch
from vector_quantize_pytorch import VectorQuantize
rq = ResidualVQ(
    num_quantizers=2,
    n_embed = 512,
    shared_codebook = True,
    dim = 256,
    decay = 0.8,
    commitment = 1.,
    kmeans_init=True, 
    kmeans_iters=2,
    use_cosine_sim = True,
    orthogonal_reg_weight = 0.5,
)
x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = rq(x)
```

## Rotation Trick in VQ

The rotation trick introduced in this paper https://arxiv.org/pdf/2410.06424, which propagate gradients smoothly through the vector quantization.

```python
import torch
from vector_quantize_pytorch import VectorQuantize
vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 4,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
    use_cosine_sim = True,
    orthogonal_reg_weight = 0.5,
    rotation_trick=True,
)
x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```