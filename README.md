## VQVAE Implementation

VQVAE model from https://arxiv.org/abs/1711.00937



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