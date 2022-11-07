import torch
from torchvision.transforms.functional import to_tensor, normalize
from .base.kd_transform import KDTransform

class ImageRangeNorm(KDTransform):
    def __call__(self, x, _=None):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        n_channels = x.size(0)
        values = tuple(0.5 for _ in range(n_channels))
        return normalize(x, mean=values, std=values, inplace=True)
