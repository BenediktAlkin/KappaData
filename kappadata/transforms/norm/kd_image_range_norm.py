import torch
from torchvision.transforms.functional import to_tensor, normalize

from .base.kd_norm_base import KDNormBase


class KDImageRangeNorm(KDNormBase):
    def normalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        if x.ndim == 3:
            # per sample norm
            n_channels = x.size(0)
            values = tuple(.5 for _ in range(n_channels))
            return normalize(x, mean=values, std=values, inplace=inplace)
        else:
            # per batch norm
            assert x.ndim == 4
            if inplace:
                return x.mul_(2).sub_(1)
            return x * 2 - 1

    def denormalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        if x.ndim == 3:
            # per sample norm
            n_channels = x.size(0)
            std = tuple(2. for _ in range(n_channels))
            mean = tuple(-.5 for _ in range(n_channels))
            zero = tuple(0. for _ in range(n_channels))
            one = tuple(1. for _ in range(n_channels))
            x = normalize(x, mean=zero, std=std, inplace=inplace)
            return normalize(x, mean=mean, std=one, inplace=inplace)
        else:
            # per batch norm
            assert x.ndim == 4
            if inplace:
                return x.add_(1).div_(2)
            return (x + 1) / 2