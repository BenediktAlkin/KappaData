import einops
import torch
from torchvision.transforms.functional import to_tensor, normalize

from .base.kd_norm_base import KDNormBase


class KDImageNorm(KDNormBase):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def normalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        if x.ndim == 3:
            # per sample norm
            return normalize(x, mean=self.mean, std=self.std, inplace=inplace)
        else:
            # per batch norm
            assert x.ndim == 4
            mean = einops.rearrange(torch.tensor(self.mean, device=x.device), "channels -> 1 channels 1 1")
            std = einops.rearrange(torch.tensor(self.std, device=x.device), "channels -> 1 channels 1 1")
            if inplace:
                return x.sub_(mean).div_(std)
            return (x - mean) / std

    def denormalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        if x.ndim == 3:
            # per sample norm
            inv_std = tuple(1. / std for std in self.std)
            inv_mean = tuple(-mean for mean in self.mean)
            zero = tuple(0. for _ in self.mean)
            one = tuple(1. for _ in self.std)
            x = normalize(x, mean=zero, std=inv_std, inplace=inplace)
            return normalize(x, mean=inv_mean, std=one, inplace=inplace)
        else:
            # per batch norm
            assert x.ndim == 4
            mean = einops.rearrange(torch.tensor(self.mean, device=x.device), "channels -> 1 channels 1 1")
            std = einops.rearrange(torch.tensor(self.std, device=x.device), "channels -> 1 channels 1 1")
            if inplace:
                return x.mul_(std).add_(mean)
