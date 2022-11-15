import torch
from torchvision.transforms.functional import to_tensor, normalize

from .base.kd_transform import KDTransform


class ImageRangeNorm(KDTransform):
    def __call__(self, x, _=None):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        n_channels = x.size(0)
        values = tuple(.5 for _ in range(n_channels))
        return normalize(x, mean=values, std=values, inplace=True)

    @staticmethod
    def denormalize(x, inplace=True):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        n_channels = x.size(0)
        stds = tuple(2. for _ in range(n_channels))
        means = tuple(-.5 for _ in range(n_channels))
        zeros = tuple(0. for _ in range(n_channels))
        ones = tuple(1. for _ in range(n_channels))
        normalize(x, mean=zeros, std=stds, inplace=inplace)
        return normalize(x, mean=means, std=ones, inplace=inplace)