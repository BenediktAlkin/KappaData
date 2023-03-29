import einops
import torch
from torchvision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class KDRearrange(KDTransform):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x, ctx=None):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        return einops.rearrange(x, self.pattern)
