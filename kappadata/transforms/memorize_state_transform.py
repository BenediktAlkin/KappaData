import torch
from torchvision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class MemorizeStateTransform(KDTransform):
    def __init__(self, state_name):
        self.state_name = state_name

    def __call__(self, x, ctx=None):
        if ctx is not None:
            if not torch.is_tensor(x):
                x = to_tensor(x)
            ctx[self.state_name] = x.clone()
        return x
