import torch
from torchvision.transforms.functional import rotate

from .base.kd_transform import KDTransform


class PatchwiseRandomRotation(KDTransform):
    def __init__(self, seed=None):
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __call__(self, x, ctx=None):
        c, l, patch_h, patch_w = x.shape
        # sample rotations
        rotations = torch.randint(0, 4, size=(l,), generator=self.rng) * 90
        if ctx is not None:
            ctx["patchwise_random_rotation"] = rotations
        # rotate patches
        for i in range(l):
            x[:, i] = rotate(x[:, i], angle=rotations[i].item())
        return x
