import numpy as np
from .kd_transform import KDTransform
from torchvision.transforms import Compose

class KDComposeTransform(KDTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def set_seed(self, seed):
        # sample seeds with seed to avoid any potential patterns in child transforms
        rng = np.random.default_rng(seed)
        for t in self.transforms:
            if isinstance(t, KDStochasticTransform):
                t.set_seed(rng.integers(99999999))

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            if isinstance(t, KDTransform):
                x = t(x, ctx)
            else:
                x = t(x)
        return x