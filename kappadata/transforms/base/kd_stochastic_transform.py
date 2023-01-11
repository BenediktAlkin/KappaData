import numpy as np

from .kd_transform import KDTransform
from kappadata.utils.global_rng import GlobalRNG

class KDStochasticTransform(KDTransform):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def get_rng(self, index):
        if self.seed is None:
            return GlobalRNG()
        return np.random.default_rng(seed=self.seed + index)

    def __call__(self, x, ctx=None):
        raise NotImplementedError
