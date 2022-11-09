import numpy as np

from .kd_transform import KDTransform

class KDStochasticTransform(KDTransform):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed=seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, x, ctx=None):
        raise NotImplementedError
