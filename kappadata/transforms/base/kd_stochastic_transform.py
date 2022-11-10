import numpy as np

from .kd_transform import KDTransform


class KDStochasticTransform(KDTransform):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def reset_seed(self):
        assert self.seed is not None
        self.rng = np.random.default_rng(seed=self.seed)

    def __call__(self, x, ctx=None):
        raise NotImplementedError
