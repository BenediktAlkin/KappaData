import numpy as np

from .kd_transform import KDTransform


class KDStochasticTransform(KDTransform):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        # TODO seed
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    # TODO reset_seed is deprecated
    def reset_seed(self):
        assert self.seed is not None
        self.rng = np.random.default_rng(seed=self.seed)

    def set_rng(self, rng):
        assert self.seed is None, "can't use set_rng on transforms with seed"
        self.rng = rng

    def __call__(self, x, ctx=None):
        raise NotImplementedError
