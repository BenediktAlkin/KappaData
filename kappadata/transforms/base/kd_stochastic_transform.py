import numpy as np

from .kd_transform import KDTransform


class KDStochasticTransform(KDTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng()

    @property
    def is_deterministic(self):
        return False

    def set_rng(self, rng):
        self.rng = rng
        return self

    def __call__(self, x, ctx=None):
        raise NotImplementedError
