import numpy as np

from .kd_transform import KDTransform


class KDStochasticTransform(KDTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # problem: if no seed is passed -> default_rng will not be affected by np.random.set_seed
        # solution: sample a random integer from np.random.randint (which is affected by np.random.set_seed)
        seed = np.random.randint(np.iinfo(np.int32).max)
        self.rng = np.random.default_rng(seed=seed)

    @property
    def is_deterministic(self):
        return False

    def set_rng(self, rng):
        self.rng = rng
        return self

    def __call__(self, x, ctx=None):
        raise NotImplementedError
