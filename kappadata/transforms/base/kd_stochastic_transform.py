import numpy as np
from torch.utils.data import get_worker_info
from .kd_transform import KDTransform


class KDStochasticTransform(KDTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = None

    @property
    def rng(self):
        # if rng is initialized in __init__ all worker processes start from the same rng
        # by lazily initializing self._rng is None when the process is spawned so we can control the initial state
        if self._rng is None:
            # problem: np.random.default_rng() is not influenced by the numpy global seed
            # info: dataloader implicitly sets a different global seed per worker process
            # solution: seed with a random int generated from the global rng
            # result: transform will be deterministic as long as num_workers are the same
            seed = np.random.randint(np.iinfo(np.int32).max)
            self._rng =  np.random.default_rng(seed=seed)
        return self._rng

    @property
    def is_deterministic(self):
        return False

    def set_rng(self, rng):
        self._rng = rng
        return self

    def __call__(self, x, ctx=None):
        raise NotImplementedError
