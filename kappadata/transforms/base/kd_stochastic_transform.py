import numpy as np
from torch.utils.data import get_worker_info
from .kd_transform import KDTransform


class KDStochasticTransform(KDTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # np.random.default_rng() is not dependent on the global numpy seed -> sample seed from np.random.randint
        # ATTENTION: as the rng is initialized in the constructor it will be the same on all dataloader processes
        # -> overwrite in worker_init_fn to have different seeds per process
        # -> determinism is dependent on num_workers
        # TODO add a check
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
