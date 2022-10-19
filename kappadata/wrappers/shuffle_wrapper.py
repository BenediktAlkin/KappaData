import numpy as np

from .base.wrapper_base import WrapperBase


class ShuffleWrapper(WrapperBase):
    def __init__(self, dataset, seed=None):
        self.seed = seed
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random
        indices = np.arange(len(dataset))
        self.rng.shuffle(indices)
        super().__init__(dataset=dataset, indices=indices)
