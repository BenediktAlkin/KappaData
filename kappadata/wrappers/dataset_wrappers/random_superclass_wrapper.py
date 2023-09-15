import math

import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.global_rng import GlobalRng


class RandomSuperclassWrapper(KDWrapper):
    """
    randomly merge classes into superclasses
    Example input: dataset with 10 classes and classes_per_superclass=2
    Example output: dataset with 5 classes where each class consists of 2 input classes
    """

    def __init__(self, dataset, classes_per_superclass, seed=None):
        super().__init__(dataset=dataset)
        self.classes_per_subclass = classes_per_superclass
        if seed is not None:
            # static pseudo labels (they dont change from epoch to epoch)
            rng = np.random.default_rng(seed=seed)
        else:
            # dynamic pseudo labels (resampled for every epoch)
            rng = GlobalRng()
        self.perm = rng.permutation(dataset.getdim_class())

    def getshape_class(self):
        return math.ceil(self.dataset.getdim_class() / self.classes_per_subclass),

    def _map_cls(self, cls):
        return self.perm[cls] // self.classes_per_subclass

    def getitem_class(self, idx, ctx=None):
        cls = self.dataset.getitem_class(idx, ctx=ctx)
        return self._map_cls(cls)

    def getall_class(self):
        classes = self.dataset.getall_class()
        return [self._map_cls(cls) for cls in classes]
