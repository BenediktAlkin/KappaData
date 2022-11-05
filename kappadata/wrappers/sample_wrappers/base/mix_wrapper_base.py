import numpy as np
import torch

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.functional import to_onehot_vector


class MixWrapperBase(KDWrapper):
    def __init__(self, *args, p=1., seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    def _getitem_class(self, idx, ctx):
        y = self.dataset.getitem_class(idx, ctx)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return to_onehot_vector(y, n_classes=self.dataset.n_classes)
