import numpy as np
import torch

from kappadata.functional import to_onehot_matrix
from .kd_collator import KDCollator


class MixCollatorBase(KDCollator):
    def __init__(self, p=1., n_classes=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        assert n_classes is None or (isinstance(n_classes, int) and n_classes > 1)
        self.p = p
        self.n_classes = n_classes
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed=seed)

    @property
    def default_collate_mode(self):
        return "before"

    def collate(self, batch, dataset_mode, ctx=None):
        if dataset_mode == "x":
            x = batch
            y = None
        elif dataset_mode == "x class":
            x, y = batch
            y = to_onehot_matrix(y, n_classes=self.n_classes).type(torch.float32)
        else:
            raise NotImplementedError
        batch_size = len(x)
        return self._collate(x, y, batch_size, ctx)

    def _collate(self, x, y, batch_size, ctx):
        raise NotImplementedError

    @staticmethod
    def _mix_y(y, lamb, idx2):
        if y is None:
            return y
        lamb_y = lamb.view(-1, 1)
        return lamb_y * y + (1. - lamb_y) * y[idx2]
