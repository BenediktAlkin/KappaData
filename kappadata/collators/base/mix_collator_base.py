import numpy as np
import torch
from kappadata.functional import to_onehot_matrix
from .collator_base import CollatorBase


class MixCollatorBase(CollatorBase):
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

        # create bool flags
        if self.dataset_mode == "x":
            self._is_x = True
            self._is_xy = False
        elif self.dataset_mode == "x class":
            self._is_x = False
            self._is_xy = True
        else:
            raise NotImplementedError

    def collate(self, batch, ctx=None):
        if self._is_x:
            x = batch
            y = None
        elif self._is_xy:
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
