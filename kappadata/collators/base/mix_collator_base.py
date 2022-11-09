import numpy as np
import torch

from kappadata.functional.onehot import to_onehot_matrix
from .kd_collator import KDCollator


class MixCollatorBase(KDCollator):
    def __init__(self, p=1., p_mode="batch", n_classes=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        assert p_mode in ["batch", "sample"]
        assert n_classes is None or (isinstance(n_classes, int) and n_classes > 1)
        self.p = p
        self._is_batch_p_mode = p_mode == "batch"
        self.n_classes = n_classes
        self.np_rng = np.random.default_rng(seed=seed)
        self.th_rng = torch.Generator()
        if seed is not None:
            self.th_rng.manual_seed(seed + 1)

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
        if self._is_batch_p_mode:
            apply = torch.rand(size=(), generator=self.th_rng) < self.p
            if apply:
                return self._collate_batchwise(x, y, batch_size, ctx)
            return x, y
        else:
            apply = torch.rand(size=(batch_size,), generator=self.th_rng) < self.p
            return self._collate_samplewise(apply=apply, x=x, y=y, batch_size=batch_size, ctx=ctx)

    def _collate_batchwise(self, x, y, batch_size, ctx):
        raise NotImplementedError

    def _collate_samplewise(self, apply, x, y, batch_size, ctx):
        raise NotImplementedError
