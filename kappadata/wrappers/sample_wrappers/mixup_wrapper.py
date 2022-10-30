import torch
from kappadata.datasets.kd_wrapper import KDWrapper
import numpy as np
from torch.nn.functional import one_hot

class MixupWrapper(KDWrapper):
    def __init__(self, alpha, seed=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(alpha, (int, float)) and alpha > 0
        self.alpha = alpha
        self.rng = np.random.default_rng(seed=seed)

    def _get_params(self, ctx):
        if ctx is None or "mixup_lambda" not in ctx:
            lamb = self.rng.beta(self.alpha, self.alpha)
            idx2 = self.rng.integers(0, len(self))
            ctx["mixup_lambda"] = lamb
            ctx["mixup_idx2"] = idx2
        else:
            lamb = ctx["mixup_lambda"]
            idx2 = ctx["mixup_idx2"]
        return lamb, idx2

    def getitem_x(self, idx, ctx=None):
        lamb, idx2 = self._get_params(ctx)
        x1 = self.dataset.getitem_x(idx, ctx)
        x2 = self.dataset.getitem_x(idx2, ctx)
        return lamb * x1 + (1. - lamb) * x2

    def getitem_class(self, idx, ctx=None):
        lamb, idx2 = self._get_params(ctx)
        y1 = self.dataset.getitem_class(idx, ctx)
        y2 = self.dataset.getitem_class(idx2, ctx)
        # onehot expects long tensor
        if not torch.is_tensor(y1):
            y1 = torch.tensor(y1)
        if not torch.is_tensor(y2):
            y2 = torch.tensor(y2)
        y1 = one_hot(y1, num_classes=self.dataset.n_classes)
        y2 = one_hot(y2, num_classes=self.dataset.n_classes)
        return lamb * y1 + (1. - lamb) * y2

