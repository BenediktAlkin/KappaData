import torch
from kappadata.datasets.kd_wrapper import KDWrapper
import numpy as np
from torch.nn.functional import one_hot

class MixupWrapper(KDWrapper):
    def __init__(self, *args, alpha, p=1., seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        self.alpha = alpha
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    def _get_params(self, ctx):
        if ctx is None or "mixup_apply" not in ctx:
            apply = self.rng.random() < self.p
            if ctx is not None:
                ctx["mixup_apply"] = apply
            if apply:
                lamb = self.rng.beta(self.alpha, self.alpha)
                idx2 = self.rng.integers(0, len(self))
                if ctx is not None:
                    ctx["mixup_lambda"] = lamb
                    ctx["mixup_idx2"] = idx2
            else:
                ctx["mixup_lambda"] = -1
                ctx["mixup_idx2"] = -1
                return False, None, None
        else:
            apply = ctx["mixup_apply"]
            if apply:
                lamb = ctx["mixup_lambda"]
                idx2 = ctx["mixup_idx2"]
            else:
                return False, None, None
        return True, lamb, idx2

    def getitem_x(self, idx, ctx=None):
        apply, lamb, idx2 = self._get_params(ctx)
        x1 = self.dataset.getitem_x(idx, ctx)
        if not apply:
            return x1
        x2 = self.dataset.getitem_x(idx2, ctx)
        return lamb * x1 + (1. - lamb) * x2


    def _getitem_class(self, idx, ctx):
        y = self.dataset.getitem_class(idx, ctx)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        if y.ndim == 0:
            y = one_hot(y, num_classes=self.dataset.n_classes)
        return y

    def getitem_class(self, idx, ctx=None):
        apply, lamb, idx2 = self._get_params(ctx)
        y1 = self._getitem_class(idx, ctx)
        if not apply:
            return y1
        y2 = self._getitem_class(idx2, ctx)
        return lamb * y1 + (1. - lamb) * y2

