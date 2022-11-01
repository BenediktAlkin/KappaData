import torch
from kappadata.datasets.kd_wrapper import KDWrapper
import numpy as np
from torch.nn.functional import one_hot

class MixupWrapper(KDWrapper):
    def __init__(self, *args, alpha, p, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        self.alpha = alpha
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    def _get_params(self, ctx):
        if ctx is None or "mixup_lambda" not in ctx:
            apply = self.rng.random() < self.p
            ctx["mixup_apply"] = apply
            if apply:
                lamb = self.rng.beta(self.alpha, self.alpha)
                idx2 = self.rng.integers(0, len(self))
                ctx["mixup_lambda"] = lamb
                ctx["mixup_idx2"] = idx2
            else:
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

    def getitem_class(self, idx, ctx=None):
        apply, lamb, idx2 = self._get_params(ctx)
        y1 = self.dataset.getitem_class(idx, ctx)
        if not torch.is_tensor(y1):
            y1 = torch.tensor(y1)
        y1 = one_hot(y1, num_classes=self.dataset.n_classes)
        if not apply:
            return y1
        y2 = self.dataset.getitem_class(idx2, ctx)
        if not torch.is_tensor(y2):
            y2 = torch.tensor(y2)
        y2 = one_hot(y2, num_classes=self.dataset.n_classes)
        return lamb * y1 + (1. - lamb) * y2

