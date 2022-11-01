import torch
from kappadata.datasets.kd_wrapper import KDWrapper
import numpy as np
from torch.nn.functional import one_hot

class CutmixWrapper(KDWrapper):
    def __init__(self, *args, alpha, p, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        self.alpha = alpha
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    @staticmethod
    def _context_has_params(ctx):
        return ctx is not None and "cutmix_apply" in ctx

    def _get_params(self, ctx, h=None, w=None):
        if not self._context_has_params(ctx):
            apply = self.rng.random() < self.p
            if ctx is not None:
                ctx["cutmix_apply"] = apply
            if apply:
                lamb = self.rng.beta(self.alpha, self.alpha)
                idx2 = self.rng.integers(0, len(self))
                bbox = self._rand_bbox(h=h, w=w, lamb=lamb)
                # correct lambda to actual area
                top, left, bot, right = bbox
                lamb = 1 - (right - left) * (bot - top) / (h * w)
                if ctx is not None:
                    ctx["cutmix_lambda"] = lamb
                    ctx["cutmix_idx2"] = idx2
                    ctx["cutmix_bbox"] = bbox
            else:
                ctx["cutmix_lambda"] = -1
                ctx["cutmix_idx2"] = -1
                ctx["cutmix_bbox"] = (-1, -1, -1, -1)
                return False, None, None, None
        else:
            apply = ctx["cutmix_apply"]
            if apply:
                lamb = ctx["cutmix_lambda"]
                idx2 = ctx["cutmix_idx2"]
                bbox = ctx["cutmix_bbox"]
            else:
                return False, None, None, None
        return True, lamb, idx2, bbox

    def _rand_bbox(self, h, w, lamb):
        cut_ratio = np.sqrt(1. - lamb)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        h_center = self.rng.integers(h)
        w_center = self.rng.integers(w)

        left = np.clip(w_center - cut_w // 2, 0, w)
        right = np.clip(w_center + cut_w // 2, 0, w)
        top = np.clip(h_center - cut_h // 2, 0, h)
        bot = np.clip(h_center + cut_h // 2, 0, h)

        return top, left, bot, right

    def getitem_x(self, idx, ctx=None):
        x1 = self.dataset.getitem_x(idx, ctx)
        assert torch.is_tensor(x1) and x1.ndim == 3, "convert image to tensor before CutmixWrapper"
        h, w = x1.shape[1:]
        apply, lamb, idx2, bbox = self._get_params(ctx=ctx, h=h, w=w)
        if not apply:
            return x1
        x2 = self.dataset.getitem_x(idx2, ctx)
        assert x1.shape == x2.shape
        top, left, bot, right = bbox
        x1[:, top:bot, left:right] = x2[:, top:bot, left:right]
        return x1

    def _getitem_class(self, idx, ctx):
        y = self.dataset.getitem_class(idx, ctx)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        if y.ndim == 0:
            y = one_hot(y, num_classes=self.dataset.n_classes)
        return y

    def getitem_class(self, idx, ctx=None):
        if self._context_has_params(ctx):
            apply, lamb, idx2, _ = self._get_params(ctx=ctx)
        else:
            h, w = self.getitem_x(idx, ctx).shape[1:]
            apply, lamb, idx2, _ = self._get_params(ctx=ctx, h=h, w=w)
        y1 = self._getitem_class(idx, ctx)
        if not apply:
            return y1
        y2 = self._getitem_class(idx2, ctx)
        return lamb * y1 + (1. - lamb) * y2

