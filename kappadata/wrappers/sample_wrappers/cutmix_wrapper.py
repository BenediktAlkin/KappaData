import torch

from kappadata.functional import get_random_bbox, get_area_of_bbox, cutmix_single
from .base.mix_wrapper_base import MixWrapperBase


class CutmixWrapper(MixWrapperBase):
    def __init__(self, *args, alpha, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

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
                bbox = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.rng)
                # correct lambda to actual area
                lamb = get_area_of_bbox(bbox=bbox, h=h, w=w)
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

    def getitem_x(self, idx, ctx=None):
        x1 = self.dataset.getitem_x(idx, ctx)
        assert torch.is_tensor(x1) and x1.ndim == 3, "convert image to tensor before CutmixWrapper"
        h, w = x1.shape[1:]
        apply, lamb, idx2, bbox = self._get_params(ctx=ctx, h=h, w=w)
        if not apply:
            return x1
        x2 = self.dataset.getitem_x(idx2, ctx)
        assert x1.shape == x2.shape
        return cutmix_single(x1=x1, x2=x2, bbox=bbox)

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
