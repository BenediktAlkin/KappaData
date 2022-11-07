import torch

from kappadata.functional import get_random_bbox, get_area_of_bbox, cutmix_single
from .base.mix_wrapper_base import MixWrapperBase


class MixWrapper(MixWrapperBase):
    def __init__(self, *args, cutmix_alpha, mixup_alpha, cutmix_p=.5, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha
        assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        assert isinstance(cutmix_p, (int, float)) and 0. < cutmix_p < 1.
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.cutmix_p = cutmix_p

    def _get_apply_and_usecutmix(self, ctx):
        if ctx is None or "mix_apply" not in ctx:
            apply = self.rng.random() < self.p
            if apply:
                use_cutmix = self.rng.random() < self.cutmix_p
            else:
                use_cutmix = False
            if ctx is not None:
                ctx["mix_apply"] = apply
                ctx["mix_usecutmix"] = use_cutmix
                if not apply:
                    ctx["mix_idx2"] = -1
                    ctx["mix_lambda"] = -1
                    ctx["mix_bbox"] = (-1, -1, -1, -1)
        else:
            apply = ctx["mix_apply"]
            use_cutmix = ctx["mix_usecutmix"]
        return apply, use_cutmix

    def _get_idx2(self, ctx):
        if ctx is not None and "mix_idx2" in ctx:
            return ctx["mix_idx2"]
        idx2 = self.rng.integers(0, len(self))
        if ctx is not None:
            ctx["mix_idx2"] = idx2
        return idx2

    def _get_cutmix_params(self, ctx, h=None, w=None):
        if ctx is None or "mix_lambda" not in ctx:
            lamb = self.rng.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbox = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.rng)
            # correct lambda to actual area
            lamb = get_area_of_bbox(bbox=bbox, h=h, w=w)
            if ctx is not None:
                ctx["mix_lambda"] = lamb
                ctx["mix_bbox"] = bbox
        else:
            lamb = ctx["mix_lambda"]
            bbox = ctx["mix_bbox"]
        return lamb, bbox

    def _get_mixup_params(self, ctx):
        if ctx is None or "mix_lambda" not in ctx:
            lamb = self.rng.beta(self.cutmix_alpha, self.cutmix_alpha)
            if ctx is not None:
                ctx["mix_lambda"] = lamb
                ctx["mix_bbox"] = (-1, -1, -1, -1)
        else:
            lamb = ctx["mix_lambda"]
        return lamb

    def getitem_x(self, idx, ctx=None):
        apply, use_cutmix = self._get_apply_and_usecutmix(ctx)
        x1 = self.dataset.getitem_x(idx, ctx)
        if not apply:
            return x1
        idx2 = self._get_idx2(ctx)
        x2 = self.dataset.getitem_x(idx2, ctx)
        if use_cutmix:
            assert torch.is_tensor(x1) and x1.ndim == 3, "convert image to tensor before MixWrapper"
            assert x1.shape == x2.shape
            h, w = x1.shape[1:]
            lamb, bbox = self._get_cutmix_params(ctx=ctx, h=h, w=w)
            return cutmix_single(x1=x1, x2=x2, bbox=bbox)
        else:
            lamb = self._get_mixup_params(ctx=ctx)
            return lamb * x1 + (1. - lamb) * x2

    def getitem_class(self, idx, ctx=None):
        apply, use_cutmix = self._get_apply_and_usecutmix(ctx)
        if not apply:
            return self._getitem_class(idx, ctx)
        if use_cutmix:
            if ctx is not None and "mix_bbox" in ctx:
                lamb, _ = self._get_cutmix_params(ctx=ctx)
            else:
                h, w = self.getitem_x(idx, ctx).shape[1:]
                lamb, _ = self._get_cutmix_params(ctx=ctx, h=h, w=w)
        else:
            lamb = self._get_mixup_params(ctx)
        y1 = self._getitem_class(idx, ctx)
        idx2 = self._get_idx2(ctx)
        y2 = self._getitem_class(idx2, ctx)
        return lamb * y1 + (1. - lamb) * y2
