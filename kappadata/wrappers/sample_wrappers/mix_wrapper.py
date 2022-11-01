import torch
from kappadata.datasets.kd_wrapper import KDWrapper
import numpy as np
from torch.nn.functional import one_hot

class MixWrapper(KDWrapper):
    def __init__(self, *args, cutmix_alpha, mixup_alpha, p=1., cutmix_p=.5, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha
        assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        assert isinstance(cutmix_p, (int, float)) and 0. < cutmix_p <= 1.
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.p = p
        self.cutmix_p = cutmix_p
        self.rng = np.random.default_rng(seed=seed)

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
            bbox = self._rand_bbox(h=h, w=w, lamb=lamb)
            # correct lambda to actual area
            top, left, bot, right = bbox
            lamb = 1 - (right - left) * (bot - top) / (h * w)
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
            top, left, bot, right = bbox
            x1[:, top:bot, left:right] = x2[:, top:bot, left:right]
            return x1
        else:
            lamb = self._get_mixup_params(ctx=ctx)
            return lamb * x1 + (1. - lamb) * x2

    def _getitem_class(self, idx, ctx):
        y = self.dataset.getitem_class(idx, ctx)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        if y.ndim == 0:
            y = one_hot(y, num_classes=self.dataset.n_classes)
        return y

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

