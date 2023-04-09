import numpy as np
import torch

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.error_messages import KD_MIX_WRAPPER_REQUIRES_SEED_OR_CONTEXT
from kappadata.utils.one_hot import to_one_hot_vector


class KDMixWrapper(KDWrapper):
    def __init__(
            self,
            dataset,
            mixup_alpha: float = None,
            cutmix_alpha: float = None,
            mixup_p: float = 0.5,
            cutmix_p: float = 0.5,
            seed: int = None,
    ):
        super().__init__(dataset=dataset)
        # check probabilities
        assert isinstance(mixup_p, (int, float)) and 0. <= mixup_p <= 1., f"invalid mixup_p {mixup_p}"
        assert isinstance(cutmix_p, (int, float)) and 0. <= cutmix_p <= 1., f"invalid mixup_p {mixup_p}"
        assert 0. < mixup_p + cutmix_p <= 1., f"0 < mixup_p + cutmix_p <= 1 (got {mixup_p + cutmix_p})"
        if mixup_p + cutmix_p != 1.:
            raise NotImplementedError

        # check alphas
        if mixup_p == 0.:
            assert mixup_alpha is None
        else:
            assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        if cutmix_p == 0.:
            assert cutmix_alpha is None
        else:
            assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha

        # initialize
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.seed = seed
        # rng with seed is set in _shared
        self.rng = np.random.default_rng()

        # TODO port to per property key
        self.ctx_key = self.ctx_prefix

    @property
    def total_p(self) -> float:
        return self.mixup_p + self.cutmix_p

    def get_random_bbox(self, h, w, lamb):
        bbox_hcenter = torch.tensor(self.rng.integers(h))
        bbox_wcenter = torch.tensor(self.rng.integers(w))

        area_half = 0.5 * (1.0 - lamb).sqrt()
        bbox_h_half = (area_half * h).floor()
        bbox_w_half = (area_half * w).floor()

        top = torch.clamp(bbox_hcenter - bbox_h_half, min=0).type(torch.long)
        bot = torch.clamp(bbox_hcenter + bbox_h_half, max=h).type(torch.long)
        left = torch.clamp(bbox_wcenter - bbox_w_half, min=0).type(torch.long)
        right = torch.clamp(bbox_wcenter + bbox_w_half, max=w).type(torch.long)
        bbox = torch.stack([top, left, bot, right], dim=0)

        lamb_adjusted = 1.0 - (bot - top) * (right - left) / (h * w)

        return bbox, lamb_adjusted

    def _shared(self, idx, ctx):
        assert ctx is not None or self.seed is not None, KD_MIX_WRAPPER_REQUIRES_SEED_OR_CONTEXT
        if ctx is not None and self.ctx_key in ctx:
            return ctx[self.ctx_key]
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed + idx)

        x = self.dataset.getitem_x(idx, ctx=ctx)
        # check if apply
        p = self.rng.random()
        if p > self.total_p:
            values = (False, -1, torch.tensor(-1.), x, torch.tensor((-1, -1, -1, -1)))
            ctx[self.ctx_key] = values
            return values

        # sample parameters
        use_cutmix = p < self.cutmix_p
        idx2 = int(self.rng.integers(len(self.dataset)))
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lamb = torch.tensor(self.rng.beta(alpha, alpha))
        if use_cutmix:
            h, w = x.shape[1:]
            bbox, lamb = self.get_random_bbox(h=h, w=w, lamb=lamb)
        else:
            bbox = torch.tensor((-1, -1, -1, -1))

        # save to ctx
        values = (use_cutmix, idx2, lamb, x, bbox)
        if ctx is not None:
            ctx[self.ctx_key] = values
        return values

    def getitem_x(self, idx, ctx=None):
        use_cutmix, idx2, lamb, x, bbox = self._shared(idx, ctx=ctx)
        if idx2 == -1:
            return x
        x2 = self.dataset.getitem_x(idx2, ctx={})
        if use_cutmix:
            top, left, bot, right = bbox
            x[..., top:bot, left:right] = x2[..., top:bot, left:right]
        else:
            x_lamb = lamb.view(*[1] * x.ndim)
            x.mul_(x_lamb).add_(x2.mul_((1. - x_lamb)))
        return x

    def getitem_class(self, idx, ctx=None):
        _, idx2, lamb, _, _ = self._shared(idx, ctx=ctx)
        y = self.dataset.getitem_class(idx, ctx=ctx)
        if idx2 == -1:
            return y
        y2 = self.dataset.getitem_class(idx2, ctx={})
        y = to_one_hot_vector(y, n_classes=self.dataset.getdim_class())
        y2 = to_one_hot_vector(y2, n_classes=self.dataset.getdim_class())

        y.mul_(lamb).add_(y2.mul_(1. - lamb))
        return y
