import numpy as np
import torch

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.error_messages import REQUIRES_MIXUP_P_OR_CUTMIX_P
from kappadata.utils.one_hot import to_one_hot_vector


class KDMixWrapper(KDWrapper):
    def __init__(self, dataset, mixup_p=None, cutmix_p=None, mixup_alpha=None, cutmix_alpha=None, seed=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)

        # check probabilities
        assert (mixup_p is not None) or (cutmix_p is not None), REQUIRES_MIXUP_P_OR_CUTMIX_P
        mixup_p = mixup_p or 0.
        cutmix_p = cutmix_p or 0.
        assert isinstance(mixup_p, (int, float)) and 0. <= mixup_p <= 1., f"invalid mixup_p {mixup_p}"
        assert isinstance(cutmix_p, (int, float)) and 0. <= cutmix_p <= 1., f"invalid mixup_p {mixup_p}"
        assert 0. < mixup_p + cutmix_p <= 1., f"0 < mixup_p + cutmix_p <= 1 (got {mixup_p + cutmix_p})"

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
        self.total_p = mixup_p + cutmix_p
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.seed = seed

    @property
    def fused_operations(self):
        return super().fused_operations + [["x", "class"]]

    def getitem_x(self, idx, ctx=None):
        return self.getitem_xclass(idx, ctx=ctx)[0]

    def getitem_class(self, idx, ctx=None):
        return self.getitem_xclass(idx, ctx=ctx)[1]

    def getitem_xclass(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        cls = self.dataset.getitem_class(idx, ctx=ctx)
        rng = np.random.default_rng(seed=self.seed + idx if self.seed is not None else None)

        # sample what operation to apply (nothing/cutmix/mixup)
        apply = rng.random()
        if apply > self.total_p:
            return x, cls
        use_cutmix = apply < self.cutmix_p

        # load second sample
        idx2 = rng.integers(len(self))
        x2 = self.dataset.getitem_x(idx2, ctx=ctx)
        cls2 = self.dataset.getitem_class(idx2, ctx=ctx)

        # convert cls to onehot
        n_classes = self.getdim_class()
        cls = to_one_hot_vector(cls, n_classes=n_classes)
        cls2 = to_one_hot_vector(cls2, n_classes=n_classes)

        # sample lambda
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lamb = torch.tensor([rng.beta(alpha, alpha)])

        # apply cutmix/mixup
        if use_cutmix:
            # cutmix
            raise NotImplementedError
        else:
            # mixup
            x_lamb = lamb.view(*[1] * x.ndim)
            x.mul_(x_lamb).add_(x2.mul_(1. - x_lamb))
            cls.mul_(lamb).add_(cls2.mul_(1. - lamb))

        return x, cls
