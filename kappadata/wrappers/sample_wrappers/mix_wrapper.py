import torch

from kappadata.functional.mix import sample_lambda, sample_permutation, mix_y_inplace, mix_y_idx2
from kappadata.functional.cutmix import get_random_bbox, cutmix_sample_inplace
from kappadata.functional.mixup import mixup_inplace
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

    @property
    def _ctx_prefix(self):
        return "mix"

    def _set_noapply_ctx_values(self, ctx):
        ctx["mix_usecutmix"] = -1
        ctx["mix_bbox"] = (-1, -1, -1, -1)

    def _get_params_from_ctx(self, ctx):
        return dict(use_cutmix=ctx["mix_usecutmix"], bbox=ctx["mix_bbox"])

    def _sample_params(self, idx, x1, ctx):
        use_cutmix = torch.rand(size=(1,), generator=self.th_rng) < self.cutmix_p
        if use_cutmix:
            if x1 is None:
                _, h, w = self.getitem_x(idx, ctx).shape
            else:
                _, h, w = x1.shape
            lamb = sample_lambda(alpha=self.cutmix_alpha, size=1, rng=self.np_rng).item()
            bbox, lamb = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.th_rng)
            lamb = lamb.item()
            if ctx is not None:
                ctx["cutmix_lambda"] = lamb
                ctx["cutmix_bbox"] = bbox
            return dict(use_cutmix=use_cutmix, bbox=bbox, lamb=lamb)
        else:
            lamb = sample_lambda(alpha=self.mixup_alpha, size=1, rng=self.np_rng).item()
            return dict(use_cutmix=use_cutmix, lamb=lamb)

    def _apply(self, x1, x2, params):
        if params["use_cutmix"]:
            return cutmix_sample_inplace(x1=x1, x2=x2, bbox=params["bbox"])
        else:
            return mixup_inplace(x1=x1, x2=x2, lamb=params["lamb"])


