import torch

from kappadata.functional.mix import sample_lambda, sample_permutation, mix_y_inplace, mix_y_idx2
from kappadata.functional.cutmix import get_random_bbox, cutmix_sample_inplace
from .base.mix_wrapper_base import MixWrapperBase


class CutmixWrapper(MixWrapperBase):
    def __init__(self, *args, alpha, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

    @property
    def _ctx_prefix(self):
        return "cutmix"

    def _set_noapply_ctx_values(self, ctx):
        ctx["cutmix_bbox"] = (-1, -1, -1, -1)

    def _get_params_from_ctx(self, ctx):
        return dict(bbox=ctx["cutmix_bbox"])

    def _sample_params(self, idx, x1, ctx):
        if x1 is None:
            _, h, w = self.getitem_x(idx, ctx).shape
        else:
            _, h, w = x1.shape
        lamb = sample_lambda(alpha=self.alpha, size=1, rng=self.np_rng)
        bbox, lamb = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.th_rng)
        lamb = lamb.item()
        if ctx is not None:
            ctx["cutmix_lambda"] = lamb
            ctx["cutmix_bbox"] = bbox
        return dict(bbox=bbox, lamb=lamb)

    def _apply(self, x1, x2, params):
        return cutmix_sample_inplace(x1=x1, x2=x2, bbox=params["bbox"])
