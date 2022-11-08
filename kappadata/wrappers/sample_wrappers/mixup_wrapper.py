from .base.mix_wrapper_base import MixWrapperBase

from kappadata.functional.mix import sample_lambda, sample_permutation, mix_y_inplace, mix_y_idx2
from kappadata.functional.mixup import mixup_inplace

class MixupWrapper(MixWrapperBase):
    def __init__(self, *args, alpha, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

    @property
    def _ctx_prefix(self):
        return "mixup"

    def _set_noapply_ctx_values(self, ctx):
        pass

    def _get_params_from_ctx(self, ctx):
        return dict(lamb=ctx["mixup_lambda"])

    def _sample_params(self, idx, x1, ctx):
        lamb = sample_lambda(alpha=self.alpha, size=1, rng=self.np_rng).item()
        if ctx is not None:
            ctx["mixup_lambda"] = lamb
        return dict(lamb=lamb)

    def _apply(self, x1, x2, params):
        return mixup_inplace(x1=x1, x2=x2, lamb=params["mixup_lambda"])
