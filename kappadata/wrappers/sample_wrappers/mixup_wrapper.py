from .base.mix_wrapper_base import MixWrapperBase


class MixupWrapper(MixWrapperBase):
    def __init__(self, *args, alpha, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

    def _get_params(self, ctx):
        if ctx is None or "mixup_apply" not in ctx:
            apply = self.rng.random() < self.p
            if ctx is not None:
                ctx["mixup_apply"] = apply
            if apply:
                lamb = self.rng.beta(self.alpha, self.alpha)
                idx2 = self.rng.integers(0, len(self))
                if ctx is not None:
                    ctx["mixup_lambda"] = lamb
                    ctx["mixup_idx2"] = idx2
            else:
                ctx["mixup_lambda"] = -1
                ctx["mixup_idx2"] = -1
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
        y1 = self._getitem_class(idx, ctx)
        if not apply:
            return y1
        y2 = self._getitem_class(idx2, ctx)
        return lamb * y1 + (1. - lamb) * y2
