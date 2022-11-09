import numpy as np
import torch

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.functional.mix import sample_permutation, mix_y_y2
from kappadata.functional.onehot import to_onehot_vector


class MixWrapperBase(KDWrapper):
    def __init__(self, *args, p=1., seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(p, (int, float)) and 0. < p <= 1.
        self.p = p
        self.np_rng = np.random.default_rng(seed=seed)
        self.th_rng = torch.Generator()
        if seed is not None:
            self.th_rng.manual_seed(seed + 1)

    @property
    def _ctx_prefix(self):
        raise NotImplementedError

    def _set_noapply_ctx_values(self, ctx):
        raise NotImplementedError

    def _get_params_from_ctx(self, ctx):
        raise NotImplementedError

    def _sample_params(self, idx, x1, ctx):
        raise NotImplementedError

    def _apply(self, x1, x2, params):
        raise NotImplementedError

    def _get_params(self, idx, x1, ctx):
        ctx_apply_key = f"{self._ctx_prefix}_apply"
        if ctx is not None and ctx_apply_key in ctx:
            apply = ctx[ctx_apply_key]
            params = self._get_params_from_ctx(ctx)
            params["lamb"] = ctx[f"{self._ctx_prefix}_lambda"]
            params["idx2"] = ctx[f"{self._ctx_prefix}_idx2"]
        else:
            apply = torch.rand(size=(), generator=self.th_rng) < self.p
            idx2 = sample_permutation(size=1, rng=self.np_rng).item()
            params = self._sample_params(idx=idx, x1=x1, ctx=ctx)
            params["idx2"] = idx2
            if ctx is not None:
                ctx[ctx_apply_key] = apply
                ctx[f"{self._ctx_prefix}_idx2"] = idx2
        return apply, params

    def getitem_x(self, idx, ctx=None):
        x1 = self.dataset.getitem_x(idx, ctx)
        assert torch.is_tensor(x1) and x1.ndim == 3, f"convert image to tensor before {type(self).__name__}"
        apply, params = self._get_params(idx=idx, x1=x1, ctx=ctx)

        if apply:
            x2 = self.dataset.getitem_x(params["idx2"], ctx)
            return self._apply(x1, x2, params)
        else:
            ctx[f"{self._ctx_prefix}_lambda"] = -1
            ctx[f"{self._ctx_prefix}_idx2"] = -1
            self._set_noapply_ctx_values(ctx)
            return x1

    def getitem_class(self, idx, ctx=None):
        y1 = to_onehot_vector(self.dataset.getitem_class(idx, ctx), n_classes=self.dataset.n_classes)
        apply, params = self._get_params(idx=idx, x1=None, ctx=ctx)

        if apply:
            y2 = to_onehot_vector(self.dataset.getitem_class(params["idx2"], ctx), n_classes=self.dataset.n_classes)
            return mix_y_y2(y=y1, y2=y2, lamb=params["lamb"])
        else:
            return y1
