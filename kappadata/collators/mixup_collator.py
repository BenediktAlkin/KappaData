import torch

from kappadata.functional.mix import sample_lambda, sample_permutation, mix_y_inplace, mix_y_idx2
from kappadata.functional.mixup import mixup_roll, mixup_idx2
from .base.mix_collator_base import MixCollatorBase


class MixupCollator(MixCollatorBase):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

    def _collate_batchwise(self, x, y, batch_size, ctx):
        # get parameters
        lamb = sample_lambda(alpha=self.alpha, size=batch_size, rng=self.np_rng)

        # store parameters
        if ctx is not None:
            ctx["mixup_lambda"] = lamb

        # mixup
        x = mixup_roll(x, lamb.view(-1, *[1] * (x.ndim - 1)))
        y = mix_y_inplace(y, lamb.view(-1, 1))
        return x, y

    def _collate_samplewise(self, apply, x, y, batch_size, ctx):
        # get parameters
        lamb = sample_lambda(alpha=self.alpha, size=batch_size, rng=self.np_rng)
        idx2 = sample_permutation(batch_size=batch_size, rng=self.np_rng)

        # store parameters
        if ctx is not None:
            ctx["mixup_lambda"] = lamb
            ctx["mixup_idx2"] = idx2

        # add dimensions for broadcasting
        lamb_x = lamb.view(-1, *[1] * (x.ndim - 1))
        apply_x = apply.view(-1, *[1] * (x.ndim - 1))

        # mixup x
        mixed_x = mixup_idx2(x=x, idx2=idx2, lamb=lamb_x)
        result_x = torch.where(apply_x, mixed_x, x)
        if y is None:
            return result_x

        # mixup y
        mixed_y = mix_y_idx2(y=y, idx2=idx2, lamb=lamb)
        result_y = torch.where(apply.view(-1, 1), mixed_y, y)
        return result_x, result_y
