import torch

from .base.mix_collator_base import MixCollatorBase


class MixupCollator(MixCollatorBase):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

    def _collate(self, x, y, batch_size, ctx):
        apply = torch.rand(size=(batch_size,), generator=self.rng) < self.p
        lamb = torch.from_numpy(self.np_rng.beta(self.alpha, self.alpha, size=batch_size)).type(torch.float32)
        idx2 = torch.from_numpy(self.np_rng.permutation(batch_size)).type(torch.long)

        if ctx is not None:
            ctx["mixup_lambda"] = lamb
            ctx["mixup_idx2"] = idx2

        # add dimensions for broadcasting
        apply_x = apply.view(-1, *[1] * (x.ndim - 1))
        lamb_x = lamb.view(-1, *[1] * (x.ndim - 1))

        mixed_x = lamb_x * x + (1. - lamb_x) * x[idx2]
        result_x = torch.where(apply_x, mixed_x, x)
        mixed_y = self._mix_y(y=y, lamb=lamb, idx2=idx2)
        result_y = torch.where(apply.view(-1, 1), mixed_y, y)
        if result_y is not None:
            return result_x, result_y
        return result_x
