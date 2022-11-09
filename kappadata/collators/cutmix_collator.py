import torch

from kappadata.functional.cutmix import cutmix_batch, get_random_bbox
from kappadata.functional.mix import sample_lambda, sample_permutation, mix_y_inplace, mix_y_idx2
from .base.mix_collator_base import MixCollatorBase


class CutmixCollator(MixCollatorBase):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(alpha, (int, float)) and 0. < alpha
        self.alpha = alpha

    def _collate_batchwise(self, x, y, batch_size, ctx):
        # get parameters
        lamb = sample_lambda(alpha=self.alpha, size=batch_size, rng=self.np_rng)
        h, w = x.shape[2:]
        bbox, lamb = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.th_rng)

        # store parameters
        if ctx is not None:
            ctx["mixup_lambda"] = lamb
            ctx["mixup_bbox"] = bbox

        # cutmix
        cutmix_batch(x1=x, x2=x.roll(1, 0), bbox=bbox, inplace=True)
        y = mix_y_inplace(y, lamb.view(-1, 1))
        return x, y

    def _collate_samplewise(self, apply, x, y, batch_size, ctx):
        # get parameters
        lamb = sample_lambda(alpha=self.alpha, size=1, rng=self.np_rng)
        idx2 = sample_permutation(size=batch_size, rng=self.np_rng)
        h, w = x.shape[2:]
        bbox, lamb = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.th_rng)

        # store parameters
        if ctx is not None:
            ctx["cutmix_lambda"] = lamb
            ctx["cutmix_idx2"] = idx2
            ctx["cutmix_bbox"] = bbox

        # cutmix
        mixed_x = cutmix_batch(x1=x, x2=x[idx2], bbox=bbox, inplace=False)
        result_x = torch.where(apply.view(-1, *[1] * (x.ndim - 1)), mixed_x, x)
        if y is None:
            return result_x
        mixed_y = mix_y_idx2(y=y, idx2=idx2, lamb=lamb)
        result_y = torch.where(apply.view(-1, 1), mixed_y, y)
        return result_x, result_y
