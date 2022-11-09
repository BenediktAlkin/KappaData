import torch

from kappadata.functional.cutmix import cutmix_batch, get_random_bbox
from kappadata.functional.mix import sample_lambda, sample_permutation, mix_y_inplace, mix_y_idx2
from kappadata.functional.mixup import mixup_roll, mixup_idx2
from .base.mix_collator_base import MixCollatorBase


class MixCollator(MixCollatorBase):
    def __init__(self, cutmix_alpha, mixup_alpha, mode, cutmix_p=.5, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha
        assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        assert isinstance(cutmix_p, (int, float)) and 0. < cutmix_p < 1.
        assert mode in ["sample", "batch"]
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.cutmix_p = cutmix_p
        self._is_batch_mode = mode == "batch"

    def _collate_batchwise(self, x, y, batch_size, ctx):
        use_cutmix_size = 1 if self._is_batch_mode else batch_size
        use_cutmix = torch.rand(size=(use_cutmix_size,), generator=self.th_rng) < self.cutmix_p

        if ctx is not None:
            ctx["use_cutmix"] = use_cutmix

        if self._is_batch_mode:
            if use_cutmix:
                # apply cutmix to whole batch
                lamb = sample_lambda(alpha=self.cutmix_alpha, size=batch_size, rng=self.np_rng)
                h, w = x.shape[2:]
                bbox, lamb = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.th_rng)
                if ctx is not None:
                    ctx["mix_lambda"] = lamb
                    ctx["mix_bbox"] = bbox
                mixed_x = cutmix_batch(x1=x, x2=x.roll(1, 0), bbox=bbox, inplace=True)
            else:
                # apply mixup to whole batch
                lamb = sample_lambda(alpha=self.mixup_alpha, size=batch_size, rng=self.np_rng)
                if ctx is not None:
                    ctx["mix_lambda"] = lamb
                    ctx["mix_bbox"] = None
                mixed_x = mixup_roll(x, lamb.view(-1, *[1] * (x.ndim - 1)))
            mixed_y = mix_y_inplace(y=y, lamb=lamb.view(-1, 1)) if y is not None else None
        else:
            # decide per sample wheter or not to apply cutmix or mixup
            # cutmix params
            cutmix_lamb = sample_lambda(alpha=self.cutmix_alpha, size=batch_size, rng=self.np_rng)
            h, w = x.shape[2:]
            bbox, cutmix_lamb = get_random_bbox(h=h, w=w, lamb=cutmix_lamb, rng=self.th_rng)
            # mixup params
            mixup_lamb = sample_lambda(alpha=self.mixup_alpha, size=batch_size, rng=self.np_rng)

            if ctx is not None:
                ctx["mix_mixup_lambda"] = mixup_lamb
                ctx["mix_cutmix_lambda"] = cutmix_lamb
                ctx["mix_bbox"] = bbox

            cutmixed_x = cutmix_batch(x1=x, x2=x.roll(1, 0), bbox=bbox, inplace=False)
            mixuped_x = mixup_roll(x=x, lamb=mixup_lamb.view(-1, *[1] * (x.ndim - 1)))
            mixed_x = torch.where(use_cutmix.view(-1, *[1] * (x.ndim - 1)), cutmixed_x, mixuped_x)

            if y is not None:
                cutmixed_y = mix_y_inplace(y=y, lamb=cutmix_lamb.view(-1, 1))
                mixuped_y = mix_y_inplace(y=y, lamb=mixup_lamb.view(-1, 1))
                mixed_y = torch.where(use_cutmix.view(-1, 1), cutmixed_y, mixuped_y)
            else:
                mixed_y = None
        if y is None:
            return mixed_x
        return mixed_x, mixed_y

    def _collate_samplewise(self, apply, x, y, batch_size, ctx):
        use_cutmix_size = 1 if self._is_batch_mode else batch_size
        idx2 = sample_permutation(size=batch_size, rng=self.np_rng)
        use_cutmix = torch.rand(size=(use_cutmix_size,), generator=self.th_rng) < self.cutmix_p

        if ctx is not None:
            ctx["mix_idx2"] = idx2
            ctx["use_cutmix"] = use_cutmix

        if self._is_batch_mode:
            if use_cutmix:
                # apply cutmix to whole batch
                lamb = sample_lambda(alpha=self.cutmix_alpha, size=batch_size, rng=self.np_rng)
                h, w = x.shape[2:]
                bbox, lamb = get_random_bbox(h=h, w=w, lamb=lamb, rng=self.th_rng)
                if ctx is not None:
                    ctx["mix_lambda"] = lamb
                    ctx["mix_bbox"] = bbox
                mixed_x = cutmix_batch(x1=x_clone, x2=x_clone[idx2], bbox=bbox, inplace=False)
            else:
                # apply mixup to whole batch
                lamb = sample_lambda(alpha=self.mixup_alpha, size=batch_size, rng=self.np_rng)
                if ctx is not None:
                    ctx["mix_lambda"] = lamb
                    ctx["mix_bbox"] = None
                mixed_x = mixup_idx2(x=x, idx2=idx2, lamb=lamb.view(-1, *[1] * (x.ndim - 1)))
            mixed_y = mix_y_idx2(y=y, lamb=lamb.view(-1, 1), idx2=idx2) if y is not None else None
        else:
            # decide per sample wheter or not to apply cutmix or mixup
            # cutmix params
            cutmix_lamb = sample_lambda(alpha=self.cutmix_alpha, size=batch_size, rng=self.np_rng)
            h, w = x.shape[2:]
            bbox, cutmix_lamb = get_random_bbox(h=h, w=w, lamb=cutmix_lamb, rng=self.th_rng)
            # mixup params
            mixup_lamb = sample_lambda(alpha=self.mixup_alpha, size=batch_size, rng=self.np_rng)

            if ctx is not None:
                ctx["mix_mixup_lambda"] = mixup_lamb
                ctx["mix_cutmix_lambda"] = cutmix_lamb
                ctx["mix_bbox"] = bbox

            cutmixed_x = cutmix_batch(x1=x, x2=x[idx2], bbox=bbox, inplace=False)
            mixuped_x = mixup_idx2(x=x, idx2=idx2, lamb=mixup_lamb.view(-1, *[1] * (x.ndim - 1)))
            mixed_x = torch.where(use_cutmix.view(-1, *[1] * (x.ndim - 1)), cutmixed_x, mixuped_x)

            if y is not None:
                cutmixed_y = mix_y_idx2(y=y, idx2=idx2, lamb=cutmix_lamb.view(-1, 1))
                mixuped_y = mix_y_idx2(y=y, idx2=idx2, lamb=mixup_lamb.view(-1, 1))
                mixed_y = torch.where(use_cutmix.view(-1, 1), cutmixed_y, mixuped_y)
            else:
                mixed_y = None

        # filter by apply
        result_x = torch.where(apply.view(-1, *[1] * (x.ndim - 1)), mixed_x, x)
        if y is None:
            return result_x
        result_y = torch.where(apply.view(-1, 1), mixed_y, y)
        return result_x, result_y
