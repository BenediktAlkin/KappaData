import torch

from kappadata.functional import get_random_bbox, cutmix_batch
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
        self.mode = mode
        self.cutmix_p = cutmix_p
        self._is_batch_mode = self.mode == "batch"

    def _collate(self, x, y, batch_size, ctx):
        if self._is_batch_mode:
            use_cutmix_size = 1
        else:
            use_cutmix_size = batch_size
        apply = torch.rand(size=(batch_size,), generator=self.rng) < self.p
        idx2 = torch.from_numpy(self.np_rng.permutation(batch_size)).type(torch.long)
        use_cutmix = torch.rand(size=(use_cutmix_size,), generator=self.rng) < self.cutmix_p

        if ctx is not None:
            ctx["mix_idx2"] = idx2
            ctx["use_cutmix"] = use_cutmix

        if self._is_batch_mode:
            if use_cutmix:
                cutmix_lamb = torch.from_numpy(self.np_rng.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size))
                cutmix_lamb = cutmix_lamb.type(torch.float32)
                # TODO batch version
                h, w = x.shape[2:]
                bbox = [get_random_bbox(h=h, w=w, lamb=lamb, rng=self.np_rng) for lamb in cutmix_lamb]
                if ctx is not None:
                    ctx["mix_lambda"] = cutmix_lamb
                    ctx["mix_bbox"] = bbox
                x_clone = x.clone()
                mixed_x = cutmix_batch(x1=x_clone, x2=x_clone[idx2], bbox=bbox)
                mixed_y = self._mix_y(y=y, lamb=cutmix_lamb.view(-1, 1), idx2=idx2)
            else:
                mixup_lamb = torch.from_numpy(self.np_rng.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
                mixup_lamb = mixup_lamb.type(torch.float32)
                if ctx is not None:
                    ctx["mix_lambda"] = mixup_lamb
                    ctx["mix_bbox"] = None
                mixup_lamb_x = mixup_lamb.view(-1, *[1] * (x.ndim - 1))
                mixed_x = mixup_lamb_x * x + (1. - mixup_lamb_x) * x[idx2]
                mixed_y = self._mix_y(y=y, lamb=mixup_lamb.view(-1, 1), idx2=idx2)
        else:
            # cutmix params
            cutmix_lamb = torch.from_numpy(self.np_rng.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size))
            cutmix_lamb = cutmix_lamb.type(torch.float32)
            # TODO batch version
            h, w = x.shape[2:]
            bbox = [get_random_bbox(h=h, w=w, lamb=lamb, rng=self.np_rng) for lamb in cutmix_lamb]
            # mixup params
            mixup_lamb = torch.from_numpy(self.np_rng.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            mixup_lamb = mixup_lamb.type(torch.float32)

            if ctx is not None:
                ctx["mix_mixup_lambda"] = mixup_lamb
                ctx["mix_cutmix_lambda"] = cutmix_lamb
                ctx["mix_bbox"] = bbox

            mixup_lamb_x = mixup_lamb.view(-1, *[1] * (x.ndim - 1))
            use_cutmix_x = use_cutmix.view(-1, *[1] * (x.ndim - 1))
            x_clone = x.clone()
            cutmixed_x = cutmix_batch(x1=x_clone, x2=x_clone[idx2], bbox=bbox)
            mixuped_x = mixup_lamb_x * x + (1. - mixup_lamb_x) * x[idx2]
            mixed_x = torch.where(use_cutmix_x, cutmixed_x, mixuped_x)

            cutmixed_y = self._mix_y(y=y, lamb=cutmix_lamb.view(-1, 1), idx2=idx2)
            mixuped_y = self._mix_y(y=y, lamb=mixup_lamb.view(-1, 1), idx2=idx2)
            mixed_y = torch.where(use_cutmix.view(-1, 1), cutmixed_y, mixuped_y)

        # filter by apply
        apply_x = apply.view(-1, *[1] * (x.ndim - 1))
        result_x = torch.where(apply_x, mixed_x, x)
        if y is not None:
            apply_y = apply.view(-1, 1)
            result_y = torch.where(apply_y, mixed_y, y)
        else:
            result_y = None

        if result_y is not None:
            return result_x, result_y
        return result_x
