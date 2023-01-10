import numpy as np
import torch

from kappadata.collators.base.kd_single_collator import KDSingleCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper


class KDMixCollator(KDSingleCollator):
    """
    apply_mode:
    - "batch": apply either all samples in the batch or don't apply
    - "sample": decide for each sample whether or not to apply mixup/cutmix
    lamb_mode:
    - "batch": use the same lambda/bbox for all samples in the batch
    - "sample": sample a lambda/bbox for each sample
    shuffle_mode:
    - "roll": mix sample 0 with sample 1; sample 1 with sample 2; ...
    - "flip": mix sample[0] with sample[-1]; sample[1] with sample[-2]; ... requires even batch_size
    - "random": mix each sample with a randomly drawn other sample
    """

    def __init__(
            self,
            mixup_alpha: float = None,
            cutmix_alpha: float = None,
            mixup_p: float = 0.5,
            cutmix_p: float = 0.5,
            apply_mode: str = "batch",
            lamb_mode: str = "batch",
            shuffle_mode: str = "flip",
            seed: int = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # check probabilities
        assert isinstance(mixup_p, (int, float)) and 0. <= mixup_p <= 1., f"invalid mixup_p {mixup_p}"
        assert isinstance(cutmix_p, (int, float)) and 0. <= cutmix_p <= 1., f"invalid mixup_p {mixup_p}"
        assert 0. < mixup_p + cutmix_p <= 1., f"0 < mixup_p + cutmix_p <= 1 (got {mixup_p + cutmix_p})"

        # check alphas
        if mixup_p == 0.:
            assert mixup_alpha is None
        else:
            assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        if cutmix_p == 0.:
            assert cutmix_alpha is None
        else:
            assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha

        # check modes
        assert apply_mode in ["batch", "sample"], f"invalid apply_mode {apply_mode}"
        assert lamb_mode in ["batch", "sample"], f"invalid lamb_mode {lamb_mode}"
        assert shuffle_mode in ["roll", "flip", "random"], f"invalid shuffle_mode {shuffle_mode}"

        # initialize
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.apply_mode = apply_mode
        self.lamb_mode = lamb_mode
        self.shuffle_mode = shuffle_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def reset_seed(self):
        self.rng = np.random.default_rng(seed=seed)

    @property
    def default_collate_mode(self) -> str:
        return "before"

    @property
    def total_p(self) -> float:
        return self.mixup_p + self.cutmix_p

    def collate(self, batch, dataset_mode, ctx=None):
        # extract properties from batch
        idx, x, y = None, None, None
        if ModeWrapper.has_item(mode=dataset_mode, item="index"):
            idx = ModeWrapper.get_item(mode=dataset_mode, item="index", batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item="x"):
            x = ModeWrapper.get_item(mode=dataset_mode, item="x", batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item="class"):
            y = ModeWrapper.get_item(mode=dataset_mode, item="class", batch=batch).type(torch.float32)
            assert y.ndim == 2, "KDMixCollator expects classes to be in one-hot format"
        batch_size = len(x)

        # sample apply
        if self.apply_mode == "batch":
            apply = torch.full(size=(batch_size,), fill_value=self.rng.random() < self.total_p)
        elif self.apply_mode == "sample":
            apply = torch.from_numpy(self.rng.random(batch_size)) < self.total_p
        else:
            raise NotImplementedError

        # sample parameters (use_cutmix, lamb, bbox)
        if self.lamb_mode == "batch":
            use_cutmix = self.rng.random() * self.total_p < self.cutmix_p
            alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
            lamb = torch.tensor([self.rng.beta(alpha, alpha)])
            if use_cutmix:
                h, w = x.shape[2:]
                bbox, lamb = self.get_random_bbox(h=h, w=w, lamb=lamb)
                bbox = bbox.repeat(batch_size, 1)
            else:
                bbox = torch.full(size=(batch_size, 4), fill_value=-1)
            use_cutmix = torch.tensor([use_cutmix]).repeat(batch_size)
        elif self.lamb_mode == "sample":
            use_cutmix = torch.from_numpy(self.rng.random(batch_size) * self.total_p) < self.cutmix_p
            mixup_lamb, cutmix_lamb = 0., 0.
            if self.mixup_p > 0.:
                mixup_lamb = torch.from_numpy(self.rng.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            if self.cutmix_p > 0.:
                cutmix_lamb = torch.from_numpy(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size))
            lamb = torch.where(use_cutmix, cutmix_lamb, mixup_lamb).float()
            h, w = x.shape[2:]
            bbox, lamb = self.get_random_bbox(h=h, w=w, lamb=lamb)
        else:
            raise NotImplementedError

        # apply
        permutation = None
        if x is not None:
            bool_shape = (-1, *[1] * (x.ndim - 1))
            x_lamb = lamb.view(*bool_shape)
            x2, permutation = self.shuffle(item=x, permutation=permutation)
            mixup_x = x * x_lamb + x2 * (1. - x_lamb)
            cutmix_x = x.clone()
            for i in range(batch_size):
                top, left, bot, right = bbox[i]
                cutmix_x[..., top:bot, left:right] = x2[..., top:bot, left:right]
            mixed_x = torch.where(use_cutmix.view(*bool_shape), cutmix_x, mixup_x)
            x = torch.where(apply.view(*bool_shape), mixed_x, x)
        if y is not None:
            y2, permutation = self.shuffle(item=y, permutation=permutation)
            y_lamb = lamb.view(-1, 1)
            mixed_y = y * y_lamb + y2 * (1. - y_lamb)
            y = torch.where(apply.view(-1, 1), mixed_y, y)

        # book keeping
        if ctx is not None:
            ctx["apply"] = apply
            ctx["use_cutmix"] = use_cutmix
            ctx["lambda"] = lamb
            ctx["bbox"] = bbox

        # update properties in batch
        if idx is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="index", batch=batch, value=idx)
        if x is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="x", batch=batch, value=x)
        if y is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="class", batch=batch, value=y)
        return batch

    def get_random_bbox(self, h, w, lamb):
        n_bboxes = len(lamb)
        bbox_hcenter = torch.from_numpy(self.rng.integers(h, size=(n_bboxes,)))
        bbox_wcenter = torch.from_numpy(self.rng.integers(w, size=(n_bboxes,)))

        area_half = 0.5 * (1.0 - lamb).sqrt()
        bbox_h_half = (area_half * h).floor()
        bbox_w_half = (area_half * w).floor()

        top = torch.clamp(bbox_hcenter - bbox_h_half, min=0).type(torch.long)
        bot = torch.clamp(bbox_hcenter + bbox_h_half, max=h).type(torch.long)
        left = torch.clamp(bbox_wcenter - bbox_w_half, min=0).type(torch.long)
        right = torch.clamp(bbox_wcenter + bbox_w_half, max=w).type(torch.long)
        bbox = torch.stack([top, left, bot, right], dim=1)

        lamb_adjusted = 1.0 - (bot - top) * (right - left) / (h * w)

        return bbox, lamb_adjusted

    def shuffle(self, item, permutation):
        if self.shuffle_mode == "roll":
            return item.roll(shifts=1, dims=0), None
        if self.shuffle_mode == "flip":
            assert len(item) % 2 == 0
            return item.flip(0), None
        if self.shuffle_mode == "random":
            if permutation is None:
                permutation = self.rng.permutation(len(item))
            return item.clone()[permutation], permutation
        raise NotImplementedError
