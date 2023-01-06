import numpy as np
import torch

from kappadata.collators.base.kd_single_collator import KDSingleCollator
from kappadata.functional.onehot import to_onehot_matrix
from kappadata.wrappers.mode_wrapper import ModeWrapper


class MixCollatorBase(KDSingleCollator):
    def __init__(self, p=1., p_mode="batch", n_classes=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(p, (int, float)) and 0. < p <= 1., f"invalid p {p}"
        assert p_mode in ["batch", "sample"], f"invalid p_mode {p_mode}"
        assert n_classes is None or n_classes > 1, f"invalid n_classes {n_classes}"
        self.p = p
        self.p_mode = p_mode
        self.n_classes = n_classes
        self.np_rng = np.random.default_rng(seed=seed)

    @property
    def default_collate_mode(self):
        return "before"

    def collate(self, batch, dataset_mode, ctx=None):
        idx, x, y = None, None, None
        if ModeWrapper.has_item(mode=dataset_mode, item="index"):
            idx = ModeWrapper.get_item(mode=dataset_mode, item="index", batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item="x"):
            x = ModeWrapper.get_item(mode=dataset_mode, item="x", batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item="class"):
            y = ModeWrapper.get_item(mode=dataset_mode, item="class", batch=batch)
            y = to_onehot_matrix(y, n_classes=self.n_classes).type(torch.float32)
        batch_size = len(x)
        if self.p_mode == "batch":
            apply = self.np_rng.random() < self.p
            if apply:
                x, y = self._collate_batchwise(x, y, batch_size, ctx)
        elif self.p_mode == "sample":
            apply = torch.from_numpy(self.np_rng.random(batch_size)) < self.p
            x, y = self._collate_samplewise(apply=apply, x=x, y=y, batch_size=batch_size, ctx=ctx)
        else:
            raise NotImplementedError

        if idx is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="index", batch=batch, value=idx)
        if x is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="x", batch=batch, value=x)
        if y is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="class", batch=batch, value=y)
        return batch

    def _collate_batchwise(self, x, y, batch_size, ctx):
        raise NotImplementedError

    def _collate_samplewise(self, apply, x, y, batch_size, ctx):
        raise NotImplementedError
