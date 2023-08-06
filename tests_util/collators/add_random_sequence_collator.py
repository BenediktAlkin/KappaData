import torch

from kappadata.collators.base.kd_single_collator import KDSingleCollator


class AddRandomSequenceCollator(KDSingleCollator):
    @property
    def default_collate_mode(self):
        return "before"

    def collate(self, batch, dataset_mode, ctx=None):
        assert dataset_mode == "x"
        assert batch.ndim == 1
        values = torch.tensor([self.rng.random() for _ in range(len(batch))])
        batch = torch.concat([batch, values])
        return batch
