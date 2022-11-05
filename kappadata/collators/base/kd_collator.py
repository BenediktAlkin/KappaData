from torch.utils.data import default_collate

from .collator_base import CollatorBase


class KDCollator:
    def __init__(self, collators, return_ctx=False):
        assert isinstance(collators, list) and all(isinstance(c, CollatorBase) for c in collators)
        self.collators = collators
        self.return_ctx = return_ctx

    def __call__(self, batch):
        batch = default_collate(batch)
        ctx = {}
        for collator in self.collators:
            batch = collator.collate(batch, ctx)
        if self.return_ctx:
            return batch, ctx
        return batch
