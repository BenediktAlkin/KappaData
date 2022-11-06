from torch.utils.data import default_collate

from .kd_collator import KDCollator


class ComposeCollator:
    def __init__(self, collators, return_ctx=False):
        assert isinstance(collators, list) and all(isinstance(c, KDCollator) for c in collators)
        self.collators = collators
        self.return_ctx = return_ctx

        self.default_collate_modes = [c.default_collate_mode for c in collators]
        assert all(dcm in [None, "before", "after"] for dcm in self.default_collate_modes)

    def __call__(self, batch):
        called_default_collate = False
        ctx = {}
        for collator, default_collate_mode in zip(self.collators, self.default_collate_modes):
            if default_collate_mode is None:
                assert not called_default_collate

            if default_collate_mode == "before" and not called_default_collate:
                batch = default_collate(batch)
                called_default_collate = True
            
            batch = collator.collate(batch, ctx)

            if default_collate_mode == "after":
                assert not called_default_collate
                batch = default_collate(batch)

        if self.return_ctx:
            return batch, ctx
        return batch
