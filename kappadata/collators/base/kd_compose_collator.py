from torch.utils.data import default_collate

from .kd_collator import KDCollator


class KDComposeCollator:
    def __init__(self, collators, dataset_mode, return_ctx=False):
        assert isinstance(collators, list) and len(collators) > 0 and all(isinstance(c, KDCollator) for c in collators)
        self.collators = collators
        self.dataset_mode = dataset_mode
        self.return_ctx = return_ctx

        self.default_collate_modes = [c.default_collate_mode for c in collators]
        assert all(dcm in [None, "before", "after"] for dcm in self.default_collate_modes)

    def __call__(self, batch):
        called_default_collate = False
        removed_ctx_from_batch = False
        ctx = {}
        for collator, default_collate_mode in zip(self.collators, self.default_collate_modes):
            if default_collate_mode is None:
                assert not called_default_collate

            if default_collate_mode == "before" and not called_default_collate:
                batch = default_collate(batch)
                if self.return_ctx:
                    batch, ctx = batch
                    assert isinstance(ctx, dict), \
                        "ModeWrapper.return_ctx should be equal to KDComposeCollator.return_ctx"
                called_default_collate = True

            if not called_default_collate and self.return_ctx and not removed_ctx_from_batch:
                # remove ctx from batch
                batch, ctx = zip(*batch)
                ctx = default_collate(ctx)
                assert isinstance(ctx, dict), "ModeWrapper.return_ctx should be equal to KDComposeCollator.return_ctx"
                removed_ctx_from_batch = True

            batch = collator.collate(batch, self.dataset_mode, ctx)

            if default_collate_mode == "after":
                assert not called_default_collate
                batch = default_collate(batch)

        if self.return_ctx:
            return batch, ctx
        return batch
