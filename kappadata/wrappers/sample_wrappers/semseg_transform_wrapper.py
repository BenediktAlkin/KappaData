import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.factory import object_to_transform


class SemsegTransformWrapper(KDWrapper):
    def __init__(self, dataset, shared_transform, x_transform=None, seed=None):
        super().__init__(dataset=dataset)
        self.shared_transform = object_to_transform(shared_transform)
        self.x_transform = object_to_transform(x_transform)
        self.seed = seed

    @property
    def requires_propagate_ctx(self):
        return True

    def _shared_transform(self, item, idx, ctx):
        # generate seed or retrieve seed from ctx
        assert ctx is not None
        if "semseg_seed" not in ctx:
            seed_rng = np.random.default_rng(self.seed)
            seed = seed_rng.integers(np.iinfo(np.int32).max)
            ctx["semseg_seed"] = seed
        else:
            seed = ctx["semseg_seed"]
        # apply shared_transform
        rng = np.random.default_rng(seed=seed + idx)
        assert self.shared_transform.is_kd_transform
        self.shared_transform.set_rng(rng)
        item = self.shared_transform(item, ctx=ctx)
        return item

    def getitem_x(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        x = self._shared_transform(item=x, idx=idx, ctx=ctx)
        # apply x_transform
        if self.x_transform is not None:
            if self.seed is not None:
                x_transform_rng = np.random.default_rng(seed=self.seed + idx)
                self.x_transform.set_rng(x_transform_rng)
            x = self.x_transform(x, ctx=ctx)
        return x

    def getitem_semseg(self, idx, ctx=None):
        semseg = self.dataset.getitem_semseg(idx, ctx=ctx)
        semseg = self._shared_transform(item=semseg, idx=idx, ctx=ctx)
        return semseg
