import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.factory import object_to_transform


class SemsegTransformWrapper(KDWrapper):
    def __init__(
            self,
            dataset,
            x_pre_transform=None,
            semseg_pre_transform=None,
            shared_transform=None,
            x_post_transform=None,
            semseg_post_transform=None,
            seed=None,
    ):
        super().__init__(dataset=dataset)
        self.x_pre_transform = object_to_transform(x_pre_transform)
        self.semseg_pre_transform = object_to_transform(semseg_pre_transform)
        self.shared_transform = object_to_transform(shared_transform)
        self.x_post_transform = object_to_transform(x_post_transform)
        self.semseg_post_transform = object_to_transform(semseg_post_transform)
        self.seed = seed

    @property
    def requires_propagate_ctx(self):
        return True

    def _shared_transform(self, item, pre_transform, idx, ctx):
        # generate seed or retrieve seed from ctx
        assert ctx is not None
        if "semseg_seed" not in ctx:
            seed_rng = np.random.default_rng(self.seed)
            seed = seed_rng.integers(np.iinfo(np.int32).max)
            ctx["semseg_seed"] = seed
        else:
            seed = ctx["semseg_seed"]
        rng = np.random.default_rng(seed=seed + idx)
        # apply pre_transform (shared rng)
        if pre_transform is not None:
            assert pre_transform.is_kd_transform
            pre_transform.set_rng(rng)
            item = pre_transform(item, ctx=ctx)
        # apply shared_transform (shared rng)
        if self.shared_transform is not None:
            assert self.shared_transform.is_kd_transform
            self.shared_transform.set_rng(rng)
            item = self.shared_transform(item, ctx=ctx)
        return item

    def getitem_x(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        x = self._shared_transform(item=x, pre_transform=self.x_pre_transform, idx=idx, ctx=ctx)
        # apply x_post_transform (independent rng)
        if self.x_post_transform is not None:
            if self.seed is not None:
                x_post_transform_rng = np.random.default_rng(seed=self.seed + idx)
                self.x_post_transform.set_rng(x_post_transform_rng)
            x = self.x_post_transform(x, ctx=ctx)
        return x

    def getitem_semseg(self, idx, ctx=None):
        semseg = self.dataset.getitem_semseg(idx, ctx=ctx)
        semseg = self._shared_transform(item=semseg, pre_transform=self.semseg_pre_transform, idx=idx, ctx=ctx)
        # apply semseg_post_transform (independent rng)
        if self.semseg_post_transform is not None:
            if self.seed is not None:
                semseg_post_transform_rng = np.random.default_rng(seed=self.seed + idx)
                self.semseg_post_transform.set_rng(semseg_post_transform_rng)
            semseg = self.semseg_post_transform(semseg, ctx=ctx)
        return semseg
