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

    def getitem_x(self, idx, ctx=None):
        raise NotImplementedError

    def getitem_semseg(self, idx, ctx=None):
        raise NotImplementedError

    def getitem_xsemseg(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        semseg = self.dataset.getitem_semseg(idx, ctx=ctx)



        return x, semseg