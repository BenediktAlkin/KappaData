import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.factory import object_to_transform
from kappadata.transforms import KDComposeTransform, KDStochasticTransform, KDTransform


class TargetTransformWrapper(KDWrapper):
    def getitem_target(self, idx, ctx=None):
        item = self.dataset.getitem_target(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)

