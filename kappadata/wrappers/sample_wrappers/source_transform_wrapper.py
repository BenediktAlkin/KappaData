import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.factory import object_to_transform
from kappadata.transforms import KDComposeTransform, KDStochasticTransform, KDTransform
from .base.transform_wrapper_base import TransformWrapperBase

class SourceTransformWrapper(TransformWrapperBase):
    def getitem_source(self, idx, ctx=None):
        item = self.dataset.getitem_source(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)
