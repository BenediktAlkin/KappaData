from .base.transform_wrapper_base import TransformWrapperBase


class XTransformWrapper(TransformWrapperBase):
    def getitem_x(self, idx, ctx=None):
        item = self.dataset.getitem_x(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)
