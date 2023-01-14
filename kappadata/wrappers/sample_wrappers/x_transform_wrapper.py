from kappadata.datasets.kd_wrapper import KDWrapper

class XTransformWrapper(KDWrapper):
    def __init__(self, *args, transform, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def getitem_x(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        return self.transform(x)