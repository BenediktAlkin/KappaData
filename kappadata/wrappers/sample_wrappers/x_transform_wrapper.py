from kappadata.datasets.kd_wrapper import KDWrapper


class XTransformWrapper(KDWrapper):
    def __init__(self, *args, x_transform, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_transform = x_transform

    def getitem_x(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx)
        transformed = self.x_transform(x)
        return transformed
