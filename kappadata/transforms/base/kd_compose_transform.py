from kappadata.factory import object_to_transform
from .kd_stochastic_transform import KDStochasticTransform
from .kd_transform import KDTransform


class KDComposeTransform(KDTransform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = [object_to_transform(transform) for transform in transforms]

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            if isinstance(t, KDTransform):
                x = t(x, ctx)
            else:
                x = t(x)
        return x

    def set_rng(self, rng):
        for t in self.transforms:
            if isinstance(t, KDStochasticTransform):
                t.set_rng(rng)
        return self

    def _scale_strength(self, factor):
        for t in self.transforms:
            if isinstance(t, KDTransform):
                t.scale_strength(factor)
