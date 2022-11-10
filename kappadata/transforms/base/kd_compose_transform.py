from .kd_transform import KDTransform
from .kd_stochastic_transform import KDStochasticTransform

class KDComposeTransform(KDTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def reset_seed(self):
        for t in self.transforms:
            if isinstance(t, KDStochasticTransform):
                t.reset_seed()

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            if isinstance(t, KDTransform):
                x = t(x, ctx)
            else:
                x = t(x)
        return x
