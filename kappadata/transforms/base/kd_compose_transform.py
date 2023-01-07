from .kd_stochastic_transform import KDStochasticTransform
from .kd_transform import KDTransform


class KDComposeTransform(KDTransform):
    def __init__(self, transforms, allow_same_seed=False):
        self.transforms = transforms
        if not allow_same_seed:
            seeds = [
                t.seed
                for t in transforms
                if isinstance(t, KDStochasticTransform) and t.seed is not None
            ]
            assert len(seeds) == len(set(seeds)), \
                f"transforms of type KDStochasticTransform should use different seeds (found seeds {seeds})"

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
