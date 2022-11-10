from torchvision.transforms.functional import hflip

from .base.kd_stochastic_transform import KDStochasticTransform


class KDRandomHorizontalFlip(KDStochasticTransform):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, x, ctx=None):
        apply = self.rng.uniform() < self.p
        if ctx is not None:
            ctx["random_hflip"] = apply
        if apply:
            return hflip(x)
        return x
