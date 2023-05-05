from .base.kd_random_apply_base import KDRandomApplyBase
from .base.kd_stochastic_transform import KDStochasticTransform
from .base.kd_compose_transform import KDComposeTransform

class KDRandomApply(KDRandomApplyBase):
    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def set_rng(self, rng):
        if isinstance(transform, (KDStochasticTransform, KDComposeTransform)):
            transform.set_rng(rng)
        return super().set_rng(rng)

    def forward(self, x, ctx):
        return self.transform(x, ctx)
