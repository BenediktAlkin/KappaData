from .base.kd_random_apply_base import KDRandomApplyBase


class KDRandomApply(KDRandomApplyBase):
    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def forward(self, x, ctx):
        return self.transform(x, ctx)
