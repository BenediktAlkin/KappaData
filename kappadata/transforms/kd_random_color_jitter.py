from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_color_jitter import KDColorJitter


class KDRandomColorJitter(KDRandomApplyBase):
    def __init__(self, brightness, contrast, saturation, hue, **kwargs):
        super().__init__(**kwargs)
        self.color_jitter = KDColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, x, ctx):
        return self.color_jitter(x, ctx)
