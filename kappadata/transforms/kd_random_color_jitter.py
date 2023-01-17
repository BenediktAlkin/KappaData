from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_color_jitter import KDColorJitter


class KDRandomColorJitter(KDRandomApplyBase):
    def __init__(self, brightness, contrast, saturation, hue, **kwargs):
        super().__init__(**kwargs)
        seed = self.seed + 1 if self.seed is not None else None
        self.color_jitter = KDColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            seed=seed
        )

    # TODO reset_seed is deprecated
    def reset_seed(self):
        super().reset_seed()
        self.color_jitter.reset_seed()

    def _scale_strength(self, factor):
        self.color_jitter.scale_strength(factor)

    def forward(self, x, ctx):
        return self.color_jitter(x, ctx)
