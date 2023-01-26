from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_gaussian_blur_pil import KDGaussianBlurPIL


class KDRandomGaussianBlurPIL(KDRandomApplyBase):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self.gaussian_blur = KDGaussianBlurPIL(sigma=sigma, ctx_prefix=self.ctx_prefix)

    def set_rng(self, rng):
        self.gaussian_blur.set_rng(rng)
        return super().set_rng(rng)

    def _populate_ctx_on_skip(self, ctx):
        ctx[self.gaussian_blur.ctx_key] = -1.

    def _scale_strength(self, factor):
        self.gaussian_blur.scale_strength(factor)

    def forward(self, x, ctx):
        return self.gaussian_blur(x, ctx)
