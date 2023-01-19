from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_gaussian_blur_pil import KDGaussianBlurPIL


class KDRandomGaussianBlurPIL(KDRandomApplyBase):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        seed = self.seed + 1 if self.seed is not None else None
        self.gaussian_blur = KDGaussianBlurPIL(sigma=sigma, seed=seed, ctx_prefix=self.ctx_prefix)

    # TODO reset_seed is deprecated
    def reset_seed(self):
        super().reset_seed()
        self.gaussian_blur.reset_seed()

    def _populate_ctx_on_skip(self, ctx):
        ctx[self.gaussian_blur.ctx_key] = -1

    def _scale_strength(self, factor):
        self.gaussian_blur.scale_strength(factor)

    def forward(self, x, ctx):
        return self.gaussian_blur(x, ctx)
