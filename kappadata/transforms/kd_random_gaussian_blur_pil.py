from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_gaussian_blur_pil import KDGaussianBlurPIL


class KDRandomGaussianBlurPIL(KDRandomApplyBase):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        self.gaussian_blur = KDGaussianBlurPIL(sigma=sigma)

    def forward(self, x, ctx):
        return self.gaussian_blur(x, ctx)
