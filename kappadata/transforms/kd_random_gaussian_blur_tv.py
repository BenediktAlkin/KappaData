from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_gaussian_blur_tv import KDGaussianBlurTV


class KDRandomGaussianBlurTV(KDRandomApplyBase):
    def __init__(self, kernel_size, sigma, **kwargs):
        super().__init__(**kwargs)
        self.gaussian_blur = KDGaussianBlurTV(kernel_size=kernel_size, sigma=sigma)

    def forward(self, x, ctx):
        return self.gaussian_blur(x, ctx)
