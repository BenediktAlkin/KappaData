from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_gaussian_blur_tv import KDGaussianBlurTV


class KDRandomGaussianBlurTV(KDRandomApplyBase):
    def __init__(self, kernel_size, sigma, **kwargs):
        super().__init__(**kwargs)
        seed = self.seed + 1 if self.seed is not None else None
        self.gaussian_blur = KDGaussianBlurTV(kernel_size=kernel_size, sigma=sigma, seed=seed)

    def reset_seed(self):
        super().reset_seed()
        self.gaussian_blur.reset_seed()

    def forward(self, x, ctx):
        return self.gaussian_blur(x, ctx)
