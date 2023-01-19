import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur

from .base.kd_stochastic_transform import KDStochasticTransform


class KDGaussianBlurTV(KDStochasticTransform):
    def __init__(self, kernel_size, sigma, **kwargs):
        super().__init__(**kwargs)
        # GaussianBlur preprocesses the parameters
        # kernel size is not used here as PIL doesn't use a kernel_size
        tv_gaussianblur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.kernel_size = tv_gaussianblur.kernel_size
        self.sigma_lb = tv_gaussianblur.sigma[0]
        self.sigma_ub = self.og_sigma_ub = tv_gaussianblur.sigma[1]
        self.ctx_key = f"{self.ctx_prefix}.sigma"

    def _scale_strength(self, factor):
        self.sigma_ub = self.sigma_lb + (self.og_sigma_ub - self.sigma_lb) * factor

    def __call__(self, x, ctx=None):
        sigma = self.get_params()
        if ctx is not None:
            ctx[self.ctx_key] = sigma
        return F.gaussian_blur(x, self.kernel_size, [sigma, sigma])

    def get_params(self):
        return self.rng.uniform(self.sigma_lb, self.sigma_ub)
