import torch
from PIL import ImageFilter
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import to_pil_image

from .base.kd_stochastic_transform import KDStochasticTransform


class KDGaussianBlurPIL(KDStochasticTransform):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        # GaussianBlur preprocesses the parameters
        # kernel size is not used here as PIL doesn't use a kernel_size
        tv_gaussianblur = GaussianBlur(kernel_size=1, sigma=sigma)
        self.sigma_lb = tv_gaussianblur.sigma[0]
        self.sigma_ub = self.og_sigma_ub = tv_gaussianblur.sigma[1]
        self.ctx_key = f"{self.ctx_prefix}.sigma"

    def _scale_strength(self, factor):
        self.sigma_ub = self.sigma_lb + (self.og_sigma_ub - self.sigma_lb) * factor

    def __call__(self, x, ctx=None):
        if torch.is_tensor(x):
            x = to_pil_image(x)
        sigma = self.get_params()
        if ctx is not None:
            ctx[self.ctx_key] = sigma
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

    def get_params(self):
        return self.rng.uniform(self.sigma_lb, self.sigma_ub)
