from PIL import ImageFilter
from torchvision.transforms import GaussianBlur

from .base.kd_stochastic_transform import KDStochasticTransform


class KDGaussianBlurPIL(KDStochasticTransform):
    def __init__(self, sigma, **kwargs):
        super().__init__(**kwargs)
        # GaussianBlur preprocesses the parameters -> just use original implementation to store parameters
        # kernel size is not used here as PIL doesn't use a kernel_size
        self.tv_gaussianblur = GaussianBlur(kernel_size=1, sigma=sigma)

    def __call__(self, x, ctx=None):
        sigma = self.get_params()
        # if ctx is not None:
        #     ctx["gaussian_blur"] = dict(sigma=sigma)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

    def get_params(self):
        return self.rng.uniform(self.tv_gaussianblur.sigma[0], self.tv_gaussianblur.sigma[1])
