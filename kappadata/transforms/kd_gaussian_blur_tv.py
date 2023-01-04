import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur

from .base.kd_stochastic_transform import KDStochasticTransform


class KDGaussianBlurTV(KDStochasticTransform):
    def __init__(self, kernel_size, sigma, **kwargs):
        super().__init__(**kwargs)
        # GaussianBlur preprocesses the parameters -> just use original implementation to store parameters
        self.tv_gaussianblur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, x, ctx=None):
        sigma = self.get_params()
        # if ctx is not None:
        #     ctx["gaussian_blur"] = dict(sigma=sigma)
        return F.gaussian_blur(x, self.tv_gaussianblur.kernel_size, [sigma, sigma])

    def get_params(self):
        return self.rng.uniform(self.tv_gaussianblur.sigma[0], self.tv_gaussianblur.sigma[1])
