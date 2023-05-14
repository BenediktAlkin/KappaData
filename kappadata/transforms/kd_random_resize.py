import math

import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, get_image_size

from kappadata.utils.param_checking import to_2tuple
from .base.kd_stochastic_transform import KDStochasticTransform


class KDRandomResize(KDStochasticTransform):
    """ resize image to base_size * ratio where ratio is randomly sampled """
    def __init__(
            self,
            base_size,
            ratio,
            interpolation="bilinear",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_size = to_2tuple(base_size)
        self.ratio = to_2tuple(ratio)
        self.interpolation = InterpolationMode(interpolation)

    def __call__(self, x, ctx=None):
        new_size = self.get_params()
        return resize(x, new_size, self.interpolation)

    def get_params(self):
        ratio = self.rng.uniform(self.ratio[0], self.ratio[1])
        height, width = self.base_size
        new_height = height * ratio
        new_width = width * ratio
        return new_height, new_width
