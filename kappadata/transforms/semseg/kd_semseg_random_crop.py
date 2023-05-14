import math

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, get_image_size

from kappadata.utils.param_checking import to_2tuple
from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDSemsegRandomCrop(KDStochasticTransform):
    def __init__(self, size, max_category_ratio, **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.max_category_ratio = max_category_ratio

    def __call__(self, xsemseg, ctx=None):
        x, semseg = xsemseg
        width, height = get_image_size(img)
        num_pixels = width * height
        for _ in range(10):
            i, j = self.get_params(height=height, width=width)
            ii = i + self.size[0]
            jj = j + self.size[1]
            crop = semseg[i:ii, j:jj]
            _, counts = crop.unique(return_counts=True)
            if counts / num_pixels <= self.max_category_ratio:
                break
        return x, semseg

    def get_params(self, height, width):
        i = self.rng.integers(height - self.size[0] + 1, size=(1,))
        j = self.rng.integers(width - self.size[1] + 1, size=(1,))
        return i, j
