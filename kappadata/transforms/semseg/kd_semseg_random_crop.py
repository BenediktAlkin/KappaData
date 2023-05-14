import math

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, get_image_size

from kappadata.utils.param_checking import to_2tuple
from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDSemsegRandomCrop(KDStochasticTransform):
    def __init__(self, size, max_category_ratio=1., ignore_index=-1, **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.max_category_ratio = max_category_ratio
        self.ignore_index = ignore_index

    def __call__(self, xsemseg, ctx=None):
        x, semseg = xsemseg
        width, height = get_image_size(x)

        top, left, bot, right = self.get_params(height=height, width=width)
        crop = semseg[top:bot, left:right]
        if self.max_category_ratio < 1.:
            for _ in range(10):
                labels, counts = crop.unique(return_counts=True)
                counts = counts[labels != self.ignore_index]
                if len(counts) > 1 and counts.max() / counts.sum() < self.max_category_ratio:
                    break
                top, left, bot, right = self.get_params(height=height, width=width)
                crop = semseg[top:bot, left:right]

        x = x[:, top:bot, left:right]
        semseg = crop
        return x, semseg

    def get_params(self, height, width):
        top = int(self.rng.integers(height - self.size[0] + 1, size=(1,)))
        left = int(self.rng.integers(width - self.size[1] + 1, size=(1,)))
        bot = top + self.size[0]
        right = left + self.size[1]
        return top, left, bot, right
