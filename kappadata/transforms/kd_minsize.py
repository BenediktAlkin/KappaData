import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode, get_dimensions

from .base import KDTransform


class KDMinsize(KDTransform):
    """
    if image is smaller than minimal size:
        resize to minimal size
    else:
        pass
    """

    def __init__(self, ctx_prefix=None, interpolation="bilinear", **kwargs):
        super().__init__(ctx_prefix=ctx_prefix)
        self.resize = Resize(interpolation=InterpolationMode(interpolation), **kwargs)

    def __call__(self, x, ctx=None):
        _, image_height, image_width = get_dimensions(x)
        if isinstance(self.resize.size, int):
            if image_height >= self.resize.size and image_width >= self.resize.size:
                return x
        elif len(self.resize.size) == 2:
            if image_height >= self.resize.size[0] and image_width >= self.resize.size[1]:
                return x
        return self.resize(x)
