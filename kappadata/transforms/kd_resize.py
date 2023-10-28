from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from .base import KDTransform


class KDResize(KDTransform):
    """ wrapper for torchvision.transforms.Resize as it doesn't support passing a string as interpolation """

    def __init__(self, *args, ctx_prefix=None, interpolation="bilinear", **kwargs):
        super().__init__(ctx_prefix=ctx_prefix)
        self.resize = Resize(*args, interpolation=InterpolationMode(interpolation), antialias=True, **kwargs)

    def __call__(self, x, ctx=None):
        return self.resize(x)
