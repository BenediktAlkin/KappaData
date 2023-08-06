import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from kappadata.transforms.base.kd_transform import KDTransform
from kappadata.utils.param_checking import to_2tuple


class KDSemsegResize(KDTransform):
    def __init__(self, size, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.interpolation = InterpolationMode(interpolation)

    def __call__(self, xsemseg, ctx=None):
        x, semseg = xsemseg
        x = resize(x, self.size, self.interpolation)
        squeeze_semseg = False
        if torch.is_tensor(semseg):
            semseg = semseg.unsqueeze(0)
            squeeze_semseg = True
        semseg = resize(semseg, self.size, InterpolationMode.NEAREST)
        if squeeze_semseg:
            semseg = semseg.squeeze(0)
        return x, semseg
