import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomRotation
from torchvision.transforms.functional import InterpolationMode

from .base.kd_stochastic_transform import KDStochasticTransform



class KDRandomRotation(KDStochasticTransform):
    def __init__(self, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.rotation = RandomRotation(interpolation=InterpolationMode(interpolation), **kwargs)
        self.degree_lb = self.og_degree_lb = self.rotation.degrees[0]
        self.degree_ub = self.og_degree_ub = self.rotation.degrees[1]

    def _scale_strength(self, factor):
        self.degree_lb = self.og_degree_lb * factor
        self.degree_ub = self.og_degree_ub * factor

    def __call__(self, x, ctx=None):
        fill = self.rotation.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = float(torch.empty(1).uniform_(float(self.degree_lb), float(self.degree_ub)).item())

        return F.rotate(img, angle, self.rotation.resample, self.rotation.expand, self.rotation.center, fill)
