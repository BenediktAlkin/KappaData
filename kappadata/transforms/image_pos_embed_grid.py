import torch
from torchvision.transforms.functional import to_tensor

from .base.kd_transform import KDTransform


class ImagePosEmbedGrid(KDTransform):
    def __call__(self, x, _=None):
        if not torch.is_tensor(x):
            x = to_tensor(x)
        _, h, w = x.shape
        h_coords = torch.linspace(-1., 1., h)
        w_coords = torch.linspace(-1., 1., w)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        return torch.concat([x, grid_h.unsqueeze(0), grid_w.unsqueeze(0)])
