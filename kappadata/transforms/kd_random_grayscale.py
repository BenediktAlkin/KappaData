import torchvision.transforms.functional as F

from .kd_random_apply import KDRandomApply


class KDRandomGrayscale(KDRandomApply):
    def __init__(self, p=0.1, **kwargs):
        super().__init__(p=p, **kwargs)

    def forward(self, x, ctx):
        if ctx is not None:
            ctx["random_grayscale"] = True
        num_output_channels = F.get_image_num_channels(x)
        return F.rgb_to_grayscale(x, num_output_channels=num_output_channels)