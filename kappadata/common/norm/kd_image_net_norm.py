from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDImageNetNorm(KDImageNorm):
    def __init__(self, **kwargs):
        super().__init__(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, **kwargs)
