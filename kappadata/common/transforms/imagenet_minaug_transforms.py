from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_solarize import KDRandomSolarize
from .norm.kd_image_net_norm import KDImageNetNorm


class ImagenetMinaugTransform(KDComposeTransform):
    def __init__(self, size=224, interpolation="bicubic", min_scale=0.08):
        super().__init__(transforms=[
            KDRandomResizedCrop(size=size, interpolation=interpolation, scale=(min_scale, 1.)),
            KDRandomHorizontalFlip(),
            KDImageNetNorm(),
        ])
