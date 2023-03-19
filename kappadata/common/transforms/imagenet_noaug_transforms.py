from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_solarize import KDRandomSolarize
from .norm.kd_image_net_norm import KDImageNetNorm
from kappadata.transforms.kd_resize import KDResize
from torchvision.transforms import CenterCrop

class ImagenetNoaugTransform(KDComposeTransform):
    def __init__(self, resize_size=256, center_crop_size=224, interpolation="bicubic"):
        assert resize_size >= center_crop_size
        super().__init__(transforms=[
            KDResize(size=resize_size, interpolation=interpolation),
            CenterCrop(size=center_crop_size),
            KDImageNetNorm(),
        ])
