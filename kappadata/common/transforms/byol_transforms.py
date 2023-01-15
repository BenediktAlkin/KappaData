from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_solarize import KDRandomSolarize
from kappadata.transforms.norm.kd_image_net_norm import KDImageNetNorm


class BYOLTransform0(KDComposeTransform):
    def __init__(self):
        super().__init__(transforms=[
            KDRandomResizedCrop(size=224, interpolation="bicubic"),
            KDRandomHorizontalFlip(),
            KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            KDGaussianBlurPIL(sigma=(0.1, 2.0)),
            KDRandomGrayscale(p=0.2),
            KDImageNetNorm(),
        ])


class BYOLTransform1(KDComposeTransform):
    def __init__(self):
        super().__init__(transforms=[
            KDRandomResizedCrop(size=224, interpolation="bicubic"),
            KDRandomHorizontalFlip(),
            KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            KDRandomGaussianBlurPIL(p=0.1, sigma=(0.1, 2.0)),
            KDRandomGrayscale(p=0.2),
            KDRandomSolarize(p=0.2, threshold=128),
            KDImageNetNorm(),
        ])
