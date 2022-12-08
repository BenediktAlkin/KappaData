import math
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np

from .base.kd_stochastic_transform import KDStochasticTransform


class KDRandAugment(KDStochasticTransform):
    """
    reimplementation based on the original paper https://arxiv.org/abs/1909.13719
    note that other implementations are vastly different:
    - timm:
      - each transform has a chance of 50% to be applied
        - number of applied transforms is stochastic
        - num_ops is essentially halved
      - magnitude std
      - more transforms than the original (solarize_add)
      - no identity transform (is essentially there because of the 50% apply rate)
    - torchvision:
      - magnitude is in [0, num_magnitude_bins)
        - if the range of a value would be in [0.0, 1.0] magnitude 9 would by default actually result in 0.3 (31 bins)
    """

    def __init__(
            self,
            num_ops: int,
            magnitude: int,
            interpolation: str,
            magnitude_std: float = 0.,
            magnitude_min: float = 0.,
            magnitude_max: float = 10.,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(num_ops, int) and 0 <= num_ops
        assert isinstance(magnitude, int) and 0 <= magnitude <= 10
        assert isinstance(magnitude_std, float) and 0. <= magnitude_std
        assert isinstance(magnitude_min, float) and 0. <= magnitude_min <= magnitude
        assert isinstance(magnitude_max, float) and magnitude <= magnitude <= 10.
        self.num_ops = num_ops
        if isinstance(interpolation, str):
            interpolation = InterpolationMode(interpolation)
        self.interpolation = interpolation
        self.ops = [
            self.identity,
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_horizontal,
            self.translate_vertical,
        ]
        self.magnitude = magnitude / 10
        self.magnitude_std = magnitude_std
        self.magnitude_min = magnitude_min / 10
        self.magnitude_max = magnitude_max / 10
        if magnitude_std == 0.:
            self.sample_magnitude = self._sample_magnitude_const
        elif magnitude_std == float("inf"):
            self.sample_magnitude = self._sample_magnitude_uniform
        else:
            self.sample_magnitude = self._sample_magnitude_normal

    def _sample_magnitude_const(self):
        return self.magnitude

    def _sample_magnitude_uniform(self):
        return self.rng.uniform(self.magnitude_min, self.magnitude)

    def _sample_magnitude_normal(self):
        sampled = self.magnitude + self.rng.normal(0, self.magnitude_std / 10)
        # convert to python float to be consistent with other sampling value dtypes
        return float(np.clip(sampled, self.magnitude_min, self.magnitude_max))

    def __call__(self, x, ctx=None):
        assert not torch.is_tensor(x), "some KDRandAugment transforms require input to be pillow image"
        transforms = self.rng.choice(self.ops, size=self.num_ops)
        for transform in transforms:
            x = transform(x, self.sample_magnitude())
        return x

    @staticmethod
    def identity(x, _):
        return x

    @staticmethod
    def auto_contrast(x, _):
        return F.autocontrast(x)

    @staticmethod
    def equalize(x, _):
        return F.equalize(x)

    def rotate(self, x, magnitude):
        # degrees in [-30, 30]
        degrees = 30 * magnitude
        if self.rng.random() < 0.5:
            degrees = -degrees
        return F.rotate(x, degrees, interpolation=self.interpolation)

    @staticmethod
    def solarize(x, magnitude):
        # lower threshold -> stronger augmentation
        # threshold >= 256 -> no augmentation
        # threshold in [0, 256]
        threshold = 256 - int(256 * magnitude)
        return F.solarize(x, threshold)

    def _adjust_factor(self, magnitude):
        offset = 0.9 * magnitude
        if self.rng.random() < 0.5:
            return 1 + offset
        return 1 - offset

    def color(self, x, magnitude):
        # factor == 0 -> black/white image
        # factor == 1 -> identity
        # factor == 2 -> double saturation
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_saturation(x, factor)

    @staticmethod
    def posterize(x, magnitude):
        # bits == 0 -> black image
        # bits == 8 -> identity
        # bits in [4, 8]
        # torchvision uses range [4, 8]
        # timm has multiple versions but the RandAug uses [0, 4]
        # timm notes that AutoAugment uses [4, 8] while TF EfficientNet uses [0, 4]
        bits = 4 + int(4 * magnitude)
        return F.posterize(x, bits)

    def contrast(self, x, magnitude):
        # factor == 0 -> solid gray image
        # factor == 1 -> identity
        # factor == 2 -> double contrast
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_contrast(x, factor)

    def brightness(self, x, magnitude):
        # factor == 0 -> black image
        # factor == 1 -> identity
        # factor == 2 -> double brightness
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_brightness(x, factor)

    def sharpness(self, x, magnitude):
        # factor == 0 -> blurred image
        # factor == 1 -> identity
        # factor == 2 -> double sharpness
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_sharpness(x, factor)

    def _shear_degrees(self, magnitude):
        # angle in [-0.3, 0.3]
        angle = 0.3 * magnitude
        # degrees roughly in [-16.7, 16.7]
        degrees = math.degrees(math.atan(angle))
        if self.rng.random() < 0.5:
            return -degrees
        return degrees

    def shear_x(self, x, magnitude):
        shear_degrees = self._shear_degrees(magnitude)
        # from torchvision
        return F.affine(
            x,
            angle=0.,
            translate=[0, 0],
            scale=1.,
            shear=[shear_degrees, 0.],
            interpolation=self.interpolation,
            center=[0, 0],
        )

    def shear_y(self, x, magnitude):
        shear_degrees = self._shear_degrees(magnitude)
        # from torchvision
        return F.affine(
            x,
            angle=0.,
            translate=[0, 0],
            scale=1.,
            shear=[0., shear_degrees],
            interpolation=self.interpolation,
            center=[0, 0],
        )

    def _translation(self, magnitude):
        # translation in [-0.45, 0.45]
        translation = 0.45 * magnitude
        if self.rng.random() < 0.5:
            return -translation
        return translation

    def translate_horizontal(self, x, magnitude):
        # PIL image sizes are (width, height)
        translation = int(self._translation(magnitude) * x.size[0])
        return F.affine(
            x,
            angle=0.,
            translate=[translation, 0],
            scale=1.,
            interpolation=self.interpolation,
            shear=[0., 0.],
        )

    def translate_vertical(self, x, magnitude):
        # PIL image sizes are (width, height)
        translation = int(self._translation(magnitude) * x.size[1])
        return F.affine(
            x,
            angle=0.,
            translate=[0, translation],
            scale=1.,
            interpolation=self.interpolation,
            shear=[0., 0.],
        )