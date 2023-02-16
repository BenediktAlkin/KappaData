import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import ImageFilter
from torchvision.transforms import (
    ColorJitter,
    GaussianBlur,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomSolarize,
)
from torchvision.transforms.functional import to_pil_image, to_tensor

from kappadata.transforms.kd_color_jitter import KDColorJitter
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_gaussian_blur_tv import KDGaussianBlurTV
from kappadata.transforms.kd_random_crop import KDRandomCrop
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_rotation import KDRandomRotation
from kappadata.transforms.kd_random_solarize import KDRandomSolarize


class TestEquivalentToTorchvision(unittest.TestCase):
    def _run(self, **kwargs):
        patch_rng = np.random.default_rng(seed=5)
        patch_uniform_fn = lambda _, low, high: torch.tensor([patch_rng.uniform(low, high)], dtype=torch.float64)
        with patch("torch.randint", lambda low, high, size: torch.tensor([int(patch_rng.integers(low, high))])):
            with patch("torch.randperm", lambda n: torch.tensor(patch_rng.permutation(n))):
                with patch("torch.rand", lambda _: torch.tensor(patch_rng.random(), dtype=torch.float64)):
                    with patch("torch.Tensor.uniform_", patch_uniform_fn):
                        self._run_impl(**kwargs)

    def _run_impl(self, kd_class, tv_class, kwargs):
        data_rng = torch.Generator().manual_seed(5)

        tv_transform = tv_class(**kwargs)
        kd_transform = kd_class(**kwargs).set_rng(np.random.default_rng(seed=5))

        for _ in range(10):
            x = torch.randn(1, 4, 4, generator=data_rng)
            x = to_pil_image(x)

            expected = to_tensor(tv_transform(x))
            actual = to_tensor(kd_transform(x))
            self.assertTrue(torch.all(expected == actual))

    def test_color_jitter(self):
        self._run(
            kd_class=KDColorJitter,
            tv_class=ColorJitter,
            kwargs=dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        )

    def test_gaussian_blur_tv(self):
        self._run(
            kd_class=KDGaussianBlurTV,
            tv_class=GaussianBlur,
            kwargs=dict(kernel_size=23, sigma=[0.1, 2.]),
        )

    def test_gaussian_blur_pil(self):
        class GaussianBlurPIL:
            def __init__(self, sigma):
                self.sigma = sigma

            def __call__(self, x):
                return x.filter(ImageFilter.GaussianBlur(self.sigma))

        self._run(
            kd_class=KDGaussianBlurPIL,
            tv_class=GaussianBlurPIL,
            kwargs=dict(sigma=2.),
        )

    def test_random_crop(self):
        self._run(
            kd_class=KDRandomCrop,
            tv_class=RandomCrop,
            kwargs=dict(size=18),
        )

    def test_random_grayscale(self):
        self._run(
            kd_class=KDRandomGrayscale,
            tv_class=RandomGrayscale,
            kwargs=dict(p=0.3),
        )

    def test_random_horizontal_flip(self):
        self._run(
            kd_class=KDRandomHorizontalFlip,
            tv_class=RandomHorizontalFlip,
            kwargs=dict(p=0.5),
        )

    def test_random_resized_crop(self):
        self._run(
            kd_class=KDRandomResizedCrop,
            tv_class=RandomResizedCrop,
            kwargs=dict(size=18),
        )

    def test_random_rotation(self):
        self._run(
            kd_class=KDRandomRotation,
            tv_class=RandomRotation,
            kwargs=dict(degrees=25),
        )

    def test_random_solarize(self):
        self._run(
            kd_class=KDRandomSolarize,
            tv_class=RandomSolarize,
            kwargs=dict(p=0.2, threshold=128),
        )
