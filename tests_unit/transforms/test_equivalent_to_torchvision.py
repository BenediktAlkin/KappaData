import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import (
    ColorJitter,
    GaussianBlur,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
)

from kappadata.transforms.kd_color_jitter import KDColorJitter
from kappadata.transforms.kd_gaussian_blur import KDGaussianBlur
from kappadata.transforms.kd_random_crop import KDRandomCrop
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop


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
        kd_transform = kd_class(seed=5, **kwargs)

        for _ in range(10):
            x = torch.randn(3, 32, 32, generator=data_rng)

            expected = tv_transform(x)
            actual = kd_transform(x)
            self.assertTrue(torch.all(expected == actual))


    def test_color_jitter(self):
        self._run(
            kd_class=KDColorJitter,
            tv_class=ColorJitter,
            kwargs=dict(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        )

    def test_gaussian_blur(self):
        self._run(
            kd_class=KDGaussianBlur,
            tv_class=GaussianBlur,
            kwargs=dict(kernel_size=3, sigma=[1., 2.]),
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
