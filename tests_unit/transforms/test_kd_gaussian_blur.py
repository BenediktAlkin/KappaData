import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import GaussianBlur

from kappadata.transforms.kd_gaussian_blur import KDGaussianBlur


class TestKDGaussianBlur(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kwargs = dict(kernel_size=3, sigma=[1., 2.])
        kd_gb = KDGaussianBlur(seed=5, **kwargs)
        tv_gb = GaussianBlur(**kwargs)

        patch_rng = np.random.default_rng(seed=5)
        patch_uniform_fn = lambda _, low, high: torch.tensor([patch_rng.uniform(low, high)], dtype=torch.float64)
        with patch("torch.Tensor.uniform_", patch_uniform_fn):
            expected = tv_gb(x)
        actual = kd_gb(x)
        self.assertTrue(torch.all(expected == actual))
