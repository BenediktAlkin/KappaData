import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import ColorJitter

from kappadata.transforms.kd_color_jitter import KDColorJitter


class TestKDColorJitter(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kd_cj = KDColorJitter(seed=5, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        tv_cj = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)

        patch_rng = np.random.default_rng(seed=5)
        patch_uniform_fn = lambda _, low, high: torch.tensor([patch_rng.uniform(low, high)], dtype=torch.float64)
        with patch("torch.Tensor.uniform_", patch_uniform_fn):
            with patch("torch.randperm", lambda n: torch.tensor(patch_rng.permutation(n))):
                expected = tv_cj(x)
        actual = kd_cj(x)
        self.assertTrue(torch.all(expected == actual))
