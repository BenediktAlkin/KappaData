import unittest
from torchvision.transforms import RandomResizedCrop
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
import torch
from unittest.mock import patch
import numpy as np

class TestKDRandomResizedCrop(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kd_rrc = KDRandomResizedCrop(size=18, seed=5)
        tv_rrc = RandomResizedCrop(size=18)

        patch_rng = np.random.default_rng(seed=5)
        with patch("torch.Tensor.uniform_", lambda _, low, high: torch.tensor([patch_rng.uniform(low, high)])):
            with patch("torch.randint", lambda low, high, size: torch.tensor([int(patch_rng.integers(low, high))])):
                expected = tv_rrc(x)
        actual = kd_rrc(x)
        self.assertTrue(torch.all(expected == actual))


