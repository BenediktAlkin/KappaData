import unittest
from torchvision.transforms import RandomCrop
from kappadata.transforms.kd_random_crop import KDRandomCrop
import torch
from unittest.mock import patch
import numpy as np

class TestKDRandomResizedCrop(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kd_rc = KDRandomCrop(size=18, seed=5)
        tv_rc = RandomCrop(size=18)

        patch_rng = np.random.default_rng(seed=5)
        with patch("torch.randint", lambda low, high, size: torch.tensor([int(patch_rng.integers(low, high))])):
            expected = tv_rc(x)
        actual = kd_rc(x)
        self.assertTrue(torch.all(expected == actual))


