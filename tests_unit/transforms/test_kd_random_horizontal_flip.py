import unittest
from torchvision.transforms import RandomHorizontalFlip
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
import torch
from unittest.mock import patch
import numpy as np

class TestKDRandomHorizontalFlip(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kd_hf = KDRandomHorizontalFlip(seed=5)
        tv_hf = RandomHorizontalFlip()

        patch_rng = np.random.default_rng(seed=5)
        with patch("torch.Tensor.uniform_", lambda _, low, high: torch.tensor([patch_rng.uniform(low, high)])):
            with patch("torch.randint", lambda low, high, size: torch.tensor([int(patch_rng.integers(low, high))])):
                expected = tv_hf(x)
        actual = kd_hf(x)
        self.assertTrue(torch.all(expected == actual))


