import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import RandomHorizontalFlip

from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip


class TestKDRandomHorizontalFlip(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kd_hf = KDRandomHorizontalFlip(seed=5)
        tv_hf = RandomHorizontalFlip()

        patch_rng = np.random.default_rng(seed=5)
        with patch("torch.rand", lambda size: torch.tensor([patch_rng.random(size=size)])):
            expected = tv_hf(x)
        actual = kd_hf(x)
        self.assertTrue(torch.all(expected == actual))
