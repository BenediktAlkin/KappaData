import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.transforms import RandomGrayscale

from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale


class TestKDRandomGrayscale(unittest.TestCase):
    def test_equivalent_to_torchvision(self):
        x = torch.randn(3, 32, 32)
        kwargs = dict(p=0.3)
        kd_rg = KDRandomGrayscale(seed=5, **kwargs)
        tv_rg = RandomGrayscale(**kwargs)

        patch_rng = np.random.default_rng(seed=5)
        with patch("torch.rand", lambda _: torch.tensor(patch_rng.random(), dtype=torch.float64)):
            expected = tv_rg(x)
        actual = kd_rg(x)
        self.assertTrue(torch.all(expected == actual))
