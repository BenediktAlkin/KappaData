import unittest

import einops
import torch

from kappadata.transforms.kd_rearrange import KDRearrange


class TestKDRearrange(unittest.TestCase):
    def test_square(self):
        x = torch.randn(3, 4, 4)
        transform = KDRearrange(pattern="c h w -> (h w) c")
        y = transform(x)
        self.assertEqual((16, 3), y.shape)
