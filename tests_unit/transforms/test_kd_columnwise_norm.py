import unittest

import torch

from kappadata.transforms.kd_columnwise_norm import KDColumnwiseNorm


class TestKDColumnwiseRangeNorm(unittest.TestCase):
    def test_range(self):
        t = KDColumnwiseNorm(mode="range")
        x = torch.randn(3, 32, 16, generator=torch.Generator().manual_seed(4389))
        y = t(x)
        columnwise_min = y.min(dim=1).values
        columnwise_max = y.max(dim=1).values
        self.assertEqual((3, 16), columnwise_min.shape)
        self.assertEqual((3, 16), columnwise_max.shape)
        self.assertTrue(torch.all(columnwise_min == 0))
        self.assertTrue(torch.all(columnwise_max == 1))

    def test_moment(self):
        t = KDColumnwiseNorm(mode="moment")
        x = torch.randn(3, 32, 16, generator=torch.Generator().manual_seed(4389))
        y = t(x)
        columnwise_mean = y.mean(dim=1)
        columnwise_std = y.std(dim=1)
        self.assertEqual((3, 16), columnwise_mean.shape)
        self.assertEqual((3, 16), columnwise_std.shape)
        self.assertTrue(torch.allclose(columnwise_mean, torch.zeros_like(columnwise_mean), atol=1e-7))
        self.assertTrue(torch.allclose(columnwise_std, torch.ones_like(columnwise_mean)))
