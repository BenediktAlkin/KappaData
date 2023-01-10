import unittest

import torch

from kappadata.transforms.norm.kd_image_range_norm import KDImageRangeNorm


class TestKDImageRangeNorm(unittest.TestCase):
    def test_3d(self):
        x = torch.linspace(0., 1., 3 * 32 * 32).view(3, 32, 32)
        norm = KDImageRangeNorm(inplace=False)
        y = norm(x)
        self.assertEqual(-1., y.min())
        self.assertEqual(1., y.max())

    def test_5d(self):
        x = torch.linspace(0., 1., 5 * 32 * 32).view(5, 32, 32)
        norm = KDImageRangeNorm(inplace=False)
        y = norm(x)
        self.assertEqual(-1., y.min())
        self.assertEqual(1., y.max())

    def test_denormalize_inplace(self):
        x = torch.linspace(-1., 1., 3 * 32 * 32).view(3, 32, 32)
        denorm = KDImageRangeNorm(inplace=True, inverse=True)
        y = denorm(x)
        self.assertEqual(0., y.min())
        self.assertEqual(1., y.max())

    def test_denormalize_outplace(self):
        x = torch.linspace(-1., 1., 3 * 32 * 32).view(3, 32, 32)
        denorm = KDImageRangeNorm(inplace=False, inverse=True)
        y = denorm(x)
        self.assertEqual(0., y.min())
        self.assertEqual(1., y.max())

    def test_normalize_denormalize(self):
        x = torch.linspace(0., 1., 5 * 32 * 32).view(5, 32, 32)
        norm = KDImageRangeNorm(inplace=False)
        denorm = KDImageRangeNorm(inplace=False, inverse=True)
        normed = norm(x)
        x_hat = denorm(normed)
        self.assertTrue(torch.allclose(x, x_hat))
