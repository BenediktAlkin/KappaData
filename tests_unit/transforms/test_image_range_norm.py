import unittest

import torch

from kappadata.transforms.image_range_norm import ImageRangeNorm


class TestImageRangeNorm(unittest.TestCase):
    def test_3d(self):
        x = torch.linspace(0., 1., 3 * 32 * 32).view(3, 32, 32)
        y = ImageRangeNorm()(x)
        self.assertEqual(-1., y.min())
        self.assertEqual(1., y.max())

    def test_5d(self):
        x = torch.linspace(0., 1., 5 * 32 * 32).view(5, 32, 32)
        y = ImageRangeNorm()(x)
        self.assertEqual(-1., y.min())
        self.assertEqual(1., y.max())
