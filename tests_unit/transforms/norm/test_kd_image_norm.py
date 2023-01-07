import unittest

import torch

from kappadata.transforms.norm.kd_image_norm import KDImageNorm
from kappadata.transforms.norm.kd_image_range_norm import KDImageRangeNorm


class TestKDImageNorm(unittest.TestCase):
    def test_equal_to_range_norm(self):
        x = torch.linspace(0., 1., 5 * 32 * 32).view(5, 32, 32)
        range_norm = KDImageRangeNorm(inplace=False)
        norm = KDImageNorm(mean=(0.5,), std=(0.5,), inplace=False)
        range_normed = range_norm.normalize(x)
        normed = norm.normalize(x)
        self.assertTrue(torch.all(range_normed == normed))
        range_denormed = range_norm.denormalize(range_normed)
        denormed = norm.denormalize(normed)
        self.assertTrue(torch.all(x == range_denormed))
        self.assertTrue(torch.all(x == denormed))
        self.assertTrue(torch.all(range_denormed == denormed))
