import unittest

import numpy as np
import torch

from kappadata.transforms import KDAdditiveGaussianNoise


class TestAddNoiseTransform(unittest.TestCase):
    def test_gauss(self):
        x = torch.randn(5, generator=torch.Generator().manual_seed(0))
        transform = KDAdditiveGaussianNoise(std=1.).set_rng(np.random.default_rng(seed=3))
        actual = transform(x).tolist()
        expected = [
            1.3221055269241333,
            -0.25761908292770386,
            -2.2274184226989746,
            0.5296622514724731,
            -1.1029881238937378,
        ]
        self.assertEqual(expected, actual, actual)

    def test_gauss_clip(self):
        x = torch.rand(10, generator=torch.Generator().manual_seed(0))
        transform = KDAdditiveGaussianNoise(std=5, clip_min=0, clip_max=1).set_rng(np.random.default_rng(seed=3))
        actual = transform(x).tolist()
        expected = [
            0.0,
            0.9472708702087402,
            0.0,
            0.0,
            0.21509423851966858,
            0.0,
            0.3907693326473236,
            0.5259208679199219,
            1.0,
            0.7289984226226807,
        ]
        self.assertEqual(expected, actual, actual)
