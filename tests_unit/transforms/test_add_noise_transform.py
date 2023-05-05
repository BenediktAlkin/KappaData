import unittest

import numpy as np
import torch

from kappadata.transforms import AddNoiseTransform


class TestAddNoiseTransform(unittest.TestCase):
    def test_gauss(self):
        x = torch.randn(5, generator=torch.Generator().manual_seed(0))
        transform = AddNoiseTransform().set_rng(np.random.default_rng(seed=3))
        actual = transform(x).tolist()
        expected = [
            1.3221055269241333,
            -0.25761908292770386,
            -2.2274184226989746,
            0.5296622514724731,
            -1.1029881238937378,
        ]
        self.assertEqual(expected, actual, actual)
