import unittest

import torch

from kappadata.transforms.kd_binarize import KDBinarize


class TestKDBinarize(unittest.TestCase):
    def test_square(self):
        t = KDBinarize()
        x = torch.rand(size=(1, 16, 16), generator=torch.Generator().manual_seed(0))
        y = t(x)
        unique = y.unique()
        self.assertTrue([0, 1], unique.tolist())
