import unittest

import torch

from kappadata.transforms import KDMinsize


class TestKDRearrange(unittest.TestCase):
    def test_1d(self):
        t = KDMinsize(size=32)
        self.assertEqual((3, 32, 32), t(torch.rand(3, 4, 4)).shape)
        self.assertEqual((3, 272, 32), t(torch.rand(3, 34, 4)).shape)
        self.assertEqual((3, 32, 272), t(torch.rand(3, 4, 34)).shape)
        self.assertEqual((3, 64, 34), t(torch.rand(3, 64, 34)).shape)

    def test_2d(self):
        t = KDMinsize(size=(32, 48))
        self.assertEqual((3, 32, 48), t(torch.rand(3, 4, 4)).shape)
        self.assertEqual((3, 32, 48), t(torch.rand(3, 43, 4)).shape)
        self.assertEqual((3, 32, 48), t(torch.rand(3, 4, 53)).shape)
        self.assertEqual((3, 54, 48), t(torch.rand(3, 54, 48)).shape)
