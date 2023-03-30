import unittest

import torch

from kappadata.transforms.kd_bucketize import KDBucketize


class TestKDBucketize(unittest.TestCase):
    def test_image(self):
        x = torch.rand(1, 4, 4, generator=torch.Generator().manual_seed(0))
        x[0, 0, 0] = 1.
        transform = KDBucketize(n_buckets=5, min_value=0., max_value=1.)
        y = transform(x)
        self.assertEqual([[[4, 3, 0, 0], [1, 3, 2, 4], [2, 3, 1, 2], [0, 0, 1, 2]]], y.tolist())
