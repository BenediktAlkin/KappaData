import unittest

import torch

from kappadata.samplers.random_sampler import RandomSampler


class TestRandomSampler(unittest.TestCase):
    def test_equal_to_torch(self):
        seed = 435
        ds = list(range(10))
        kd = RandomSampler(ds, generator=torch.Generator().manual_seed(seed))
        th = torch.utils.data.RandomSampler(ds, generator=torch.Generator().manual_seed(seed))
        self.assertEqual(list(iter(kd)), list(iter(th)))

    def test_repeated_aug(self):
        seed = 435
        ds = list(range(10))
        kd = RandomSampler(ds, num_repeats=3, generator=torch.Generator().manual_seed(seed))
        self.assertEqual([2, 2, 2, 3, 3, 3, 1, 1, 1, 9], list(iter(kd)))