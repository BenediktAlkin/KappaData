import unittest

import torch

from kappadata.samplers.distributed_sampler import DistributedSampler


class TestDistributedSampler(unittest.TestCase):
    def test_equal_to_torch(self):
        seed = 435
        ds = list(range(10))
        kd = DistributedSampler(ds, rank=0, num_replicas=3, seed=seed)
        th = torch.utils.data.DistributedSampler(ds, rank=0, num_replicas=3, seed=seed)
        self.assertEqual(list(iter(kd)), list(iter(th)))

    def test_repeated_aug(self):
        seed = 435
        ds = list(range(10))
        kd0 = DistributedSampler(ds, num_repeats=4, rank=0, num_replicas=3, seed=seed, drop_last=True)
        kd1 = DistributedSampler(ds, num_repeats=4, rank=1, num_replicas=3, seed=seed, drop_last=True)
        kd2 = DistributedSampler(ds, num_repeats=4, rank=2, num_replicas=3, seed=seed, drop_last=True)
        self.assertEqual([2, 2, 3], list(iter(kd0)))
        self.assertEqual([2, 3, 3], list(iter(kd1)))
        self.assertEqual([2, 3, 1], list(iter(kd2)))
