import unittest

import torch

from kappadata.samplers.weighted_sampler import WeightedSampler


class TestWeightedSampler(unittest.TestCase):
    def test_length(self):
        ds = torch.arange(10)
        weights = torch.arange(1., 11.)
        sampler1 = WeightedSampler(ds, weights=weights)
        self.assertEqual(10, len(sampler1))
        sampler2 = WeightedSampler(ds, weights=weights, size=5)
        self.assertEqual(5, len(sampler2))
        # check that length doesnt influence sampling
        samples1 = [i for i in sampler1]
        samples2 = [i for i in sampler2]
        self.assertEqual(samples1[:5], samples2)

    def test_length_rank0worldsize2(self):
        ds = torch.arange(10)
        weights = torch.arange(1., 11.)
        sampler1 = WeightedSampler(ds, weights=weights, rank=0, world_size=2)
        self.assertEqual(5, len(sampler1))
        sampler2 = WeightedSampler(ds, weights=weights, size=5, rank=0, world_size=2)
        self.assertEqual(2, len(sampler2))
        # check that length doesnt influence sampling
        samples1 = [i for i in sampler1]
        samples2 = [i for i in sampler2]
        self.assertEqual(samples1[:2], samples2)

    def test_no_replacement(self):
        ds = torch.arange(10)
        weights = torch.arange(1., 11.)
        sampler = WeightedSampler(ds, weights=weights)
        self.assertEqual(10, len(sampler))
        samples = torch.tensor([i for i in sampler])
        self.assertEqual(10, samples.unique().numel())

    def test_no_replacement_zeroweight(self):
        ds = torch.arange(10)
        weights = torch.arange(0., 10.)
        sampler = WeightedSampler(ds, weights=weights)
        self.assertEqual(10, len(sampler))
        samples = torch.tensor([i for i in sampler])
        self.assertEqual(10, samples.unique().numel())