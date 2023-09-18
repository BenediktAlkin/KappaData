import unittest

import torch

from kappadata.samplers.class_balanced_sampler import ClassBalancedSampler
from tests_util.datasets import ClassDataset


class TestClassBalancedSampler(unittest.TestCase):
    def test_autolength(self):
        ds = ClassDataset(classes=[0, 1, 0, 0])
        sampler = ClassBalancedSampler(ds)
        self.assertEqual(6, len(sampler))

    def test_manuallength(self):
        ds = ClassDataset(classes=[0, 1, 0, 0])
        sampler = ClassBalancedSampler(ds, samples_per_class=2)
        self.assertEqual(4, len(sampler))

    def test_sample_noshuffle(self):
        ds = ClassDataset(classes=[0, 1, 1, 1, 0])
        sampler = ClassBalancedSampler(ds, shuffle=False)
        indices = [i for i in sampler]
        classes = torch.tensor([ds.getitem_class(i) for i in indices])
        _, counts = classes.unique(return_counts=True)
        self.assertEqual([0, 4, 0, 1, 2, 3], indices)
        self.assertEqual([3, 3], counts.tolist())

    def test_sample_shuffle(self):
        ds = ClassDataset(classes=[0, 1, 1, 1, 0])
        sampler = ClassBalancedSampler(ds, shuffle=True, seed=0)
        indices = [i for i in sampler]
        classes = torch.tensor([ds.getitem_class(i) for i in indices])
        _, counts = classes.unique(return_counts=True)
        self.assertEqual([4, 1, 0, 2, 4, 3], indices)
        self.assertEqual([3, 3], counts.tolist())