import unittest

import torch

from kappadata.samplers.semi_sampler import SemiSampler
from tests_util.datasets.class_dataset import ClassDataset

class TestSemiSampler(unittest.TestCase):
    def test_1x1(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler = SemiSampler(dataset=ds, generator=torch.Generator().manual_seed(9243))
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual([3, -1, 0, -1, 2, -1], cls)

    def test_1x1_noshuffle(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler = SemiSampler(dataset=ds, shuffle=False)
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual([0, -1, 1, -1, 2, -1], cls)

    def test_1x2(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler = SemiSampler(
            dataset=ds,
            num_labeled=1,
            num_unlabeled=2,
            generator=torch.Generator().manual_seed(9243),
        )
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual([3, -1, -1, 0, -1, -1], cls)

    def test_1x4(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3, 4, 5, -1, 6])
        sampler = SemiSampler(
            dataset=ds,
            num_labeled=1,
            num_unlabeled=4,
            generator=torch.Generator().manual_seed(9243),
        )
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual([0, -1, -1, -1, -1, 6, -1, -1, -1, -1, ], cls)
