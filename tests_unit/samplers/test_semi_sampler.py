import unittest

import torch

from kappadata.samplers.semi_sampler import SemiSampler
from tests_util.datasets.class_dataset import ClassDataset

class TestSemiSampler(unittest.TestCase):
    def test_len(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler = SemiSampler(dataset=ds, num_labeled=1, num_unlabeled=2, length_mode="labeled")
        self.assertEqual(12, len(sampler))
        self.assertEqual(12, sampler.effective_length)
        sampler = SemiSampler(dataset=ds, num_labeled=1, num_unlabeled=2, world_size=2, length_mode="labeled")
        self.assertEqual(6, len(sampler))
        self.assertEqual(12, sampler.effective_length)
        sampler = SemiSampler(dataset=ds, num_labeled=1, num_unlabeled=2, world_size=4, length_mode="labeled")
        self.assertEqual(3, len(sampler))
        self.assertEqual(12, sampler.effective_length)
        sampler = SemiSampler(dataset=ds, num_labeled=1, num_unlabeled=5, world_size=4, length_mode="labeled")
        self.assertEqual(6, len(sampler))
        self.assertEqual(24, sampler.effective_length)
        
        sampler = SemiSampler(dataset=ds, num_labeled=1, num_unlabeled=2, length_mode="unlabeled")
        self.assertEqual(3, len(sampler))
        self.assertEqual(3, sampler.effective_length)
        sampler = SemiSampler(dataset=ds, num_labeled=1, num_unlabeled=2, length_mode="all")
        self.assertEqual(6, len(sampler))
        self.assertEqual(6, sampler.effective_length)


    def test_1x1(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler = SemiSampler(dataset=ds, seed=9243, length_mode="labeled")
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual([3, -1, 0, -1, 1, -1, 2, -1], cls)

    def test_1x1_worldsize2_fulllen(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler0 = SemiSampler(dataset=ds, seed=9243, rank=0, length_mode="labeled")
        sampler1 = SemiSampler(dataset=ds, seed=9243, rank=1, length_mode="labeled")
        idx00 = list(sampler0)
        idx01 = list(sampler1)
        self.assertEqual([5, 3, 0, 1, 2, 1, 4, 3], idx00)
        self.assertEqual([2, 1, 0, 3, 5, 3, 4, 1], idx01)
        cls00 = [ds.getitem_class(idx) for idx in sampler0]
        cls10 = [ds.getitem_class(idx) for idx in sampler1]
        self.assertEqual([3, -1, 0, -1, 1, -1, 2, -1], cls00)
        self.assertEqual([1, -1, 0, -1, 3, -1, 2, -1], cls10)
        sampler0.set_epoch(1)
        sampler1.set_epoch(1)
        idx10 = list(sampler0)
        idx11 = list(sampler1)
        self.assertEqual([2, 1, 0, 3, 5, 3, 4, 1], idx10)
        self.assertEqual([5, 3, 2, 1, 4, 1, 0, 3], idx11)
        cls01 = [ds.getitem_class(idx) for idx in sampler0]
        cls11 = [ds.getitem_class(idx) for idx in sampler1]
        self.assertEqual([1, -1, 0, -1, 3, -1, 2, -1], cls01)
        self.assertEqual([3, -1, 1, -1, 2, -1, 0, -1], cls11)

    def test_1x1_worldsize2_truncatedlen(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3, 1])
        sampler = SemiSampler(dataset=ds, seed=9243, rank=0, world_size=2, length_mode="labeled")
        idx = list(sampler)
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual([0, 1, 5, 3, 2], idx)
        self.assertEqual([0, -1, 3, -1, 1], cls)

    def test_1x2(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3])
        sampler = SemiSampler(
            dataset=ds,
            num_labeled=1,
            num_unlabeled=2,
            seed=9243,
            length_mode="labeled",
        )
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual(
            [
                3, -1, -1,
                0, -1, -1,
                1, -1, -1,
                2, -1, -1,
            ],
            cls,
        )

    def test_1x4(self):
        ds = ClassDataset(classes=[0, -1, 1, -1, 2, 3, 4, 5, -1, 6])
        sampler = SemiSampler(
            dataset=ds,
            num_labeled=1,
            num_unlabeled=4,
            seed=9243,
            length_mode="labeled",
        )
        cls = [ds.getitem_class(idx) for idx in sampler]
        self.assertEqual(
            [
                1, -1, -1, -1, -1,
                3, -1, -1, -1, -1,
                6, -1, -1, -1, -1,
                2, -1, -1, -1, -1,
                0, -1, -1, -1, -1,
                4, -1, -1, -1, -1,
                5, -1, -1, -1, -1,
            ],
            cls,
        )
