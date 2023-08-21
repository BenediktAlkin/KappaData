import unittest

import torch

from tests_util.datasets.class_dataset import ClassDataset
from kappadata.wrappers.sample_wrappers import KDRandomClassWrapper


class TestOneHotWrapper(unittest.TestCase):
    @staticmethod
    def _new_dataset(size):
        return ClassDataset(classes=[-1] * size, n_classes=-1)

    @staticmethod
    def iterate(dataset):
        return [dataset.getitem_class(i) for i in range(len(dataset))]

    def test_random(self):
        ds = KDRandomClassWrapper(dataset=self._new_dataset(size=10), mode="random", num_classes=5, seed=2)
        self.assertEqual((5,), ds.getshape_class())
        classes = self.iterate(ds)
        self.assertEqual([3, 2, 1, 4, 3, 4, 2, 0, 0, 1], classes)

    def test_randperm(self):
        ds = KDRandomClassWrapper(dataset=self._new_dataset(size=10), mode="randperm", num_classes=5, seed=0)
        self.assertEqual((5,), ds.getshape_class())
        classes = self.iterate(ds)
        self.assertEqual([4, 0, 1, 3, 2] * 2, classes)

    def test_setter_mode(self):
        ds = KDRandomClassWrapper(dataset=self._new_dataset(size=10), mode="random", num_classes=5, seed=2)
        ds.mode = "randperm"
        classes = self.iterate(ds)
        self.assertEqual([3, 4, 1, 0, 2] * 2, classes)

    def test_setter_num_classes(self):
        ds = KDRandomClassWrapper(dataset=self._new_dataset(size=10), mode="random", num_classes=5, seed=2)
        ds.num_classes = 10
        self.assertEqual((10,), ds.getshape_class())
        classes = self.iterate(ds)
        self.assertEqual([8, 7, 1, 4, 8, 9, 2, 5, 0, 1], classes)

    def test_setter_seed(self):
        ds = KDRandomClassWrapper(dataset=self._new_dataset(size=10), mode="random", num_classes=5, seed=2)
        ds.seed = 3
        classes = self.iterate(ds)
        self.assertEqual([1, 3, 2, 2, 0, 0, 0, 0, 1, 4], classes)