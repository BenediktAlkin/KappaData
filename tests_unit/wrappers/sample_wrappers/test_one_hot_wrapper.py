import unittest

import torch

from kappadata.wrappers.sample_wrappers.one_hot_wrapper import OneHotWrapper
from tests_util.datasets import create_class_dataset


class TestOneHotWrapper(unittest.TestCase):
    def test_dtype(self):
        ds = OneHotWrapper(dataset=create_class_dataset(size=1, n_classes=5, seed=234))
        self.assertEqual(torch.float32, ds.getitem_class(0).dtype)

    def test_onehot(self):
        ds = OneHotWrapper(dataset=create_class_dataset(size=4, n_classes=5, seed=234))
        self.assertEqual(torch.float32, ds.getitem_class(0).dtype)
        self.assertEqual([1., 0., 0., 0., 0.], ds.getitem_class(0).tolist())
        self.assertEqual([0., 1., 0., 0., 0.], ds.getitem_class(1).tolist())
        self.assertEqual([0., 0., 0., 1., 0.], ds.getitem_class(2).tolist())
        self.assertEqual([0., 1., 0., 0., 0.], ds.getitem_class(3).tolist())
