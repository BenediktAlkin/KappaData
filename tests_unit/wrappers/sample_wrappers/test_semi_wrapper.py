import unittest

import torch

from kappadata.wrappers.sample_wrappers.semi_wrapper import SemiWrapper
from tests_util.datasets import create_class_dataset


class TestSemiWrapper(unittest.TestCase):
    def test(self):
        ds = SemiWrapper(dataset=create_class_dataset(size=100, n_classes=5, seed=234), semi_percent=0.1, seed=0)
        getitem = torch.tensor([ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(10, (getitem == -1).sum())
        getall = torch.tensor(ds.getall_class())
        self.assertEqual(10, (getall == -1).sum())
