import unittest

import torch

from kappadata.datasets import KDDataset
from kappadata.utils import getall_as_tensor


class TestGetallAsTensor(unittest.TestCase):
    def _test(self, dataset, expected, item):
        actual = getall_as_tensor(dataset=dataset, item=item)
        self.assertTrue(torch.is_tensor(actual))
        self.assertEqual(torch.long, actual.dtype)
        self.assertEqual(expected, actual.tolist())

    def test_getitem_class_int(self):
        class Dataset(KDDataset):
            # noinspection PyUnusedLocal
            @staticmethod
            def getitem_class(idx, ctx=None):
                return idx

            def __len__(self):
                return 5

        self._test(dataset=Dataset(), expected=[0, 1, 2, 3, 4], item="class")

    def test_getitem_ogclass_int(self):
        class Dataset(KDDataset):
            # noinspection PyUnusedLocal
            @staticmethod
            def getitem_ogclass(idx, ctx=None):
                return idx

            def __len__(self):
                return 5

        self._test(dataset=Dataset(), expected=[0, 1, 2, 3, 4], item="ogclass")

    def test_getall_class_int(self):
        class Dataset(KDDataset):
            def getall_class(self):
                return list(range(len(self)))

            def __len__(self):
                return 5

        self._test(dataset=Dataset(), expected=[0, 1, 2, 3, 4], item="class")

    def test_getall_ogclass_int(self):
        class Dataset(KDDataset):
            def getall_ogclass(self):
                return list(range(len(self)))

            def __len__(self):
                return 5

        self._test(dataset=Dataset(), expected=[0, 1, 2, 3, 4], item="ogclass")
