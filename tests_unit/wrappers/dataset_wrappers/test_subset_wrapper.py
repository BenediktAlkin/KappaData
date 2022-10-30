import unittest

from kappadata.wrappers.dataset_wrappers.subset_wrapper import SubsetWrapper
from tests_util.index_dataset import IndexDataset


class TestSubsetWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, indices=[0, 1], start_index=0))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, indices=[0, 1], end_index=0))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, start_index=3.4))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, end_index=3.4))
        _ = SubsetWrapper(IndexDataset(size=5), indices=[0, 1])
        _ = SubsetWrapper(IndexDataset(size=5), start_index=3)
        _ = SubsetWrapper(IndexDataset(size=5), end_index=3)

    def test_indices(self):
        ds = SubsetWrapper(IndexDataset(size=10), indices=[3, 2, 5])
        self.assertEqual(3, len(ds))
        self.assertEqual([3, 2, 5], [ds.getitem_x(i) for i in range(len(ds))])

    def test_start_index(self):
        ds = SubsetWrapper(IndexDataset(size=10), start_index=6)
        self.assertEqual(4, len(ds))
        self.assertEqual([6, 7, 8, 9], [ds.getitem_x(i) for i in range(len(ds))])

    def test_end_index(self):
        ds = SubsetWrapper(IndexDataset(size=10), end_index=2)
        self.assertEqual(2, len(ds))
        self.assertEqual([0, 1], [ds.getitem_x(i) for i in range(len(ds))])

    def test_start_and_end_index(self):
        ds = SubsetWrapper(IndexDataset(size=10), start_index=3, end_index=6)
        self.assertEqual(3, len(ds))
        self.assertEqual([3, 4, 5], [ds.getitem_x(i) for i in range(len(ds))])
