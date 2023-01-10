import unittest

from kappadata.wrappers.dataset_wrappers.subset_wrapper import SubsetWrapper
from tests_util.datasets.index_dataset import IndexDataset


class TestSubsetWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, indices=[0, 1], start_index=0))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, indices=[0, 1], end_index=0))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, start_index=3.4))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, end_index=3.4))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(IndexDataset(size=10), end_index=3, start_index=4))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, start_percent=5))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, end_percent=5))
        self.assertRaises(AssertionError, lambda: SubsetWrapper(None, start_percent=0.25, end_percent=0.1))
        _ = SubsetWrapper(IndexDataset(size=5), indices=[0, 1])
        _ = SubsetWrapper(IndexDataset(size=5), start_index=3)
        _ = SubsetWrapper(IndexDataset(size=5), end_index=3)

    def test_indices(self):
        ds = SubsetWrapper(IndexDataset(size=10), indices=[3, 2, 5])
        self.assertEqual(3, len(ds))
        self.assertEqual([3, 2, 5], [ds.getitem_x(i) for i in range(len(ds))])

    def test_check_indices(self):
        ds = IndexDataset(size=10)
        with self.assertRaises(AssertionError):
            SubsetWrapper(ds, indices=[10])
        with self.assertRaises(AssertionError):
            SubsetWrapper(ds, indices=[-11])
        last = SubsetWrapper(ds, indices=[9])
        first = SubsetWrapper(ds, indices=[-10])
        self.assertEqual(9, last.getitem_x(0))
        self.assertEqual(0, first.getitem_x(0))

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

    def test_start_percent(self):
        ds = SubsetWrapper(IndexDataset(size=10), start_percent=0.8)
        self.assertEqual(2, len(ds))
        self.assertEqual([8, 9], [ds.getitem_x(i) for i in range(len(ds))])

    def test_start_percent_round(self):
        ds = SubsetWrapper(IndexDataset(size=10), start_percent=0.75)
        self.assertEqual(3, len(ds))
        self.assertEqual([7, 8, 9], [ds.getitem_x(i) for i in range(len(ds))])

    def test_end_percent(self):
        ds = SubsetWrapper(IndexDataset(size=10), end_percent=0.2)
        self.assertEqual(2, len(ds))
        self.assertEqual([0, 1], [ds.getitem_x(i) for i in range(len(ds))])

    def test_end_percent_round(self):
        ds = SubsetWrapper(IndexDataset(size=10), end_percent=0.25)
        self.assertEqual(2, len(ds))
        self.assertEqual([0, 1], [ds.getitem_x(i) for i in range(len(ds))])

    def test_start_and_end_percent(self):
        ds = SubsetWrapper(IndexDataset(size=10), start_percent=0.2, end_percent=0.7)
        self.assertEqual(5, len(ds))
        self.assertEqual([2, 3, 4, 5, 6], [ds.getitem_x(i) for i in range(len(ds))])

    def test_start_and_end_percent_round(self):
        ds = SubsetWrapper(IndexDataset(size=10), start_percent=0.25, end_percent=0.75)
        self.assertEqual(5, len(ds))
        self.assertEqual([2, 3, 4, 5, 6], [ds.getitem_x(i) for i in range(len(ds))])
