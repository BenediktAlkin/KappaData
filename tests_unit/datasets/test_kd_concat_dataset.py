import unittest

from kappadata.datasets.kd_concat_dataset import KDConcatDataset
from kappadata.errors import UseModeWrapperException
from tests_util.datasets.class_dataset import ClassDataset
from tests_util.datasets.index_dataset import IndexDataset


class TestKDConcatDataset(unittest.TestCase):
    def test_getattr(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual(1, ds.getitem_x(1))
        self.assertEqual(1, ds.getitem_x(4))
        self.assertEqual([ds0, ds1], ds.datasets)
        self.assertEqual(3, ds.size)

    def test_dispose(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        ds.dispose()
        self.assertTrue(ds0.disposed)
        self.assertTrue(ds1.disposed)

    def test_dispose_context_manager(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        with KDConcatDataset([ds0, ds1]):
            pass
        self.assertTrue(ds0.disposed)
        self.assertTrue(ds1.disposed)

    def test_getitem(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        with self.assertRaises(UseModeWrapperException):
            _ = ds[0]

    def test_root_dataset_single(self):
        ds0 = IndexDataset(size=3)
        ds = KDConcatDataset([ds0])
        self.assertEqual(ds0, ds.root_dataset)

    def test_root_dataset_multi(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual(ds0, ds.root_dataset)

    def test_has_wrapper(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertFalse(ds.has_wrapper(None))

    def test_has_wrapper_type(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertFalse(ds.has_wrapper_type(None))

    def test_all_wrappers(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual([], ds.all_wrappers)

    def test_all_wrapper_types(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual([], ds.all_wrapper_types)

    def test_getall_class(self):
        ds0 = ClassDataset(classes=[0, 0, 1, 1, 0])
        ds1 = ClassDataset(classes=[2, 3, 0, 1])
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual([0, 0, 1, 1, 0, 2, 3, 0, 1], ds.getall_class())

    def test_balanced_sampling(self):
        ds0 = ClassDataset(classes=[0, 1])
        ds1 = ClassDataset(classes=[2, 3, 4, 5, 6, 7, 8])
        ds = KDConcatDataset([ds0, ds1], balanced_sampling=True)
        expected = [0, 2, 1, 3, 0, 4, 1, 5, 0, 6, 1, 7, 0, 8, 1, 2]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(expected))])

    def test_balanced_sampling_len(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1], balanced_sampling=True)
        with self.assertRaises(AssertionError):
            len(ds)

    def test_len(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual(7, len(ds))
