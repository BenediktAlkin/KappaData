import unittest

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.errors import UseModeWrapperException
from tests_mock.index_dataset import IndexDataset


class TestKDWrapper(unittest.TestCase):
    def test_getattr_getitem(self):
        root_ds = IndexDataset(size=3)
        ds = KDWrapper(root_ds)
        self.assertEquals(0, ds.getitem_x(0))
        self.assertEquals(1, ds.getitem_x(1))
        self.assertEquals(2, ds.getitem_x(2))

    def test_getattr_getitem_recursive(self):
        root_ds = IndexDataset(size=2)
        ds = KDWrapper(KDWrapper(root_ds))
        self.assertEquals(0, ds.getitem_x(0))
        self.assertEquals(1, ds.getitem_x(1))

    def test_getattr_dataset(self):
        root_ds = IndexDataset(size=3)
        ds = KDWrapper(root_ds)
        self.assertEqual(root_ds, ds.dataset)

    def test_getattr_dataset_recursive(self):
        ds0 = IndexDataset(size=3)
        ds1 = KDWrapper(ds0)
        ds = KDWrapper(ds1)
        self.assertEqual(ds1, ds.dataset)
        self.assertEqual(ds0, ds.dataset.dataset)

    def test_getattr(self):
        ds = KDWrapper(IndexDataset(size=3))
        self.assertEqual(3, ds.size)

    def test_getattr_recursive(self):
        ds = KDWrapper(KDWrapper(IndexDataset(size=3)))
        self.assertEqual(3, ds.size)

    def test_dispose(self):
        ds = KDWrapper(IndexDataset(size=3))
        ds.dispose()
        self.assertTrue(ds.disposed)

    def test_dispose_context_manager(self):
        with KDWrapper(IndexDataset(size=3)) as ds:
            pass
        self.assertTrue(ds.disposed)

    def test_root_dataset(self):
        root_ds = IndexDataset(size=3)
        ds = KDWrapper(root_ds)
        self.assertEqual(root_ds, ds.root_dataset)

    def test_root_dataset_recursive(self):
        root_ds = IndexDataset(size=3)
        ds = KDWrapper(KDWrapper(root_ds))
        self.assertEqual(root_ds, ds.root_dataset)

    def test_getitem(self):
        ds = KDWrapper(IndexDataset(size=3))
        with self.assertRaises(UseModeWrapperException):
            _ = ds[0]

    def test_has_wrapper(self):
        root_ds = IndexDataset(size=3)
        ds = KDWrapper(root_ds)
        self.assertFalse(root_ds.has_wrapper(ds))
        self.assertTrue(ds.has_wrapper(ds))

    def test_has_wrapper_type(self):
        root_ds = IndexDataset(size=3)
        ds = KDWrapper(root_ds)
        self.assertFalse(root_ds.has_wrapper_type(KDWrapper))
        self.assertTrue(ds.has_wrapper_type(KDWrapper))

    def test_all_wrappers(self):
        root_ds = IndexDataset(size=3)
        wrapper1 = KDWrapper(root_ds)
        wrapper2 = KDWrapper(wrapper1)
        self.assertEqual([wrapper1], wrapper1.all_wrappers)
        self.assertEqual([wrapper2, wrapper1], wrapper2.all_wrappers)

    def test_all_wrapper_types(self):
        root_ds = IndexDataset(size=3)
        wrapper1 = KDWrapper(root_ds)
        wrapper2 = KDWrapper(wrapper1)
        self.assertEqual([KDWrapper], wrapper1.all_wrapper_types)
        self.assertEqual([KDWrapper, KDWrapper], wrapper2.all_wrapper_types)
