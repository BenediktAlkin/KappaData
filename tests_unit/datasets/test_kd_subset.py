import unittest

from kappadata.datasets.kd_subset import KDSubset
from kappadata.errors import UseModeWrapperException
from tests_util.datasets.class_dataset import ClassDataset
from tests_util.datasets.index_dataset import IndexDataset


class TestKDSubset(unittest.TestCase):
    def test_getattr_getitem(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(root_ds, indices=[1, 2])
        self.assertEqual(1, ds.getitem_x(0))
        self.assertEqual(2, ds.getitem_x(1))

    def test_getattr_getitem_recursive(self):
        root_ds = IndexDataset(size=4)
        ds = KDSubset(KDSubset(root_ds, indices=[1, 2]), indices=[1])
        self.assertEqual(2, ds.getitem_x(0))

    def test_getattr_dataset(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(root_ds, indices=[1, 2])
        self.assertEqual(root_ds, ds.dataset)

    def test_getattr_dataset_recursive(self):
        ds0 = IndexDataset(size=3)
        ds1 = KDSubset(ds0, indices=[1, 2])
        ds = KDSubset(ds1, indices=[1])
        self.assertEqual(ds1, ds.dataset)
        self.assertEqual(ds0, ds.dataset.dataset)

    def test_getattr(self):
        ds = KDSubset(IndexDataset(size=3), indices=[1, 2])
        self.assertEqual(3, ds.size)

    def test_getattr_recursive(self):
        ds = KDSubset(KDSubset(IndexDataset(size=3), indices=[1, 2]), indices=[0])
        self.assertEqual(3, ds.size)

    def test_dispose(self):
        ds = KDSubset(IndexDataset(size=3), indices=[0, 1])
        ds.dispose()
        self.assertTrue(ds.disposed)

    def test_dispose_context_manager(self):
        with KDSubset(IndexDataset(size=3), indices=[0, 1]) as ds:
            pass
        self.assertTrue(ds.disposed)

    def test_root_dataset(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(root_ds, indices=[0, 1])
        self.assertEqual(root_ds, ds.root_dataset)

    def test_root_dataset_recursive(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(KDSubset(root_ds, indices=[0, 1]), indices=[0])
        self.assertEqual(root_ds, ds.root_dataset)

    def test_getitem(self):
        ds = KDSubset(IndexDataset(size=3), indices=[0, 1])
        with self.assertRaises(UseModeWrapperException):
            _ = ds[0]

    def test_has_wrapper(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(root_ds, indices=[0, 1])
        self.assertFalse(root_ds.has_wrapper(ds))
        self.assertTrue(ds.has_wrapper(ds))

    def test_has_wrapper_type(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(root_ds, indices=[0, 1])
        self.assertFalse(root_ds.has_wrapper_type(KDSubset))
        self.assertTrue(ds.has_wrapper_type(KDSubset))

    def test_all_wrappers(self):
        root_ds = IndexDataset(size=3)
        wrapper1 = KDSubset(root_ds, indices=[0, 1])
        wrapper2 = KDSubset(wrapper1, indices=[1, 0])
        self.assertEqual([wrapper1], wrapper1.all_wrappers)
        self.assertEqual([wrapper2, wrapper1], wrapper2.all_wrappers)

    def test_all_wrapper_types(self):
        root_ds = IndexDataset(size=3)
        wrapper1 = KDSubset(root_ds, indices=[0, 1])
        wrapper2 = KDSubset(wrapper1, indices=[1, 0])
        self.assertEqual([KDSubset], wrapper1.all_wrapper_types)
        self.assertEqual([KDSubset, KDSubset], wrapper2.all_wrapper_types)

    def test_getall_class(self):
        root_ds = ClassDataset(classes=[0, 0, 1, 1, 0, 2])
        wrapper1 = KDSubset(root_ds, indices=[0, 2, 1, 5])
        self.assertEqual([0, 1, 0, 2], wrapper1.getall_class())

    def test_getshape(self):
        root_ds = ClassDataset(classes=[0, 0, 1, 1, 0, 2])
        wrapper1 = KDSubset(root_ds, indices=[0, 2, 1, 5])
        self.assertEqual((3,), wrapper1.getshape("class"))

    def test_getdim(self):
        root_ds = ClassDataset(classes=[0, 0, 1, 1, 0, 2])
        wrapper1 = KDSubset(root_ds, indices=[0, 2, 1, 5])
        self.assertEqual(3, wrapper1.getdim("class"))
        self.assertEqual(3, wrapper1.getdim_class())
