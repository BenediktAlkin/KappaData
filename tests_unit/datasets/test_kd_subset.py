import unittest
from kappadata.datasets.kd_dataset import KDDataset
from kappadata.datasets.kd_subset import KDSubset
from tests_mock.index_dataset import IndexDataset
from kappadata.errors import UseModeWrapperException

class TestKDSubset(unittest.TestCase):
    def test_getattr_getitem(self):
        root_ds = IndexDataset(size=3)
        ds = KDSubset(root_ds, indices=[1, 2])
        self.assertEquals(1, ds.getitem_x(0))
        self.assertEquals(2, ds.getitem_x(1))

    def test_getattr_getitem_recursive(self):
        root_ds = IndexDataset(size=4)
        ds = KDSubset(KDSubset(root_ds, indices=[1, 2]), indices=[1])
        self.assertEquals(2, ds.getitem_x(0))

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