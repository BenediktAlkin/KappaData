import unittest
from kappadata.datasets.kd_concat_dataset import KDConcatDataset
from tests_mock.index_dataset import IndexDataset
from kappadata.errors import UseModeWrapperException

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

    def test_getitem(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        with self.assertRaises(UseModeWrapperException):
            _ = ds[0]
            
    def test_root_dataset_len1(self):
        ds0 = IndexDataset(size=3)
        ds = KDConcatDataset([ds0])
        self.assertEqual(ds0, ds.root_dataset)

    def test_root_dataset_len2(self):
        ds0 = IndexDataset(size=3)
        ds1 = IndexDataset(size=4)
        ds = KDConcatDataset([ds0, ds1])
        self.assertEqual(ds0, ds.root_dataset)