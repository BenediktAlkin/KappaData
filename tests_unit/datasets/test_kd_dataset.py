import unittest

from tests_mock.index_dataset import IndexDataset
from kappadata.errors import UseModeWrapperException

class TestKDDataset(unittest.TestCase):
    def test_dispose(self):
        ds = IndexDataset(size=3)
        ds.dispose()
        self.assertTrue(ds.disposed)

    def test_dispose_context_manager(self):
        with IndexDataset(size=3) as ds:
            pass
        self.assertTrue(ds.disposed)

    def test_root_dataset(self):
        ds = IndexDataset(size=3)
        self.assertEqual(ds, ds.root_dataset)

    def test_getitem(self):
        ds = IndexDataset(size=3)
        with self.assertRaises(UseModeWrapperException):
            _ = ds[0]
