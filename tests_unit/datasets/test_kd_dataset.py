import unittest

from kappadata.errors import UseModeWrapperException
from tests_util.index_dataset import IndexDataset


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

    def test_has_wrapper(self):
        root_ds = IndexDataset(size=3)
        self.assertFalse(root_ds.has_wrapper(None))

    def test_has_wrapper_type(self):
        root_ds = IndexDataset(size=3)
        self.assertFalse(root_ds.has_wrapper_type(None))

    def test_all_wrappers(self):
        root_ds = IndexDataset(size=3)
        self.assertEqual([], root_ds.all_wrappers)

    def test_all_wrapper_types(self):
        root_ds = IndexDataset(size=3)
        self.assertEqual([], root_ds.all_wrapper_types)