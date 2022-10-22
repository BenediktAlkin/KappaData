import unittest
from kappadata.datasets.kd_dataset import KDDataset
from kappadata.datasets.kd_subset import KDSubset
from tests_mock.index_dataset import IndexDataset

class TestKDSubset(unittest.TestCase):
    class MockWrapper(KDSubset):
        def __init__(self, dataset):
            indices = list(reversed(range(len(dataset))))
            super().__init__(dataset=dataset, indices=indices)

    def test_getattr(self):
        ds = IndexDataset(size=10)
        wrapped = self.MockWrapper(ds)
        double_wrapped = self.MockWrapper(wrapped)
        x = range(len(wrapped))
        for i, j in zip(x, reversed(x)):
            self.assertEquals(i, ds.getitem_x(i))
            self.assertEquals(j, wrapped.getitem_x(i))
            self.assertEquals(i, double_wrapped.getitem_x(i))
        self.assertEquals(ds, wrapped.dataset)
        self.assertEquals(ds, wrapped.root_dataset)
        self.assertEquals(wrapped, double_wrapped.dataset)
        self.assertEquals(ds, double_wrapped.root_dataset)
