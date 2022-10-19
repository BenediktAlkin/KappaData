import unittest
from kappadata.wrappers.base.dataset_base import DatasetBase
from kappadata.wrappers.base.wrapper_base import WrapperBase

class TestWrapperBase(unittest.TestCase):
    class MockDataset(DatasetBase):
        @staticmethod
        def idxget_x(idx):
            return idx

        def __len__(self):
            return 10

    class MockWrapper(WrapperBase):
        def __init__(self, dataset):
            indices = list(reversed(range(len(dataset))))
            super().__init__(dataset=dataset, indices=indices)

    def test_getattr(self):
        ds = self.MockDataset()
        wrapped = self.MockWrapper(ds)
        double_wrapped = self.MockWrapper(wrapped)
        x = range(len(wrapped))
        for i, j in zip(x, reversed(x)):
            self.assertEquals(i, ds.idxget_x(i))
            self.assertEquals(j, wrapped.idxget_x(i))
            self.assertEquals(i, double_wrapped.idxget_x(i))
        self.assertEquals(ds, wrapped.dataset)
        self.assertEquals(ds, wrapped.root_dataset)
        self.assertEquals(wrapped, double_wrapped.dataset)
        self.assertEquals(ds, double_wrapped.root_dataset)
        