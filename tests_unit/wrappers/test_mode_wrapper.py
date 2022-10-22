import unittest

from kappadata.datasets.kd_dataset import KDDataset
from kappadata.datasets.kd_subset import KDSubset
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_mock.index_dataset import IndexDataset


class TestModeWrapper(unittest.TestCase):
    class CustomGetitemDataset(KDDataset):
        @staticmethod
        def getitem_custom(idx, _=None):
            return idx

        def __len__(self):
            return 1

    def test_custom_getitem_fn(self):
        ds = ModeWrapper(dataset=self.CustomGetitemDataset(), mode="custom")
        self.assertEqual((0, {}), ds[0])

    class ContextPropagationDataset(KDDataset):
        @staticmethod
        def getitem_first(idx, ctx=None):
            ctx["message"] = 50
            return idx

        @staticmethod
        def getitem_second(idx, ctx=None):
            return idx + ctx["message"]

        def __len__(self):
            return 1

    def test_context_propagation(self):
        ds = ModeWrapper(dataset=self.ContextPropagationDataset(), mode="first second")
        self.assertEqual((0, 50, {"message": 50}), ds[0])

    def test_getitem_slices(self):
        ds = ModeWrapper(dataset=IndexDataset(size=10), mode="index")

        def unpack(data_ctx_list):
            return [data_ctx_item[0] for data_ctx_item in data_ctx_list]

        self.assertEquals([0, 1, 2], unpack(ds[:3]))
        self.assertEquals([3, 4, 5], unpack(ds[3:6]))
        self.assertEquals([8, 9], unpack(ds[8:]))
        self.assertEquals([9], unpack(ds[-1:]))
        self.assertEquals([0, 1], unpack(ds[:-8]))
        self.assertEquals([3, 2, 1, 0], unpack(ds[3::-1]))
        self.assertEquals(9, ds[-1][0])

    def test_getitem(self):
        ds = ModeWrapper(dataset=IndexDataset(size=10), mode="index x")
        self.assertEqual((0, 0, {}), ds[0])

    def test_subset_getitem(self):
        ds = ModeWrapper(dataset=KDSubset(IndexDataset(size=10), indices=[5, 6]), mode="index x")
        self.assertEqual((0, 5, {}), ds[0])
        self.assertEqual((1, 6, {}), ds[1])

    def test_len(self):
        ds = ModeWrapper(dataset=IndexDataset(size=10), mode="index x")
        self.assertEqual(10, len(ds))

    def test_subset_len(self):
        ds = ModeWrapper(dataset=KDSubset(IndexDataset(size=10), indices=[5, 6]), mode="index x")
        self.assertEqual(2, len(ds))

    def test_getattr_dataset(self):
        ds0 = IndexDataset(size=10)
        ds = ModeWrapper(dataset=ds0, mode="index x")
        self.assertEqual(ds0, ds.dataset)

    def test_getattr(self):
        ds = ModeWrapper(dataset=IndexDataset(size=10), mode="index x")
        self.assertEqual(10, ds.size)

    def test_iter(self):
        ds = ModeWrapper(dataset=IndexDataset(size=3), mode="index x")
        samples = [sample for sample in ds]
        self.assertEqual((0, 0, {}), samples[0])
        self.assertEqual((1, 1, {}), samples[1])
        self.assertEqual((2, 2, {}), samples[2])
