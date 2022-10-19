import unittest
from tests_mock.index_dataset import IndexDataset
from kappadata.wrappers.base.mode_wrapper import ModeWrapper

class TestModeWrapper(unittest.TestCase):
    def test_slices(self):
        ds = ModeWrapper(dataset=IndexDataset(size=10), mode="index")
        self.assertEquals([(0,), (1,), (2,)], ds[:3])
        self.assertEquals([(3,), (4,), (5,)], ds[3:6])
        self.assertEquals([(8,), (9,)], ds[8:])
        self.assertEquals([(9,)], ds[-1:])
        self.assertEquals([(0,), (1,)], ds[:-8])
        self.assertEquals([(3,), (2,), (1,), (0,)], ds[3::-1])
        self.assertEquals((9,), ds[-1])