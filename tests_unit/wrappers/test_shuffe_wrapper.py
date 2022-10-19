import unittest

from kappadata.wrappers.shuffle_wrapper import ShuffleWrapper
from kappadata.wrappers.base.dataset_base import DatasetBase

class TestShuffleWrapper(unittest.TestCase):


    def test_shuffle_seeded(self):
        ds = self.IndexDataset(size=10)
        shuffled = ShuffleWrapper(ds, seed=5)
        self.assertEquals([7, 6, 1, 3, 2, 4, 0, 9, 5, 8], [shuffled.idxget_x(i) for i in range(len(ds))])