import unittest

from kappadata.wrappers.dataset_wrappers import ShuffleWrapper
from tests_util.index_dataset import IndexDataset


class TestShuffleWrapper(unittest.TestCase):
    def test_shuffle_seeded(self):
        ds = IndexDataset(size=10)
        shuffled = ShuffleWrapper(ds, seed=5)
        self.assertEqual([7, 6, 1, 3, 2, 4, 0, 9, 5, 8], [shuffled.getitem_x(i) for i in range(len(ds))])
