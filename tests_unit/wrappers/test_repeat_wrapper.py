import unittest

import numpy as np

from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.repeat_wrapper import RepeatWrapper
from tests_mock.index_dataset import IndexDataset


class TestRepeatWrapper(unittest.TestCase):
    def assert_correct_x(self, ds, expected):
        self.assertEqual(len(expected), len(ds))
        batch = ModeWrapper(ds, mode="x")[:]
        x = [b[0] for b in batch]
        self.assertEqual(expected, x)
        _, counts = np.unique(ds.indices, return_counts=True)
        self.assertTrue(np.all(counts == ds.repetitions))

    def test_repetitions(self):
        ds = RepeatWrapper(IndexDataset(size=5), repetitions=2)
        self.assert_correct_x(ds, expected=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    def test_minsize_imperfectfit(self):
        ds = RepeatWrapper(IndexDataset(size=5), min_size=9)
        self.assert_correct_x(ds, expected=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    def test_minsize_perfectfit(self):
        ds = RepeatWrapper(IndexDataset(size=5), min_size=10)
        self.assert_correct_x(ds, expected=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
