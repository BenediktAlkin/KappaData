import unittest

from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.percent_filter_wrapper import PercentFilterWrapper
from tests_mock.index_dataset import IndexDataset


class TestPercentWrapper(unittest.TestCase):
    def assert_correct_x(self, ds, expected):
        self.assertEqual(len(expected), len(ds))
        self.assertEqual(expected, ModeWrapper(ds, mode="x")[:])

    def test_perfectfit_to(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), to_percent=0.3)
        self.assert_correct_x(ds, expected=[0, 1, 2])

    def test_perfectfit_from(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), from_percent=0.3)
        self.assert_correct_x(ds, expected=[3, 4, 5, 6, 7, 8, 9])

    def test_perfectfit_fromto(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), from_percent=0.3, to_percent=0.5)
        self.assert_correct_x(ds, expected=[3, 4])

    def test_imperfectfit_to_noceil(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), to_percent=0.25, ceil_to_index=False)
        self.assert_correct_x(ds, expected=[0, 1])

    def test_imperfectfit_to_ceil(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), to_percent=0.25, ceil_to_index=True)
        self.assert_correct_x(ds, expected=[0, 1, 2])

    def test_imperfectfit_from_noceil(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), from_percent=0.25, ceil_from_index=False)
        self.assert_correct_x(ds, expected=[2, 3, 4, 5, 6, 7, 8, 9])

    def test_imperfectfit_from_ceil(self):
        ds = PercentFilterWrapper(IndexDataset(size=10), from_percent=0.25, ceil_from_index=True)
        self.assert_correct_x(ds, expected=[3, 4, 5, 6, 7, 8, 9])
