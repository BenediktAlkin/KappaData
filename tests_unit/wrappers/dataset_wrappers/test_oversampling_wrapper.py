import unittest

from kappadata.utils import get_class_counts_from_dataset
from kappadata.wrappers.dataset_wrappers.oversampling_wrapper import OversamplingWrapper
from tests_mock.class_dataset import ClassDataset


class TestOversamplingWrapper(unittest.TestCase):
    def test_multiply_minority_above_50percent(self):
        ds = OversamplingWrapper(ClassDataset(classes=[0] * 10 + [1] * 6), strategy="multiply")
        self.assertEqual(16, len(ds))
        self.assertEqual([0] * 10 + [1] * 6, [ds.getitem_class(i) for i in range(len(ds))])

        class_counts = get_class_counts_from_dataset(ds)
        self.assertEqual(10, class_counts[0])
        self.assertEqual(6, class_counts[1])

    def test_multiply_minority_nosamples(self):
        ds = OversamplingWrapper(ClassDataset(classes=[1] * 10), strategy="multiply")
        self.assertEqual(10, len(ds))
        self.assertEqual([1] * 10, [ds.getitem_class(i) for i in range(len(ds))])

        class_counts = get_class_counts_from_dataset(ds)
        self.assertEqual(0, class_counts[0])
        self.assertEqual(10, class_counts[1])

    def test_multiply_perfectfit(self):
        ds = OversamplingWrapper(ClassDataset(classes=[0] * 10 + [1] * 5), strategy="multiply")
        self.assertEqual(20, len(ds))
        self.assertEqual([0] * 10 + [1] * 10, [ds.getitem_class(i) for i in range(len(ds))])

        class_counts = get_class_counts_from_dataset(ds)
        self.assertEqual(10, class_counts[0])
        self.assertEqual(10, class_counts[1])

    def test_multiply_imperfectfit(self):
        ds = OversamplingWrapper(ClassDataset(classes=[0] * 10 + [1] * 4), strategy="multiply")
        self.assertEqual(18, len(ds))
        self.assertEqual([0] * 10 + [1] * 8, [ds.getitem_class(i) for i in range(len(ds))])

        class_counts = get_class_counts_from_dataset(ds)
        self.assertEqual(10, class_counts[0])
        self.assertEqual(8, class_counts[1])
