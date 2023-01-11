import unittest

from kappadata.utils.class_counts import get_class_counts_and_indices
from tests_util.datasets.class_dataset import ClassDataset


class TestClassCounts(unittest.TestCase):
    def test_get_class_counts_and_indices(self):
        dataset = ClassDataset(classes=[0, 1, 0, 0, 2, 1, 0])
        counts, indices = get_class_counts_and_indices(dataset)
        self.assertEqual([4, 2, 1], counts.tolist())
        self.assertEqual([0, 2, 3, 6], indices[0].tolist())
        self.assertEqual([1, 5], indices[1].tolist())
        self.assertEqual([4], indices[2].tolist())

    def test_get_class_counts_and_indices_zero_class(self):
        dataset = ClassDataset(classes=[0, 1, 3, 0])
        counts, indices = get_class_counts_and_indices(dataset)
        self.assertEqual([2, 1, 0, 1], counts.tolist())
        self.assertEqual([0, 3], indices[0].tolist())
        self.assertEqual([1], indices[1].tolist())
        self.assertEqual([], indices[2].tolist())
        self.assertEqual([2], indices[3].tolist())
