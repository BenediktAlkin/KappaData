import torch
import numpy as np
import unittest

from kappadata.utils.class_counts import get_class_counts_and_indices, get_class_counts
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

    def test_get_class_count_torch(self):
        classes = torch.tensor([0, 1, 1, 1, 0, 2, 5])
        counts, unlabeled_count = get_class_counts(classes=classes, n_classes=6)
        self.assertEqual([2, 3, 1, 0, 0, 1], counts.tolist())
        self.assertEqual(0, unlabeled_count)

    def test_get_class_count_list(self):
        classes = [0, 1, 1, 1, 0, 2, 5]
        counts, unlabeled_count = get_class_counts(classes=classes, n_classes=6)
        self.assertEqual([2, 3, 1, 0, 0, 1], counts.tolist())
        self.assertEqual(0, unlabeled_count)

    def test_get_class_count_unlabeled(self):
        classes = torch.tensor([0, 1, 1, 1, -1, -1, 0, -1, 2, 5, -1])
        counts, unlabeled_count = get_class_counts(classes=classes, n_classes=6)
        self.assertEqual([2, 3, 1, 0, 0, 1], counts.tolist())
        self.assertEqual(4, unlabeled_count)

    def test_get_class_count_invalid_class_negative(self):
        classes = torch.tensor([0, 1, -2, 1, -1, -1, 0, -1, 2, 5, -1])
        with self.assertRaises(AssertionError):
            get_class_counts(classes=classes, n_classes=6)

    def test_get_class_count_invalid_class_positive(self):
        classes = torch.tensor([0, 1, -1, 1, -1, -1, 0, -1, 2, 6, -1])
        with self.assertRaises(AssertionError):
            get_class_counts(classes=classes, n_classes=6)