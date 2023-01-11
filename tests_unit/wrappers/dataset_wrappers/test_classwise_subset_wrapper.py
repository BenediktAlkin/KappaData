import unittest

from kappadata.wrappers.dataset_wrappers.classwise_subset_wrapper import ClasswiseSubsetWrapper
from kappadata.wrappers.dataset_wrappers.shuffle_wrapper import ShuffleWrapper
from tests_util.datasets.class_dataset import ClassDataset
from tests_util.datasets.classification_dataset import ClassificationDataset
from itertools import chain
from kappadata.utils.class_counts import get_class_counts_from_dataset

class TestClasswiseSubsetWrapper(unittest.TestCase):
    def test(self):
        dataset = ClassificationDataset(x=list(range(6)), classes=[0, 1, 0, 0, 1, 0])
        subset = ClasswiseSubsetWrapper(dataset=dataset, end_percent=0.5)
        self.assertEqual(3, len(subset))
        items = [subset.getitem_x(i) for i in range(len(subset))]
        self.assertEqual([0, 2, 1], items)

    def test_shuffled(self):
        counts = [100, 150, 250]
        classes = list(chain(*[[i] * count for i, count in enumerate(counts)]))
        dataset = ShuffleWrapper(dataset=ClassDataset(classes=classes), seed=5)
        subset = ClasswiseSubsetWrapper(dataset=dataset, end_percent=0.1)
        self.assertEqual(50, len(subset))
        counts = get_class_counts_from_dataset(subset)
        self.assertEqual([10, 15, 25], counts.tolist())