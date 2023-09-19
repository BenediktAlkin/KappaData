import unittest

from kappadata.wrappers.dataset_wrappers.intra_class_shuffle_wrapper import IntraClassShuffleWrapper
from tests_util.datasets.classification_dataset import ClassificationDataset


class TestIntraClassShuffleWrapper(unittest.TestCase):
    def _test(self, classes, expected_x, seed):
        dataset = IntraClassShuffleWrapper(
            dataset=ClassificationDataset(
                x=list(range(len(classes))),
                classes=classes,
            ),
            seed=seed,
        )
        x = [dataset.getitem_x(i) for i in range(len(dataset))]
        actual_classes = [dataset.getitem_class(i) for i in range(len(dataset))]
        self.assertEqual(expected_x, x)
        self.assertEqual(classes, actual_classes)

    def test_repeat(self):
        self._test(
            classes=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            expected_x=[0, 1, 2, 8, 9, 5, 6, 7, 3, 4],
            seed=0,
        )

    def test_repeatinterleave(self):
        self._test(
            classes=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            expected_x=[1, 0, 2, 3, 4, 5, 7, 6, 9, 8],
            seed=348,
        )

    def test_repeatinterleave_missing(self):
        self._test(
            classes=[0, 0, 1, 1, 3, 3, 4, 4],
            expected_x=[0, 1, 3, 2, 5, 4, 6, 7],
            seed=234,
        )

