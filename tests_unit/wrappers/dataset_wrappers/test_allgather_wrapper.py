import unittest

from kappadata.wrappers.dataset_wrappers.allgather_wrapper import AllgatherWrapper
from tests_util.datasets.classification_dataset import ClassificationDataset


class TestAllgatherWrapper(unittest.TestCase):
    def _test(self, world_size, classes, expected_x, expected_classes):
        dataset = AllgatherWrapper(
            dataset=ClassificationDataset(
                x=list(range(len(classes))),
                classes=classes,
            ),
            world_size=world_size,
        )
        actual_x = [dataset.getitem_x(i) for i in range(len(dataset))]
        actual_classes = [dataset.getitem_class(i) for i in range(len(dataset))]
        self.assertEqual(expected_x, actual_x)
        self.assertEqual(expected_classes, actual_classes)

    def test_sorted_balanced_nopadding(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            expected_x=[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
            expected_classes=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_sorted_balanced_padding1(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            expected_x=[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11],
            expected_classes=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        )

    def test_sorted_balanced_padding2(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            expected_x=[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 0, 3, 7],
            expected_classes=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 0, 1],
        )

    def test_sorted_balanced_padding3(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3],
            expected_x=[0, 4, 8, 12, 1, 5, 9, 0, 2, 6, 10, 1, 3],
            expected_classes=[0, 1, 2, 3, 0, 1, 2, 0, 0, 1, 2, 0, 0],
        )
