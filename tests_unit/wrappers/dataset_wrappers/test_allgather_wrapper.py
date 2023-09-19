import unittest

from kappadata.wrappers.dataset_wrappers.allgather_class_wrapper import AllgatherClassWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestAllgatherClassWrapper(unittest.TestCase):
    def _test(self, world_size, classes, expected):
        dataset = AllgatherClassWrapper(
            dataset=ClassDataset(classes=classes),
            world_size=world_size,
        )
        actual_classes = [dataset.getitem_class(i) for i in range(len(dataset))]
        self.assertEqual(expected, actual_classes)

    def test_sorted_balanced_nopadding(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            expected=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_sorted_balanced_padding1(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            expected=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        )

    def test_sorted_balanced_padding2(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            expected=[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 0, 1],
        )

    def test_sorted_balanced_padding3(self):
        self._test(
            world_size=4,
            classes=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3],
            expected=[0, 1, 2, 3, 0, 1, 2, 0, 0, 1, 2, 0, 0],
        )
