import unittest

from kappadata.wrappers.dataset_wrappers.sort_by_class_wrapper import SortByClassWrapper
from tests_util.datasets.classification_dataset import ClassificationDataset


class TestSortByClassWrapper(unittest.TestCase):
    def _test(self, classes, expected_x=None):
        dataset = SortByClassWrapper(
            dataset=ClassificationDataset(
                x=list(range(10)),
                classes=classes,
            ),
        )
        cur_max_cls = 0
        x = [dataset.getitem_x(i) for i in range(len(dataset))]
        sorted_classes = [dataset.getitem_class(i) for i in range(len(dataset))]
        if expected_x is not None:
            self.assertEqual(expected_x, x)
        for cls in sorted_classes:
            self.assertLessEqual(cur_max_cls, cls)
            cur_max_cls = max(cur_max_cls, cls)

    def test_repeated(self):
        self._test(
            classes=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            expected_x=[0, 5, 1, 6, 2, 7, 3, 8, 4, 9],
        )

    def test_missing(self):
        self._test(
            classes=[0, 1, 2, 3, 6, 0, 1, 2, 3, 6],
            expected_x=[0, 5, 1, 6, 2, 7, 3, 8, 4, 9],
        )

    def test_random(self):
        self._test(
            classes=[5, 1, 0, 3, 4,  2, 3, 4, 1, 2],
            expected_x=[2, 1, 8, 5, 9,  3, 6, 4, 7, 0],
        )
