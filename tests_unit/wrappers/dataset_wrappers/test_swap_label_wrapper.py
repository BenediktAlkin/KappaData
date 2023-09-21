import unittest

from kappadata.wrappers.dataset_wrappers.swap_label_wrapper import SwapLabelWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestSwapLabelWrapper(unittest.TestCase):
    def _test(self, classes, expected_class, expected_apply, p, seed):
        dataset = SwapLabelWrapper(dataset=ClassDataset(classes=classes), p=p, seed=seed)
        actual_classes = [dataset.getitem_class(i) for i in range(len(dataset))]
        actual_applies = [dataset.getitem_apply(i) for i in range(len(dataset))]
        self.assertEqual(expected_class, actual_classes)
        self.assertEqual(expected_apply, actual_applies)
        for actual_cls, actual_apply, expected_cls in zip(actual_classes, actual_applies, expected_class):
            if not actual_apply:
                self.assertEqual(expected_cls, actual_cls)

    def test(self):
        self._test(
            classes=[0, 0, 0, 1, 1, 1, 2, 2],
            expected_class=[0, 1, 1, 2, 1, 1, 2, 2],
            expected_apply=[False, True, True, True, False, False, False, False],
            p=0.5,
            seed=0,
        )
