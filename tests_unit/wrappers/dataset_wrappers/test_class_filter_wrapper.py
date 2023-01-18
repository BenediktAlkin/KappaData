import unittest

from kappadata.wrappers.dataset_wrappers.class_filter_wrapper import ClassFilterWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestClassFilterWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: ClassFilterWrapper(None, valid_classes=[1], invalid_classes=[3]))
        self.assertRaises(AssertionError, lambda: ClassFilterWrapper(None, valid_classes=1))
        self.assertRaises(AssertionError, lambda: ClassFilterWrapper(None, invalid_classes=3))
        _ = ClassFilterWrapper(ClassDataset(classes=list(range(4))), valid_classes=[1])
        _ = ClassFilterWrapper(ClassDataset(classes=list(range(4))), invalid_classes=[3])

    def test_valid_classes(self):
        ds = ClassFilterWrapper(ClassDataset(classes=[0, 1, 1, 1, 5, 3, 1, 0, 3, 2, 1]), valid_classes=[1])
        self.assertEqual(5, len(ds))
        self.assertEqual([1] * 5, [ds.getitem_class(i) for i in range(len(ds))])

    def test_invalid_classes(self):
        ds = ClassFilterWrapper(ClassDataset(classes=[0, 1, 1, 1, 5, 3, 1, 0, 3, 2, 1]), invalid_classes=[1])
        self.assertEqual(6, len(ds))
        self.assertEqual([0, 5, 3, 0, 3, 2], [ds.getitem_class(i) for i in range(len(ds))])

    def test_valid_class_names(self):
        ds = ClassDataset(
            classes=[0, 1, 1, 1, 5, 3, 1, 0, 3, 2, 1],
            class_names=["zero", "one", "two", "three", "four", "five"],
        )
        ds = ClassFilterWrapper(ds, valid_class_names=["one", "five"])
        self.assertEqual(6, len(ds))
        self.assertEqual([1, 1, 1, 5, 1, 1], [ds.getitem_class(i) for i in range(len(ds))])

    def test_invalid_class_names(self):
        ds = ClassDataset(
            classes=[0, 1, 1, 1, 5, 3, 1, 0, 3, 2, 1],
            class_names=["zero", "one", "two", "three", "four", "five"],
        )
        ds = ClassFilterWrapper(ds, invalid_class_names=["one", "five"])
        self.assertEqual(5, len(ds))
        self.assertEqual([0, 3, 0, 3, 2], [ds.getitem_class(i) for i in range(len(ds))])
