import unittest

from kappadata.wrappers.class_filter_wrapper import ClassFilterWrapper
from kappadata.datasets.kd_dataset import KDDataset
from tests_mock.class_dataset import ClassDataset

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