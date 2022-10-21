import unittest

from kappadata.wrappers.class_filter_wrapper import ClassFilterWrapper
from kappadata.wrappers.base.dataset_base import DatasetBase
from tests_mock.index_dataset import IndexDataset

class TestClassFilterWrapper(unittest.TestCase):
    class ClassDataset(DatasetBase):
        def __init__(self, classes):
            super().__init__()
            self.classes = classes

        def idxget_class(self, idx):
            return self.classes[idx]

        def __len__(self):
            return len(self.classes)


    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: ClassFilterWrapper(None, valid_classes=[1], invalid_classes=[3]))
        self.assertRaises(AssertionError, lambda: ClassFilterWrapper(None, valid_classes=1))
        self.assertRaises(AssertionError, lambda: ClassFilterWrapper(None, invalid_classes=3))
        _ = ClassFilterWrapper(IndexDataset(size=3), valid_classes=[1])
        _ = ClassFilterWrapper(IndexDataset(size=3), invalid_classes=[3])

    def test_valid_classes(self):
        ds = ClassFilterWrapper(self.ClassDataset(classes=[0, 1, 1, 1, 5, 3, 1, 0, 3, 2, 1]), valid_classes=[1])
        self.assertEqual(5, len(ds))
        self.assertEqual([1] * 5, [ds.idxget_class(i) for i in range(len(ds))])

    def test_invalid_classes(self):
        ds = ClassFilterWrapper(self.ClassDataset(classes=[0, 1, 1, 1, 5, 3, 1, 0, 3, 2, 1]), invalid_classes=[1])
        self.assertEqual(6, len(ds))
        self.assertEqual([0, 5, 3, 0, 3, 2], [ds.idxget_class(i) for i in range(len(ds))])