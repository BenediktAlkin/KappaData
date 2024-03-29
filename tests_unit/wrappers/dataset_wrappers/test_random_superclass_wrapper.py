import unittest

from kappadata.wrappers.dataset_wrappers.random_superclass_wrapper import RandomSuperclassWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestRandomSuperclassWrapper(unittest.TestCase):
    def test_identity(self):
        ds = RandomSuperclassWrapper(
            dataset=ClassDataset(classes=[0, 1, 2, 3, 0, 1, 2, 3]),
            classes_per_superclass=1,
            shuffle=False,
        )
        expected = [0, 1, 2, 3, 0, 1, 2, 3]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, ds.getall_class())
        self.assertEqual(4, ds.getshape_class()[0])
        self.assertEqual(4, ds.getdim_class())

    def test_2superclasses(self):
        ds = RandomSuperclassWrapper(
            dataset=ClassDataset(classes=[0, 1, 2, 3, 0, 1, 2, 3]),
            classes_per_superclass=2,
            seed=0,
        )
        expected = [1, 0, 0, 1, 1, 0, 0, 1]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, ds.getall_class())
        self.assertEqual(2, ds.getshape_class()[0])
        self.assertEqual(2, ds.getdim_class())

    def test_2superclasses_2splits(self):
        ds = RandomSuperclassWrapper(
            dataset=ClassDataset(classes=[0, 1, 2, 3, 0, 1, 2, 3]),
            classes_per_superclass=2,
            superclass_splits=2,
            seed=2,
        )
        expected = [1, 1, 0, 2, 3, 3, 2, 0]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, ds.getall_class())
        self.assertEqual(4, ds.getshape_class()[0])
        self.assertEqual(4, ds.getdim_class())

    def test_undivisible_num_class(self):
        ds = RandomSuperclassWrapper(
            dataset=ClassDataset(classes=[0, 1, 2, 3, 5, 0, 1, 2, 3]),
            classes_per_superclass=2,
            seed=0,
        )
        expected = [1, 1, 2, 2, 0, 1, 1, 2, 2]
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, ds.getall_class())
        self.assertEqual(3, ds.getshape_class()[0])
        self.assertEqual(3, ds.getdim_class())