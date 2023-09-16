import unittest

from kappadata.wrappers.dataset_wrappers.class_groups_wrapper import ClassGroupsWrapper
from tests_util.datasets.class_dataset import ClassDataset


class TestClassGroupsWrapper(unittest.TestCase):
    def _test(self, classes, expected, classes_per_group, shuffle, seed=None):
        if shuffle:
            assert seed is not None
        ds = ClassGroupsWrapper(
            dataset=ClassDataset(classes=classes),
            classes_per_group=classes_per_group,
            shuffle=shuffle,
            seed=seed,
        )
        self.assertEqual(expected, [ds.getitem_class(i) for i in range(len(ds))])
        self.assertEqual(expected, ds.getall_class())

    def test_identity(self):
        self._test(
            classes=[0, 1, 2, 3, 0, 1, 2, 3],
            expected=[0, 1, 2, 3, 0, 1, 2, 3],
            classes_per_group=1,
            shuffle=False,
        )

    def test_2groups_noshuffle(self):
        self._test(
            classes=[0, 1, 2, 3, 0, 1, 2, 3],
            expected=[0, 0, 2, 2, 1, 1, 3, 3],
            classes_per_group=2,
            shuffle=False,
        )

    def test_2groups_shuffle(self):
        self._test(
            classes=[0, 1, 2, 3, 0, 1, 2, 3],
            expected=[0, 2, 2, 0, 1, 3, 3, 1],
            classes_per_group=2,
            shuffle=True,
            seed=87,
        )

    def test_undivisible_noshuffle(self):
        self._test(
            classes=[0, 1, 2, 3, 4, 0, 1, 4, 2, 3],
            expected=[0, 0, 2, 2, 4, 1, 1, 5, 3, 3],
            classes_per_group=2,
            shuffle=False,
        )
