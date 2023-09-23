import unittest
from tests_util.datasets import ClassificationDataset
from kappadata.samplers import ClassBalancedSampler
from kappadata.wrappers import ClassGroupsWrapper


class TestBalancedClassGroupSampler(unittest.TestCase):
    def _test(self, classes, expected, classes_per_group, samples_per_class):
        dataset = ClassGroupsWrapper(
            dataset=ClassificationDataset(
                x=list(range(len(classes))),
                classes=classes,
            ),
            classes_per_group=classes_per_group,
            shuffle=False,
        )
        sampler = ClassBalancedSampler(
            dataset=dataset,
            shuffle=False,
            samples_per_class=samples_per_class,
            getall_item="class_before_grouping",
        )
        idxs = [i for i in sampler]
        self.assertEqual(expected, idxs)

    def test_balanced(self):
        classes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        self._test(
            classes=classes,
            expected=list(range(len(classes))),
            classes_per_group=2,
            samples_per_class=2,
        )

    def test_unbalanced1(self):
        self._test(
            classes=[0, 0, 0, 0, 1],
            expected=[0, 1, 4, 4],
            classes_per_group=2,
            samples_per_class=2,
        )

    def test_unbalanced2(self):
        self._test(
            classes=[0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3],
            expected=[0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 5, 6, 7, 8, 9, 10],
            classes_per_group=2,
            samples_per_class=4,
        )