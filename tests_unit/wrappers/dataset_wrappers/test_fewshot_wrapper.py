import unittest

from kappadata.wrappers.dataset_wrappers import FewshotWrapper
from tests_util.datasets import create_class_dataset, ClassDataset


class TestFewshotWrapper(unittest.TestCase):
    def test_less_than_shots(self):
        dataset = ClassDataset(classes=[0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        dataset = FewshotWrapper(dataset=dataset, num_shots=3)
        self.assertEqual(11, len(dataset))

    def test_5shot(self):
        num_classes = 10
        dataset = create_class_dataset(size=1000, n_classes=num_classes, seed=8904)
        dataset = FewshotWrapper(dataset=dataset, num_shots=5)
        self.assertEqual(50, len(dataset))
