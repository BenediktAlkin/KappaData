import torch
import unittest

from kappadata import XTransformWrapper, KDRandomHorizontalFlip
from tests_util.datasets import XDataset


class TestXTransformWrapper(unittest.TestCase):
    def test_object_to_transform(self):
        t = XTransformWrapper(
            dataset=XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5))),
            transform=dict(kind="kd_random_horizontal_flip"),
        )
        self.assertIsInstance(t.transform, KDRandomHorizontalFlip)