import unittest

import torch

from kappadata.transforms import AddGaussianNoiseTransform
from kappadata.wrappers.sample_wrappers import XTransformWrapper
from tests_util.datasets.x_dataset import XDataset
import numpy as np

class TestXTransformWrapper(unittest.TestCase):
    def test_seed(self):
        ds = XDataset(x=torch.randn(10, 5, generator=torch.Generator().manual_seed(5)))
        ds = XTransformWrapper(dataset=ds, transform=AddGaussianNoiseTransform(), seed=5)
        for i in range(len(ds)):
            self.assertNotEqual(ds.x[0].tolist(), ds.getitem_x(i).tolist())
            self.assertEqual(ds.getitem_x(i).tolist(), ds.getitem_x(i).tolist())

    def test_noseed(self):
        ds = XDataset(x=torch.randn(10, 5, generator=torch.Generator().manual_seed(5)))
        transform = AddGaussianNoiseTransform().set_rng(np.random.default_rng(seed=5))
        ds = XTransformWrapper(dataset=ds, transform=transform)
        for i in range(len(ds)):
            self.assertNotEqual(ds.getitem_x(i).tolist(), ds.getitem_x(i).tolist())

