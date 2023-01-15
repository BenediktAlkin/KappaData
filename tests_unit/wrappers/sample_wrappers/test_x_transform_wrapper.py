import unittest

import torch

from kappadata.wrappers.sample_wrappers import XTransformWrapper
from tests_util.datasets.x_dataset import XDataset
from kappadata.transforms import AddGaussianNoiseTransform


class TestXTransformWrapper(unittest.TestCase):
    def test_seed(self):
        ds = XDataset(x=torch.randn(10, 5, generator=torch.Generator().manual_seed(5)))
        ds = XTransformWrapper(dataset=ds, transform=AddGaussianNoiseTransform(), seed=5)
        for i in range(len(ds)):
            self.assertNotEqual(ds.x[0].tolist(), ds.getitem_x(i).tolist())
            self.assertEqual(ds.getitem_x(i).tolist(), ds.getitem_x(i).tolist())

    def test_noseed(self):
        ds = XDataset(x=torch.randn(10, 5, generator=torch.Generator().manual_seed(5)))
        ds = XTransformWrapper(dataset=ds, transform=AddGaussianNoiseTransform(seed=5))
        for i in range(len(ds)):
            self.assertNotEqual(ds.getitem_x(i).tolist(), ds.getitem_x(i).tolist())

    def test_transform_already_has_seed(self):
        ds = XDataset(x=torch.randn(10, 5, generator=torch.Generator().manual_seed(5)))
        ds = XTransformWrapper(dataset=ds, transform=AddGaussianNoiseTransform(seed=5), seed=4)
        with self.assertRaises(AssertionError) as ex:
            ds.getitem_x(0)
        self.assertEqual("can't use set_rng on transforms with seed", str(ex.exception))