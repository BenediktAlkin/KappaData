import unittest

import torch

from kappadata.transforms import AddGaussianNoiseTransform
from kappadata.wrappers.sample_wrappers.kd_multi_view_wrapper import KDMultiViewWrapper
from tests_util.datasets.x_dataset import XDataset


class TestKDMultiViewWrapper(unittest.TestCase):
    def test_2views(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[1, (1, lambda x: -x)],
        )
        for i in range(len(ds)):
            sample = ds.getitem_x(i)
            self.assertIsInstance(sample, list)
            self.assertEqual(data[i].item(), sample[0].item())
            self.assertEqual((-data[i]).item(), sample[1].item())

    def test_2views_seed(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[AddGaussianNoiseTransform(magnitude=0.5), AddGaussianNoiseTransform(magnitude=0.2)],
            seed=3,
        )
        for i in range(len(ds)):
            sample0 = ds.getitem_x(i)
            sample1 = ds.getitem_x(i)
            self.assertIsInstance(sample0, list)
            self.assertIsInstance(sample1, list)
            self.assertEqual(len(sample0), len(sample1))
            for j in range(len(sample0)):
                self.assertEqual(sample0[j].tolist(), sample1[j].tolist())

    def test_2views_identity(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[2],
        )
        for i in range(len(ds)):
            sample0 = ds.getitem_x(i)
            sample1 = ds.getitem_x(i)
            self.assertIsInstance(sample0, list)
            self.assertIsInstance(sample1, list)
            self.assertEqual(2, len(sample0))
            self.assertEqual(2, len(sample1))
            for j in range(len(sample0)):
                self.assertEqual(sample0[j].tolist(), sample1[j].tolist())

    def test_2views_identity_dict(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = KDMultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[dict(n_views=2)],
        )
        for i in range(len(ds)):
            sample0 = ds.getitem_x(i)
            sample1 = ds.getitem_x(i)
            self.assertIsInstance(sample0, list)
            self.assertIsInstance(sample1, list)
            self.assertEqual(2, len(sample0))
            self.assertEqual(2, len(sample1))
            for j in range(len(sample0)):
                self.assertEqual(sample0[j].tolist(), sample1[j].tolist())