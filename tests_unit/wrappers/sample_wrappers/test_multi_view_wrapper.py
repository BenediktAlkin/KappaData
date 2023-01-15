import torch
import unittest

from kappadata.wrappers.sample_wrappers.multi_view_wrapper import MultiViewConfig, MultiViewWrapper
from kappadata.transforms import AddGaussianNoiseTransform
from tests_util.datasets.x_dataset import XDataset

class TestMultiViewWrapper(unittest.TestCase):
    def test_2views(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = MultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[(1, lambda x: x), (1, lambda x: -x)],
        )
        for i in range(len(ds)):
            sample = ds.getitem_x(i)
            self.assertIsInstance(sample, list)
            self.assertEqual(data[i].item(), sample[0].item())
            self.assertEqual((-data[i]).item(), sample[1].item())

    def test_2views_seed(self):
        data = torch.randn(10, generator=torch.Generator().manual_seed(5))
        ds = MultiViewWrapper(
            dataset=XDataset(x=data),
            configs=[(1, AddGaussianNoiseTransform(magnitude=0.5)), (1, AddGaussianNoiseTransform(magnitude=0.2))],
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
