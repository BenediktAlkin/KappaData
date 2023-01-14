import torch
import unittest

from kappadata.wrappers.sample_wrappers.multi_view_wrapper import MultiViewConfig, MultiViewWrapper
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
