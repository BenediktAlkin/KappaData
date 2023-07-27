import unittest
import torch
from tests_util.datasets import XDataset
from kappadata.common.wrappers import ByolMultiViewWrapper
from torch.utils.data import DataLoader
from kappadata.wrappers import ModeWrapper
from torchvision.transforms.functional import to_pil_image

class TestDeterministicByolaug(unittest.TestCase):
    def test(self):
        x = [to_pil_image(xx) for xx in torch.randn(20, 3, 32, 32, generator=torch.Generator().manual_seed(39))]
        dataset = XDataset(x=x)
        dataset = ByolMultiViewWrapper(dataset=dataset, seed=0)
        dataset = ModeWrapper(dataset=dataset, mode="x")

        iter0_workers0 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=4)])
        iter1_workers0 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=2)])
        self.assertTrue(torch.all(iter0_workers0 == iter1_workers0))

        iter0_workers2 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=4, num_workers=2)])
        self.assertTrue(torch.all(iter0_workers0 == iter0_workers2))
        iter1_workers2 = torch.concat([torch.stack(x, dim=1) for x in DataLoader(dataset, batch_size=2, num_workers=2)])
        self.assertTrue(torch.all(iter0_workers0 == iter1_workers2))

