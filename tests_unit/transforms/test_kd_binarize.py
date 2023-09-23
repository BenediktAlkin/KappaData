import unittest

import torch

from kappadata.transforms.kd_binarize import KDBinarize
from torchvision.transforms.functional import to_pil_image


class TestKDBinarize(unittest.TestCase):
    def test_tensor(self):
        t = KDBinarize()
        x = torch.rand(size=(1, 16, 16), generator=torch.Generator().manual_seed(0))
        y = t(x)
        unique = y.unique()
        self.assertTrue([0, 1], unique.tolist())

    def test_pil(self):
        t = KDBinarize()
        x = to_pil_image(torch.rand(size=(1, 16, 16), generator=torch.Generator().manual_seed(0)))
        y = t(x)
        unique = y.unique()
        self.assertTrue([0, 1], unique.tolist())
