import unittest

import einops
import torch
from torch import nn
from kappadata.transforms import PatchwiseTransform, Identity, KDGaussianBlurPIL


class TestPatchwiseTransform(unittest.TestCase):
    def test_identity(self):
        transform = PatchwiseTransform(patch_size=16, transform=Identity())
        x = torch.rand(3, 32, 48, generator=torch.Generator().manual_seed(452098))
        y = transform(x.clone())
        self.assertTrue(torch.all(x == y))

    def test_gaussian_blur_pil(self):
        transform = PatchwiseTransform(patch_size=16, transform=KDGaussianBlurPIL(sigma=[1, 1]))
        x = torch.rand(3, 32, 48, generator=torch.Generator().manual_seed(452098))
        y = transform(x.clone())
        self.assertTrue(torch.all(x != y))
