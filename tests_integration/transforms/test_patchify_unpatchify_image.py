import unittest

import torch

from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage


class TestPatchifyUnpatchifyImage(unittest.TestCase):
    def test_squareimg_squarepatch(self):
        x = torch.randn(4, 8, 8)
        forward = PatchifyImage(patch_size=2)
        backward = UnpatchifyImage()
        ctx = {}
        self.assertTrue(torch.all(x == backward(forward(x, ctx), ctx)))

    def test_squareimg_rectpatch(self):
        x = torch.randn(4, 8, 8)
        forward = PatchifyImage(patch_size=(4, 2))
        backward = UnpatchifyImage()
        ctx = {}
        self.assertTrue(torch.all(x == backward(forward(x, ctx), ctx)))

    def test_rectimg_squarepatch(self):
        x = torch.randn(4, 8, 16)
        forward = PatchifyImage(patch_size=2)
        backward = UnpatchifyImage()
        ctx = {}
        self.assertTrue(torch.all(x == backward(forward(x, ctx), ctx)))

    def test_rectimg_rectpatch(self):
        x = torch.randn(4, 8, 16)
        forward = PatchifyImage(patch_size=(4, 2))
        backward = UnpatchifyImage()
        ctx = {}
        self.assertTrue(torch.all(x == backward(forward(x, ctx), ctx)))
