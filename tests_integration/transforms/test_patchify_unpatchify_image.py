import einops
import torch
import unittest
from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.unpatchify_image import UnpatchifyImage


class TestPatchifyUnpatchifyImage(unittest.TestCase):
    def test_squareimg_squarepatch(self):
        x = torch.randn(4, 8, 8)
        forward = PatchifyImage(patch_size=2)
        backward = UnpatchifyImage(patch_size=2)
        self.assertTrue(torch.all(x == backward(forward(x))))

    def test_squareimg_rectpatch(self):
        x = torch.randn(4, 8, 8)
        forward = PatchifyImage(patch_size=(4, 2))
        backward = UnpatchifyImage(patch_size=(4, 2))
        self.assertTrue(torch.all(x == backward(forward(x))))

    def test_rectimg_squarepatch(self):
        x = torch.randn(4, 8, 16)
        forward = PatchifyImage(patch_size=2)
        backward = UnpatchifyImage(patch_size=2)
        self.assertTrue(torch.all(x == backward(forward(x))))

    def test_rectimg_rectpatch(self):
        x = torch.randn(4, 8, 16)
        forward = PatchifyImage(patch_size=(4, 2))
        backward = UnpatchifyImage(patch_size=(4, 2))
        self.assertTrue(torch.all(x == backward(forward(x))))