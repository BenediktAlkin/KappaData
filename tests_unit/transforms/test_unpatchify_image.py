import einops
import torch
import unittest
from kappadata.transforms.unpatchify_image import UnpatchifyImage


class TestUnpatchifyImage(unittest.TestCase):
    def test_square(self):
        #  1  2  3  4
        #  5  6  7  8
        #  9 10 11 12
        # 13 14 15 16
        patches = torch.tensor([[[1, 2, 5, 6], [3, 4, 7, 8]], [[9, 10, 13, 14], [11, 12, 15, 16]]])
        patches = einops.rearrange(patches, "h w d -> d h w")
        expected = torch.arange(16).view(1, 4, 4) + 1
        transform = UnpatchifyImage(patch_size=2)
        actual = transform(patches)
        self.assertTrue(torch.all(expected == actual))