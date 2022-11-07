import unittest

import einops
import torch

from kappadata.transforms.unpatchify_image import UnpatchifyImage


class TestUnpatchifyImage(unittest.TestCase):
    def test_square(self):
        #  1  2  3  4
        #  5  6  7  8
        #  9 10 11 12
        # 13 14 15 16
        patches = torch.tensor([[[1, 2, 5, 6], [3, 4, 7, 8]], [[9, 10, 13, 14], [11, 12, 15, 16]]])
        patches = einops.rearrange(patches, "h w (ph pw) -> 1 (h w) ph pw", ph=2, pw=2)
        expected = torch.arange(16).view(1, 4, 4) + 1
        transform = UnpatchifyImage()
        actual = transform(patches, dict(patchify_lh=2, patchify_lw=2))
        self.assertTrue(torch.all(expected == actual))
