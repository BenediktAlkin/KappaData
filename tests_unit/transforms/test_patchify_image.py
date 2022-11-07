import unittest

import einops
import torch

from kappadata.transforms.patchify_image import PatchifyImage


class TestPatchifyImage(unittest.TestCase):
    def test_square(self):
        #  1  2  3  4
        #  5  6  7  8
        #  9 10 11 12
        # 13 14 15 16
        img = torch.arange(16).view(1, 4, 4) + 1
        expected = torch.tensor([[[1, 2, 5, 6], [3, 4, 7, 8]], [[9, 10, 13, 14], [11, 12, 15, 16]]])
        expected = einops.rearrange(expected, "h w (ph pw) -> 1 (h w) ph pw", ph=2, pw=2)

        transform = PatchifyImage(patch_size=2)
        patches = transform(img)
        self.assertTrue(torch.all(patches == expected))
