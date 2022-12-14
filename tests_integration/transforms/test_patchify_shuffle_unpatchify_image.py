import unittest

import torch

from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.patchwise_shuffle import PatchwiseShuffle
from kappadata.transforms.unpatchify_image import UnpatchifyImage


class TestPatchifyShuffleUnpatchifyImage(unittest.TestCase):
    def test(self):
        #  1  2  3  4
        #  5  6  7  8
        #  9 10 11 12
        # 13 14 15 16
        source = torch.arange(16).view(1, 4, 4) + 1
        # seed=235 -> permutation=[1, 2, 3, 0]
        #  3  4  9 10
        #  7  8 13 14
        # 11 12  1  2
        # 15 16  5  6
        target = torch.tensor([3, 4, 9, 10, 7, 8, 13, 14, 11, 12, 1, 2, 15, 16, 5, 6]).view(1, 4, 4)
        forward = PatchifyImage(patch_size=2)

        rotate = PatchwiseShuffle(seed=235)
        backward = UnpatchifyImage()
        ctx = {}
        rotated = rotate(forward(source, ctx), ctx)
        actual = backward(rotated, ctx)
        self.assertTrue(torch.all(target == actual))
