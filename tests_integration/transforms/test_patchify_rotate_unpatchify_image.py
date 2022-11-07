import unittest

import torch

from kappadata.transforms.patchify_image import PatchifyImage
from kappadata.transforms.patchwise_random_rotation import PatchwiseRandomRotation
from kappadata.transforms.unpatchify_image import UnpatchifyImage


class TestPatchifyRotateUnpatchifyImage(unittest.TestCase):
    def test(self):
        #  1  2  3  4
        #  5  6  7  8
        #  9 10 11 12
        # 13 14 15 16
        source = torch.arange(16).view(1, 4, 4) + 1
        # seed=6 -> rotations=[2, 1, 3, 0]
        # top_left:  [ 1,  2] -> [ 6,  5]
        #            [ 5,  6] -> [ 2,  1]
        # top_right: [ 3,  4] -> [ 4,  8]
        #            [ 7,  8] -> [ 3,  7]
        # bot_left:  [ 9, 10] -> [13,  9]
        #            [13, 14] -> [14, 10]
        # bot_right: [11, 12]
        #            [15, 16]
        target = torch.tensor([6, 5, 4, 8, 2, 1, 3, 7, 13, 9, 11, 12, 14, 10, 15, 16]).view(1, 4, 4)
        forward = PatchifyImage(patch_size=2)
        rotate = PatchwiseRandomRotation(seed=6)
        backward = UnpatchifyImage()
        ctx = {}
        self.assertTrue(torch.all(target == backward(rotate(forward(source, ctx), ctx), ctx)))
