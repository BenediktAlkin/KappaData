import unittest

import einops
import torch

from kappadata.transforms.semseg.kd_semseg_overlapped_multi_crop import KDSemsegOverlappedMultiCrop


class TestKDSemsegOverlappedMultiCrop(unittest.TestCase):
    def test(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(16, 16), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")

        t = KDSemsegOverlappedMultiCrop(crop_size=(8, 8))
        x_t, semseg_t = t((x, semseg))
        x_t = x_t.squeeze(0)
        self.assertEqual((9, 1, 8, 8), x_t.shape)
        self.assertEqual((9, 8, 8), semseg_t.shape)
