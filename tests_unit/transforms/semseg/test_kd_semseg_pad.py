import unittest

import einops
import torch

from kappadata.transforms.semseg.kd_semseg_pad import KDSemsegPad


class TestKDSemsegPad(unittest.TestCase):
    def test_same_padding(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(16, 16), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")

        t = KDSemsegPad(size=(17, 18))
        x_t, semseg_t = t((x, semseg))
        x_t = x_t.squeeze(0)
        self.assertTrue(torch.all((x_t == 0.).nonzero() == (semseg_t == -1).nonzero()))
        self.assertTrue(torch.all(x_t[x_t != 0.] * 255 == semseg_t[semseg_t != -1]))

    def test_uneven_padding(self):
        x = torch.ones(1, 16, 16)
        semseg = torch.ones(16, 16)

        t = KDSemsegPad(size=(17, 18))
        x_t, semseg_t = t((x, semseg))
        self.assertEqual((1, 17, 18), x_t.shape)
        self.assertEqual((17, 18), semseg_t.shape)
