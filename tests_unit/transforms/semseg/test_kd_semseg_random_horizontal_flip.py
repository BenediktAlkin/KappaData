import unittest

import einops
import numpy as np
import torch

from kappadata.transforms.semseg.kd_semseg_random_horizontal_flip import KDSemsegRandomHorizontalFlip


class TestKDSemsegRandomHorizontalFlip(unittest.TestCase):
    def test(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(32, 32), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")

        t = KDSemsegRandomHorizontalFlip()
        t.set_rng(np.random.default_rng(seed=9823))
        for _ in range(10):
            x_t, semseg_t = t((x, semseg))
            self.assertTrue(torch.all(x_t * 255 == semseg_t))
