import unittest

import einops
import numpy as np
import torch

from kappadata.transforms.semseg.kd_semseg_resize import KDSemsegResize


class TestKDSemsegResize(unittest.TestCase):
    def test_same_op(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(16, 16), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")
        semseg = semseg.unsqueeze(0)

        t = KDSemsegResize(size=(64, 32), interpolation="nearest")
        t.set_rng(np.random.default_rng(seed=9823))
        x_t, semseg_t = t((x, semseg))
        self.assertTrue(torch.all(x_t * 255 == semseg_t))
