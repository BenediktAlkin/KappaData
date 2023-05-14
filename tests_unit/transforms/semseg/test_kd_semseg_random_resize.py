import unittest

import einops
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from kappadata.transforms.semseg.kd_semseg_random_resize import KDSemsegRandomResize


class TestKDSemsegRandomResize(unittest.TestCase):
    def test_same_op(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(16, 16), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")
        semseg = semseg.unsqueeze(0)

        t = KDSemsegRandomResize(base_size=(64, 32), ratio=(0.5, 2.0), interpolation="nearest")
        t.set_rng(np.random.default_rng(seed=9823))
        x_t, semseg_t = t((x, semseg))
        self.assertTrue(torch.all(x_t * 255 == semseg_t))

    def test_pil_semseg(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(16, 16), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")
        semseg = to_pil_image(semseg.byte(), mode="L")

        t = KDSemsegRandomResize(base_size=(64, 32), ratio=(0.5, 2.0), interpolation="nearest")
        t.set_rng(np.random.default_rng(seed=9823))
        x_t, semseg_t = t((x, semseg))
        # nearest interpolation is different for PIL and tensor
        self.assertEqual(576, (x_t == to_tensor(semseg_t)).sum())
        self.assertEqual(414, (x_t != to_tensor(semseg_t)).sum())
