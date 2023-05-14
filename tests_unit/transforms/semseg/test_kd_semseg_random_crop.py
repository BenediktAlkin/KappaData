import einops
import numpy as np
import unittest
from kappadata.transforms.semseg.kd_semseg_random_crop import KDSemsegRandomCrop
import torch
from torchvision.transforms.functional import to_pil_image

class TestKDSemsegRandomCrop(unittest.TestCase):
    def test_default(self):
        rng = torch.Generator().manual_seed(98435)
        x = torch.rand(3, 32, 32, generator=rng)
        semseg = torch.randint(150, size=x.shape[1:], generator=rng)

        t = KDSemsegRandomCrop(size=16)
        t.set_rng(np.random.default_rng(seed=9823))
        x_t, semseg_t = t((x, semseg))
        self.assertEqual((3, 16, 16), x_t.shape)
        self.assertEqual((16, 16), semseg_t.shape)

    def test_same_crop(self):
        rng = torch.Generator().manual_seed(98435)
        semseg = torch.randint(150, size=(32, 32), generator=rng)
        x = einops.rearrange(semseg / 255, "h w -> 1 h w")

        t = KDSemsegRandomCrop(size=16)
        t.set_rng(np.random.default_rng(seed=9823))
        x_t, semseg_t = t((x, semseg))
        self.assertTrue(torch.all(x_t * 255 == semseg_t))

    def test_maxcategoryratio(self):
        rng = torch.Generator().manual_seed(98435)
        x = torch.rand(3, 32, 32, generator=rng)
        semseg = torch.randint(150, size=x.shape[1:], generator=rng)
        replace_with_background = torch.rand(size=semseg.shape, generator=rng) < 0.8
        semseg[replace_with_background] = -1

        t0 = KDSemsegRandomCrop(size=16)
        t1 = KDSemsegRandomCrop(size=16, max_category_ratio=0.7)
        t2 = KDSemsegRandomCrop(size=16, max_category_ratio=0.7, ignore_index=-2)
        for t in [t0, t1, t2]:
            t.set_rng(np.random.default_rng(seed=9823))
        x_t0, semseg_t0 = t0((x, semseg))
        x_t1, semseg_t1 = t1((x, semseg))
        x_t2, _ = t2((x, semseg))
        self.assertTrue(torch.all(x_t0 == x_t1))
        self.assertTrue(torch.all(semseg_t0 == semseg_t1))
        self.assertTrue(torch.all(x_t0 != x_t2))
