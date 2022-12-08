import unittest
from unittest.mock import patch

import torch
from torchvision.transforms.functional import to_pil_image

from kappadata.transforms.kd_rand_augment import KDRandAugment


class TestKDRandAug(unittest.TestCase):
    def test_doesnt_crash(self):
        kd_ra = KDRandAugment(num_ops=2, magnitude=9, interpolation="nearest")
        x = to_pil_image(torch.randn(3, 32, 32))
        for _ in range(100):
            kd_ra(x)
