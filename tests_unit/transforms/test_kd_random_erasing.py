import unittest

import numpy as np
import torch
from timm.data.random_erasing import RandomErasing

from kappadata.transforms.kd_random_erasing import KDRandomErasing
from tests_util.patch_rng import patch_rng


class TestKDRandomErasing(unittest.TestCase):
    @patch_rng(fn_names=["random.randint", "random.random", "random.uniform", "torch.Tensor.normal_"])
    def test_equivalent_to_timm(self, seed):
        images = torch.randn(16, 3, 32, 32, generator=torch.Generator().manual_seed(123))
        timm_fn = RandomErasing(probability=0.25, mode="pixel", device="cpu")
        kd_fn = KDRandomErasing(p=0.25, mode="pixelwise").set_rng(np.random.default_rng(seed=seed))

        timm_images = [timm_fn(img) for img in images.clone()]
        kd_images = [kd_fn(img) for img in images.clone()]
        for i, (timm_image, kd_image) in enumerate(zip(timm_images, kd_images)):
            self.assertTrue(torch.all(timm_image == kd_image), f"images are unequal idx={i}")
