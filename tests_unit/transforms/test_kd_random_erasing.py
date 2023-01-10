import unittest
from unittest.mock import patch

import numpy as np
import torch
from timm.data.random_erasing import RandomErasing

from kappadata.transforms.kd_random_erasing import KDRandomErasing


class TestKDRandomErasing(unittest.TestCase):
    @staticmethod
    def _run(run_fn):
        patch_rng = np.random.default_rng(seed=5)
        patch_normal_fn = lambda self: torch.from_numpy(patch_rng.standard_normal(size=self.shape, dtype=np.float32))
        with patch("random.random", lambda: patch_rng.random()):
            with patch("random.uniform", lambda low, high: patch_rng.uniform(low, high)):
                with patch("random.randint", lambda low, high: int(patch_rng.integers(low, high))):
                    with patch("torch.Tensor.normal_", patch_normal_fn):
                        return run_fn()

    @staticmethod
    def _forward(images, fn):
        return [fn(img) for img in images]

    def test_equivalent_to_timm(self):
        images = torch.randn(16, 3, 32, 32, generator=torch.Generator().manual_seed(3))
        timm_fn = RandomErasing(probability=0.25, mode="pixel", device="cpu")
        kd_fn = KDRandomErasing(p=0.25, mode="pixelwise", seed=5)

        timm_images = self._run(lambda: self._forward(images.clone(), timm_fn))
        kd_images = self._forward(images.clone(), kd_fn)
        for i, (timm_image, kd_image) in enumerate(zip(timm_images, kd_images)):
            self.assertTrue(torch.all(timm_image == kd_image), f"images are unequal idx={i}")
