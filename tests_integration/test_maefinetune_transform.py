import unittest
from unittest.mock import patch

import numpy as np
import torch
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader

from kappadata.collators.kd_mix_collator import KDMixCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.sample_wrappers.label_smoothing_wrapper import LabelSmoothingWrapper
from tests_util.datasets import create_image_classification_dataset
from timm.data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper
from torchvision.transforms.functional import to_pil_image
from kappadata.transforms.kd_random_erasing import KDRandomErasing
from kappadata.transforms.kd_rand_augment import KDRandAugment
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.common.transforms.norm.kd_image_net_norm import KDImageNetNorm
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform

class TestMaeFinetuneTransform(unittest.TestCase):
    def run_test(self, *args, **kwargs):
        rng = np.random.default_rng(seed=5)
        patch_normal_fn = lambda tensor: torch.from_numpy(rng.standard_normal(size=tensor.shape, dtype=np.float32))
        with patch("random.uniform", lambda low, high: rng.uniform(low, high)):
            with patch("random.randint", lambda low, high: int(rng.integers(low, high + 1)) if low != high else low):
                with patch("torch.rand", lambda _: torch.tensor(rng.random(), dtype=torch.float64)):
                    with patch("random.random", lambda: rng.random()):
                        with patch("random.gauss", lambda mu, sigma: rng.normal(mu, sigma)):
                            with patch("numpy.random.choice", rng.choice):
                                with patch("torch.Tensor.normal_", patch_normal_fn):
                                    self._run_test(*args, **kwargs)

    def _run_test(self, images):
        timm_transform = create_transform(
            input_size=32,
            is_training=True,
            color_jitter=None,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        kd_transform = KDComposeTransform([
            KDRandomResizedCrop(size=32, interpolation="bicubic"),
            KDRandomHorizontalFlip(),
            KDRandAugment(
                num_ops=2,
                magnitude=9,
                magnitude_std=0.5,
                fill_color=[124, 116, 104],
                interpolation="bicubic",
            ),
            KDImageNetNorm(),
            KDRandomErasing(p=0.25, mode="pixelwise", max_count=1),
        ])
        kd_transform.set_rng(np.random.default_rng(seed=5))

        for i in range(len(images)):
            x = to_pil_image(images[i].clone())
            timm_image = timm_transform(x)
            x = to_pil_image(images[i].clone())
            kd_image = kd_transform(x)
            self.assertEqual(timm_image.shape, kd_image.shape)
            self.assertTrue(torch.allclose(timm_image, kd_image), f"image is unequal for i={i}")

    def test(self):
        images = torch.rand(100, 3, 32, 32, generator=torch.Generator().manual_seed(513))
        self.run_test(images)
