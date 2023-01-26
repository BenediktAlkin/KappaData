import unittest
from unittest.mock import patch

import numpy as np
import torch
from timm.data.auto_augment import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor

from kappadata.transforms.kd_rand_augment import KDRandAugment


class TestKDRandAug(unittest.TestCase):
    @staticmethod
    def _run(run_fn):
        patch_rng = np.random.default_rng(seed=5)
        with patch("numpy.random.choice", patch_rng.choice):
            with patch("random.random", lambda: patch_rng.random()):
                with patch("random.uniform", lambda low, high: patch_rng.uniform(low, high)):
                    with patch("random.gauss", lambda mu, sigma: patch_rng.normal(mu, sigma)):
                        return run_fn()

    @staticmethod
    def create_mae_randaug(magnitude, magnitude_std):
        config_str = f"rand-m{magnitude}-mstd{magnitude_std}-inc1"
        img_size = 224
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
            interpolation=3,  # bicubic
        )
        return rand_augment_transform(config_str=config_str, hparams=aa_params)

    @staticmethod
    def _forward(fn):
        images = [
            to_pil_image(tensor)
            for tensor in torch.randn(100, 3, 224, 224, generator=torch.Generator().manual_seed(52))
        ]
        # results = []
        # for i, img in enumerate(images):
        #     results.append(to_tensor(fn(img)))
        # return results
        return [to_tensor(fn(img)) for img in images]

    def test_equivalent_to_timm(self):
        kd_fn = KDRandAugment(
            num_ops=2,
            magnitude=9,
            magnitude_std=0.5,
            interpolation="bicubic",
            fill_color=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        ).set_rng(np.random.default_rng(seed=5))
        timm_fn = self.create_mae_randaug(magnitude=9, magnitude_std=0.5)

        timm_images = self._run(lambda: self._forward(timm_fn))
        kd_images = self._forward(kd_fn)
        for i, (timm_image, kd_image) in enumerate(zip(timm_images, kd_images)):
            self.assertTrue(torch.all(timm_image == kd_image), f"images are unequal idx={i}")
