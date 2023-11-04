import unittest

import numpy as np
import torch
from timm.data.auto_augment import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor

from kappadata.transforms.kd_rand_augment import KDRandAugment
from tests_util.patch_rng import patch_rng


class TestKDRandAug(unittest.TestCase):
    @staticmethod
    def create_mae_randaug(magnitude, magnitude_std, interpolation):
        config_str = f"rand-m{magnitude}-mstd{magnitude_std}-inc1"
        img_size = 224
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        )
        if interpolation == "random":
            # defaults to random (bilinear + bicubic) interpolation
            pass
        else:
            # bicubic
            aa_params["interpolation"] = 3

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

    @patch_rng(fn_names=["numpy.random.choice", "random.random", "random.uniform", "random.gauss"])
    def test_equivalent_to_timm_bicubic(self, seed):
        kd_fn = KDRandAugment(
            num_ops=2,
            magnitude=9,
            magnitude_std=0.5,
            interpolation="bicubic",
            fill_color=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        ).set_rng(np.random.default_rng(seed=seed))
        timm_fn = self.create_mae_randaug(magnitude=9, magnitude_std=0.5, interpolation="bicubic")

        timm_images = self._forward(timm_fn)
        kd_images = self._forward(kd_fn)
        for i, (timm_image, kd_image) in enumerate(zip(timm_images, kd_images)):
            self.assertTrue(torch.all(timm_image == kd_image), f"images are unequal idx={i}")

    @patch_rng(fn_names=["numpy.random.choice", "random.choice", "random.random", "random.uniform", "random.gauss"])
    def test_equivalent_to_timm_randominterpolation(self, seed):
        kd_fn = KDRandAugment(
            num_ops=2,
            magnitude=9,
            magnitude_std=0.5,
            interpolation="random",
            fill_color=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        ).set_rng(np.random.default_rng(seed=seed))
        timm_fn = self.create_mae_randaug(magnitude=9, magnitude_std=0.5, interpolation="random")

        timm_images = self._forward(timm_fn)
        kd_images = self._forward(kd_fn)
        for i, (timm_image, kd_image) in enumerate(zip(timm_images, kd_images)):
            self.assertTrue(torch.all(timm_image == kd_image), f"images are unequal idx={i}")
