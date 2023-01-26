import numpy as np
import unittest

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_color_jitter import KDColorJitter
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_gaussian_blur_tv import KDGaussianBlurTV
from kappadata.transforms.kd_rand_augment import KDRandAugment
from kappadata.transforms.kd_rand_augment_custom import KDRandAugmentCustom
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.kd_random_crop import KDRandomCrop
from kappadata.transforms.kd_random_erasing import KDRandomErasing
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_gaussian_blur_tv import KDRandomGaussianBlurTV
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_solarize import KDRandomSolarize


class TestIsDeterministic(unittest.TestCase):
    @staticmethod
    def _create_images(as_tensor):
        images = torch.rand(8, 3, 32, 32, generator=torch.Generator().manual_seed(9))
        if not as_tensor:
            images = [to_pil_image(image) for image in images]
        return images

    def _run(self, transform, as_tensor):
        self._run_single(transform, as_tensor=as_tensor)
        compose = KDComposeTransform([transform])
        self._run_single(compose, as_tensor=as_tensor)

    def _run_single(self, transform, as_tensor=False):
        images = self._create_images(as_tensor=as_tensor)
        transformed_history = []
        contexts_history = []
        for _ in range(3):
            transform.set_rng(np.random.default_rng(5))
            transformed = []
            contexts = []
            for image in images:
                ctx = {}
                image = transform(image, ctx=ctx)
                if not torch.is_tensor(image):
                    image = to_tensor(image)
                transformed.append(image)
                contexts.append(ctx)
            transformed = torch.stack(transformed)
            transformed_history.append(transformed)
            contexts_history.append(contexts)
            self.assertTrue(torch.all(transformed_history[0] == transformed))
            self.assertTrue(contexts_history[0] == contexts)

    def test_color_jitter(self):
        self._run(KDColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), as_tensor=False)
        self._run(KDColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), as_tensor=True)

    def test_gaussian_blur_pil(self):
        self._run(KDGaussianBlurPIL(sigma=(0.1, 2.0)), as_tensor=False)

    def test_gaussian_blur_tv(self):
        self._run(KDGaussianBlurTV(sigma=(0.1, 2.0), kernel_size=23), as_tensor=False)
        self._run(KDGaussianBlurTV(sigma=(0.1, 2.0), kernel_size=23), as_tensor=True)

    def test_rand_augment(self):
        t = KDRandAugment(num_ops=2, magnitude=9, fill_color=(128, 128, 128), interpolation="bicubic")
        self._run(t, as_tensor=False)

    def test_rand_augment_custom(self):
        t = KDRandAugmentCustom(num_ops=2, magnitude=9, fill_color=(128, 128, 128), interpolation="bicubic")
        self._run(t, as_tensor=False)

    def test_random_color_jitter(self):
        t0 = KDRandomColorJitter(p=0.3, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        self._run(t0, as_tensor=False)
        t1 = KDRandomColorJitter(p=0.3, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        self._run(t1, as_tensor=True)

    def test_random_crop(self):
        self._run(KDRandomCrop(size=32, padding=4), as_tensor=False)
        self._run(KDRandomCrop(size=32, padding=4), as_tensor=True)

    def test_random_erasing(self):
        self._run(KDRandomErasing(p=0.25), as_tensor=True)

    def test_random_gaussian_blur_pil(self):
        self._run(KDRandomGaussianBlurPIL(p=0.25, sigma=(0.1, 2.0)), as_tensor=False)

    def test_random_gaussian_blur_tv(self):
        self._run(KDRandomGaussianBlurTV(p=0.25, sigma=(0.1, 2.0), kernel_size=23), as_tensor=False)
        self._run(KDRandomGaussianBlurTV(p=0.25, sigma=(0.1, 2.0), kernel_size=23), as_tensor=True)

    def test_random_grayscale(self):
        self._run(KDRandomGrayscale(p=0.25), as_tensor=False)
        self._run(KDRandomGrayscale(p=0.25), as_tensor=True)

    def test_random_horizontal_flip(self):
        self._run(KDRandomHorizontalFlip(p=0.25), as_tensor=False)
        self._run(KDRandomHorizontalFlip(p=0.25), as_tensor=True)

    def test_random_resized_crop(self):
        self._run(KDRandomResizedCrop(size=32), as_tensor=False)
        self._run(KDRandomResizedCrop(size=32), as_tensor=True)

    def test_random_solarize(self):
        self._run(KDRandomSolarize(p=0.25, threshold=128), as_tensor=False)
        self._run(KDRandomSolarize(p=0.25, threshold=0.5), as_tensor=True)
