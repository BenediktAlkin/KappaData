import unittest

from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_gaussian_blur_tv import KDRandomGaussianBlurTV
from kappadata.transforms.kd_random_solarize import KDRandomSolarize
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.datasets.x_dataset import XDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
import torch

class TestPopulateContext(unittest.TestCase):
    def _test(self, ctx_key, transform, expected_skip_counts, ctx_is_empty_fn=None):
        ctx_is_empty_fn = ctx_is_empty_fn or (lambda ctx_item: ctx_item == -1)
        transform.rng = np.random.default_rng(seed=5)
        ds = XDataset(x=torch.randn(10, 3, 4, 4), transform=KDComposeTransform([ToPILImage(), transform, ToTensor()]))
        loader = DataLoader(ModeWrapper(dataset=ds, mode="x", return_ctx=True), batch_size=len(ds))
        for expected_skip_count in expected_skip_counts:
            for _, ctx in loader:
                skip_count = sum(ctx_is_empty_fn(ctx_item) for ctx_item in ctx[ctx_key])
                self.assertTrue(expected_skip_count, skip_count.item())

    def test_random_solarize(self):
        self._test(
            ctx_key="KDRandomSolarize.threshold",
            transform=KDRandomSolarize(p=0.5, threshold=128),
            expected_skip_counts=[4, 5, 6, 4],
        )

    def test_random_gaussian_blur_pil(self):
        self._test(
            ctx_key="KDRandomGaussianBlurPIL.sigma",
            transform=KDRandomGaussianBlurPIL(p=0.2, sigma=(0.1, 2.0)),
            expected_skip_counts=[7, 9],
        )

    def test_random_gaussian_blur_tv(self):
        self._test(
            ctx_key="KDRandomGaussianBlurTV.sigma",
            transform=KDRandomGaussianBlurTV(p=0.4, kernel_size=7, sigma=(0.1, 2.0)),
            expected_skip_counts=[5, 7],
        )

    def test_random_color_jitter(self):
        self._test(
            ctx_key="KDRandomColorJitter.saturation",
            transform=KDRandomColorJitter(p=0.3, brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1),
            expected_skip_counts=[6, 8],
        )

    def test_random_color_jitter_novalue(self):
        self._test(
            ctx_key="KDRandomColorJitter.saturation",
            transform=KDRandomColorJitter(p=0.3, brightness=0.2, contrast=0.3, saturation=0, hue=0.1),
            expected_skip_counts=[10, 10],
        )