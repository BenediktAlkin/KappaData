import unittest

import torch

from kappadata.transforms import (
    KDSemsegRandomHorizontalFlip,
    KDSemsegPad,
    KDSemsegRandomResize,
    KDSemsegRandomCrop,
    KDColorJitter,
    KDImageRangeNorm,
)
from kappadata.wrappers import SemsegTransformWrapper, ModeWrapper
from tests_util.datasets.semseg_dataset import SemsegDataset


class TestSemsegDataloading(unittest.TestCase):
    def test_ade20k_beit_pipeline(self):
        rng = torch.Generator().manual_seed(98435)
        x = [
            torch.rand(1, 329, 592, generator=rng),
            torch.rand(1, 592, 329, generator=rng),
            torch.rand(1, 64, 91, generator=rng),
            torch.rand(1, 4108, 91, generator=rng),
            torch.rand(1, 91, 4108, generator=rng),
            torch.rand(1, 4310, 4108, generator=rng),
        ]
        semseg = [torch.randint(150, size=xx.shape[1:], generator=rng) for xx in x]
        ds = SemsegTransformWrapper(
            dataset=SemsegDataset(x, semseg),
            transforms=[
                KDSemsegRandomResize(base_size=(2048, 512), ratio=(0.5, 2.0)),
                KDSemsegRandomCrop(size=512, max_category_ratio=0.75),
                KDSemsegRandomHorizontalFlip(),
                KDColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                KDImageRangeNorm(),
                KDSemsegPad(size=512),
            ],
            seed=943,
        )

        for x, semseg in ModeWrapper(dataset=ds, mode="x semseg", return_ctx=False):
            self.assertEqual((1, 512, 512), x.shape)
            self.assertEqual((512, 512), semseg.shape)
            self.assertLess(semseg.max(), 150)
            self.assertGreaterEqual(semseg.min(), -1)
