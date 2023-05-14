import unittest

import einops

from kappadata.wrappers.sample_wrappers.semseg_transform_wrapper import SemsegTransformWrapper
from tests_util.datasets.semseg_dataset import SemsegDataset
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from kappadata.transforms.kd_random_crop import KDRandomCrop
from kappadata.transforms.norm.kd_image_range_norm import KDImageRangeNorm
from kappadata.wrappers.mode_wrapper import ModeWrapper

class TestSemsegTransformWrapper(unittest.TestCase):
    def test_shared_transform(self):
        rng = torch.Generator().manual_seed(84432)
        data = torch.randint(150, size=(10, 32, 32), generator=rng)
        ds = SemsegTransformWrapper(
            dataset=SemsegDataset(
                x=[to_pil_image(einops.repeat(data[i] / 255, "h w -> three h w", three=3)) for i in range(len(data))],
                semseg=data,
            ),
            shared_transform=KDRandomCrop(size=16),
        )
        for x, semseg in ModeWrapper(dataset=ds, mode="x semseg"):
            self.assertTrue(torch.all(to_tensor(x) * 255 == semseg))



