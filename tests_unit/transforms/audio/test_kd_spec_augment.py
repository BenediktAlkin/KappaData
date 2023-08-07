import numpy as np
import unittest

import torch

from kappadata.transforms.audio import KDSpecAugment
from tests_util.patch_rng import patch_rng
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose

class TestKDSpecAugment(unittest.TestCase):
    def _test_equal_to_torchaudio(self, kd_transform, ta_transform):
        # init data/seed
        x = torch.randn(1, 128, 512, generator=torch.Generator().manual_seed(438))
        seed = 4389
        kd_transform.set_rng(np.random.default_rng(seed=seed))

        # apply transform
        kd_y = kd_transform(x)
        with patch_rng(fn_names=["torch.rand"], seed=seed):
            ta_y = ta_transform(x)

        self.assertTrue(torch.all(kd_y == ta_y))

    def test_equal_to_torchaudio_frequencymasking(self):
        self._test_equal_to_torchaudio(
            kd_transform=KDSpecAugment(frequency_masking=48),
            ta_transform=FrequencyMasking(freq_mask_param=48),
        )

    def test_equal_to_torchaudio_timemasking(self):
        self._test_equal_to_torchaudio(
            kd_transform=KDSpecAugment(time_masking=192),
            ta_transform=TimeMasking(time_mask_param=192),
        )

    def test_equal_to_torchaudio(self):
        self._test_equal_to_torchaudio(
            kd_transform=KDSpecAugment(frequency_masking=48, time_masking=192),
            ta_transform=Compose([
                FrequencyMasking(freq_mask_param=48),
                TimeMasking(time_mask_param=192),
            ]),
        )