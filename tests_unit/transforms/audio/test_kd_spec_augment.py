import einops
import numpy as np
import unittest

import torch

from kappadata.transforms.audio import KDSpecAugment
from tests_util.patch_rng import patch_rng
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose
from tests_util.mock_rng import MockRng

class TestKDSpecAugment(unittest.TestCase):
    def _test_equal_to_torchaudio(self, kd_transform, ta_transform):
        # init data/seed
        ta_x = torch.randn(1, 128, 512, generator=torch.Generator().manual_seed(438))
        seed = 4389
        kd_transform.set_rng(np.random.default_rng(seed=seed))

        # apply transform (KDSpecAugment expects TIMExFREQ instead of FREQxTIME)
        kd_x = einops.rearrange(ta_x, "1 freq time -> 1 time freq")
        kd_y = kd_transform(kd_x)
        with patch_rng(fn_names=["torch.rand"], seed=seed):
            ta_y = ta_transform(ta_x)
        expected = einops.rearrange(kd_y, "1 time freq -> 1 freq time")
        self.assertTrue(torch.all(expected == ta_y), (expected - ta_y).abs().max())

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
                TimeMasking(time_mask_param=192),
                FrequencyMasking(freq_mask_param=48),
            ]),
        )

    def test_conversion_error(self):
        # edge case where torch.tensor(0.99999999) results in 1. due to conversion impresicions
        t = KDSpecAugment(frequency_masking=48, time_masking=192)
        rng = MockRng(
            dict(
                random=[0.99999999, 0.99999999, 0.99999999, 0.0, 0.99999999, 0.5],
            ),
        )
        t.set_rng(rng)
        x = torch.ones(1, 128, 512)
        for _ in range(3):
            t(x)
