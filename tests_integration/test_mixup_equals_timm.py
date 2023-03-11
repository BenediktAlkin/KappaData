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
from tests_util.patch_rng import patch_rng


class TestMixupEqualsTimm(unittest.TestCase):
    @patch_rng(fn_names=["numpy.random.rand", "numpy.random.beta", "numpy.random.randint"])
    def _run(self, batch_size, smoothing, mixup_alpha, cutmix_alpha, mixup_p, cutmix_p, mode):
        with patch("torch.Tensor.flip", lambda tensor, dim: tensor.roll(1, dim)):
            ds = create_image_classification_dataset(seed=552, size=100, channels=3, resolution=32, n_classes=10)
            mixup_ds = LabelSmoothingWrapper(dataset=ds, smoothing=smoothing)
            collator = KDMixCollator(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mixup_p=mixup_p,
                cutmix_p=cutmix_p,
                apply_mode=mode,
                lamb_mode=mode,
                shuffle_mode="flip",
                seed=0,
                dataset_mode="x class",
                return_ctx=False,
            )
            kd_loader = DataLoader(ModeWrapper(mixup_ds, mode="x class"), batch_size=batch_size, collate_fn=collator)

            timm_mixup = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=mixup_p + cutmix_p,
                switch_prob=cutmix_p,
                mode=mode,
                label_smoothing=smoothing,
                num_classes=ds.n_classes,
            )
            raw_loader = DataLoader(ModeWrapper(ds, mode="x class"), batch_size=batch_size)
            for i, ((raw_x, raw_y), (kd_x, kd_y)) in enumerate(zip(raw_loader, kd_loader)):
                timm_x, timm_y = timm_mixup(raw_x, raw_y)
                self.assertEqual(timm_x.shape, kd_x.shape)
                self.assertEqual(timm_y.shape, kd_y.shape)
                self.assertTrue(torch.allclose(timm_x, kd_x), f"x is unequal for i={i}")
                self.assertTrue(torch.allclose(timm_y, kd_y), f"y is unequal for i={i} timm={timm_y} kd={kd_y}")

    def test(self):
        self._run(
            batch_size=4,
            smoothing=0.1,
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            mixup_p=0.5,
            cutmix_p=0.5,
            mode="batch",
        )
