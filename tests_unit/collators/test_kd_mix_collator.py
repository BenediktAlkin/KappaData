import unittest

import torch
import numpy as np
from torch.utils.data import DataLoader

from kappadata.collators.kd_mix_collator import KDMixCollator
from kappadata.error_messages import REQUIRES_MIXUP_P_OR_CUTMIX_P
from kappadata.utils.one_hot import to_one_hot_matrix
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.sample_wrappers.one_hot_wrapper import OneHotWrapper
from tests_util.datasets import create_image_classification_dataset


class TestKDMixCollator(unittest.TestCase):
    def test_ctor_probs(self):
        self.assertEqual(0., KDMixCollator(mixup_p=1., mixup_alpha=0.8).cutmix_p)
        self.assertEqual(0., KDMixCollator(cutmix_p=1., cutmix_alpha=1.0).mixup_p)
        with self.assertRaises(AssertionError) as ex:
            self.assertEqual(0., KDMixCollator())
        self.assertEqual(REQUIRES_MIXUP_P_OR_CUTMIX_P, str(ex.exception))

    def test_binary(self):
        ds = create_image_classification_dataset(size=16, seed=19521, channels=1, resolution=8, n_classes=2)
        ds_mode = "x class"
        ds = ModeWrapper(dataset=ds, mode=ds_mode, return_ctx=False)
        # mixup
        mix_collator = KDMixCollator(
            mixup_alpha=1.,
            mixup_p=1.,
            apply_mode="sample",
            lamb_mode="sample",
            shuffle_mode="roll",
            dataset_mode=ds_mode,
            return_ctx=False,
        ).set_rng(np.random.default_rng(seed=3))
        dl = DataLoader(ds, batch_size=len(ds), collate_fn=mix_collator)
        (_, y) = next(iter(dl))
        self.assertEqual(1, y.ndim)

    def test_mixup(self):
        ds = create_image_classification_dataset(size=16, seed=19521, channels=1, resolution=8, n_classes=4)
        ds = OneHotWrapper(dataset=ds)
        ds_mode = "x class"
        ds = ModeWrapper(dataset=ds, mode=ds_mode, return_ctx=True)

        # mixup
        mix_collator = KDMixCollator(
            mixup_alpha=1.,
            mixup_p=1.,
            apply_mode="sample",
            lamb_mode="sample",
            shuffle_mode="roll",
            dataset_mode=ds_mode,
            return_ctx=True,
        ).set_rng(np.random.default_rng(seed=3))
        dl = DataLoader(ds, batch_size=len(ds), collate_fn=mix_collator)
        (x, y), ctx = next(iter(dl))
        lamb = ctx["lambda"]
        lamb_x = lamb.view(-1, 1, 1, 1)
        lamb_y = lamb.view(-1, 1)

        # check x
        expected_x = ds.x * lamb_x + ds.x.roll(shifts=1, dims=0) * (1. - lamb_x)
        self.assertTrue(torch.all(x == expected_x))

        # check y
        og_y = to_one_hot_matrix(ds.classes, n_classes=ds.getdim_class())
        expected_y = og_y * lamb_y + og_y.roll(shifts=1, dims=0) * (1. - lamb_y)
        self.assertTrue(torch.all(y == expected_y))

        # check y has probability mass at most 2 entries and it sums to 1
        n_probabilty_masses = (y > 0).sum(dim=1)
        self.assertTrue(torch.all(torch.logical_or(n_probabilty_masses == 1, n_probabilty_masses == 2)))
        self.assertTrue(torch.all(y.sum(dim=1) == 1.))
