import unittest

import torch

from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.sample_wrappers.kd_mix_wrapper import KDMixWrapper
from tests_util.datasets.classification_dataset import ClassificationDataset


class TestKDMixWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        ds = ClassificationDataset(x=list(range(2)), classes=list(range(2)))
        self.assertRaises(AssertionError, lambda: KDMixWrapper(dataset=ds))
        self.assertRaises(AssertionError, lambda: KDMixWrapper(dataset=ds, mixup_p=0.5, cutmix_p=0.6))
        self.assertRaises(AssertionError, lambda: KDMixWrapper(dataset=ds, mixup_p=0.6, cutmix_p=0.5))
        self.assertRaises(AssertionError, lambda: KDMixWrapper(dataset=ds, mixup_p=0.5, cutmix_alpha=0.6))
        self.assertRaises(AssertionError, lambda: KDMixWrapper(dataset=ds, cutmix_p=0.5, mixup_alpha=0.6))
        _ = KDMixWrapper(dataset=ds, mixup_p=0.5, mixup_alpha=0.8)

    def test_non_onehot_class(self):
        ds = ClassificationDataset(
            x=torch.randn(5, 1, 8, 8, generator=torch.Generator().manual_seed(0)),
            classes=list(range(5)),
        )
        ds = KDMixWrapper(dataset=ds, mixup_p=1.0, mixup_alpha=0.8, seed=0)
        ds = ModeWrapper(dataset=ds, mode="x class")
        x, cls = ds[0]
        self.assertEqual((1, 8, 8), x.shape)
        self.assertEqual((5,), cls.shape)

    def test_pad_or_cut_end(self):
        generator = torch.Generator().manual_seed(0)
        ds = ClassificationDataset(
            x=[
                torch.randn(1, 8, 8, generator=generator),
                torch.randn(2, 5, 12, generator=generator),
                torch.randn(1, 10, 10, generator=generator),
            ],
            classes=list(range(3)),
        )
        ds = KDMixWrapper(dataset=ds, mixup_p=1.0, mixup_alpha=0.8, seed=0, mixup_unify_shapes_mode="pad_or_cut_end")
        ds = ModeWrapper(dataset=ds, mode="x class")
        # each sample should have the shape of its index -> second sample is adjusted to first sample
        for i, expected_shape in enumerate(map(lambda xx: xx.shape, ds.root_dataset.x)):
            for _ in range(5):
                x, _ = ds[i]
                self.assertEqual(expected_shape, x.shape)
