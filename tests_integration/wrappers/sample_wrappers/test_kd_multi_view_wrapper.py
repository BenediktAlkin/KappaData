import unittest

import torch

from kappadata.transforms import KDAdditiveGaussianNoise, KDIdentityTransform, KDComposeTransform
from kappadata.wrappers.sample_wrappers.kd_multi_view_wrapper import KDMultiViewWrapper
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper
from tests_util.datasets.x_dataset import XDataset


class TestKDMultiViewWrapper(unittest.TestCase):
    def test_check_deterministic_x_transform_single_stochastic(self):
        with self.assertRaises(AssertionError):
            KDMultiViewWrapper(
                dataset=XTransformWrapper(
                    dataset=XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5))),
                    transform=KDAdditiveGaussianNoise(),
                ),
                configs=[2],
            )

    def test_check_deterministic_x_transform_compose_stochastic(self):
        with self.assertRaises(AssertionError):
            KDMultiViewWrapper(
                dataset=XTransformWrapper(
                    dataset=XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5))),
                    transform=KDComposeTransform([KDAdditiveGaussianNoise()]),
                ),
                configs=[2],
            )

    def test_check_deterministic_x_transform_compose_mix(self):
        with self.assertRaises(AssertionError):
            KDMultiViewWrapper(
                dataset=XTransformWrapper(
                    dataset=XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5))),
                    transform=KDComposeTransform([KDIdentityTransform(), KDAdditiveGaussianNoise()]),
                ),
                configs=[2],
            )

    def test_check_deterministic_x_transform_deterministic(self):
        KDMultiViewWrapper(
            dataset=XTransformWrapper(
                dataset=XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5))),
                transform=KDIdentityTransform(),
            ),
            configs=[2],
        )
        KDMultiViewWrapper(
            dataset=XTransformWrapper(
                dataset=XDataset(x=torch.randn(10, generator=torch.Generator().manual_seed(5))),
                transform=KDComposeTransform([KDIdentityTransform()]),
            ),
            configs=[2],
        )
