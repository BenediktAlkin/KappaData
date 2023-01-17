import unittest

import torch

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.norm.kd_image_net_norm import KDImageNetNorm


class TestKDComposeTransform(unittest.TestCase):
    def test_check_consistent_seeds_error(self):
        with self.assertRaises(AssertionError) as ex:
            KDComposeTransform([
                KDRandomGrayscale(p=0.2, seed=5),
                KDRandomGrayscale(p=0.2, seed=5),
            ])
        msg = "transforms of type KDStochasticTransform within a KDComposeTransform should have: " \
              "1. seed is set for all KDStochasticTransforms or for none + " \
              "2. the seeds should be different to avoid patterns"
        self.assertEqual(msg, str(ex.exception))

    def test_check_consistent_seeds(self):
        KDComposeTransform([
            KDRandomGrayscale(p=0.2, seed=5),
            KDRandomGrayscale(p=0.2, seed=5),
        ], check_consistent_seeds=False)

    def test_scale_strength(self):
        grayscale = KDRandomGrayscale(p=0.2)
        blur = KDRandomGaussianBlurPIL(p=1.0, sigma=(0.1, 2.0))
        transform = KDComposeTransform([
            grayscale,
            blur,
            KDImageNetNorm(),
        ])
        for factor in torch.linspace(0, 1, 11).tolist():
            transform.scale_strength(factor)
            self.assertEqual(0.2 * factor, grayscale.p)
            self.assertEqual(0.1, blur.gaussian_blur.sigma_lb)
            self.assertEqual(0.1 + (2.0 - 0.1) * factor, blur.gaussian_blur.sigma_ub)
