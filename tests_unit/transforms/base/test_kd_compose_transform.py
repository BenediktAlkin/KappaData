import torch
import unittest

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.norm.kd_image_net_norm import KDImageNetNorm


class TestKDComposeTransform(unittest.TestCase):
    def test_disallow_same_seed(self):
        with self.assertRaises(AssertionError) as ex:
            KDComposeTransform([
                KDRandomGrayscale(p=0.2, seed=5),
                KDRandomGrayscale(p=0.2, seed=5),
            ])
        msg = "transforms of type KDStochasticTransform should use different seeds (found seeds [5, 5])"
        self.assertEqual(msg, str(ex.exception))

    def test_allow_same_seed(self):
        KDComposeTransform([
            KDRandomGrayscale(p=0.2, seed=5),
            KDRandomGrayscale(p=0.2, seed=5),
        ], allow_same_seed=True)

    def test_scale_probs(self):
        grayscale = KDRandomGrayscale(p=0.2)
        blur = KDRandomGaussianBlurPIL(p=1.0, sigma=(0.1, 2.0))
        transform = KDComposeTransform([
            grayscale,
            blur,
            KDImageNetNorm(),
        ])
        for scale in torch.linspace(0, 1, 11).tolist():
            transform.scale_probs(scale)
            self.assertEqual(0.2 * scale, grayscale.p)
            self.assertEqual(scale, blur.p)