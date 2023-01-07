import unittest

from kappadata.transforms.base.kd_scheduled_compose_transform import KDScheduledComposeTransform
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.norm.kd_image_net_norm import KDImageNetNorm
import torch

class TestKDScheduledComposeTransform(unittest.TestCase):
    def test_scale_probs(self):
        grayscale = KDRandomGrayscale(p=0.2)
        blur = KDRandomGaussianBlurPIL(p=1.0, sigma=(0.1, 2.0))
        transform = KDScheduledComposeTransform([
            grayscale,
            blur,
            KDImageNetNorm(),
        ])
        for scale in torch.linspace(0, 1, 11).tolist():
            transform.scale_probs(scale)
            self.assertEqual(0.2 * scale, grayscale.p)
            self.assertEqual(scale, blur.p)