import unittest

from torchvision.transforms import ToTensor
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.norm.kd_image_net_norm import KDImageNetNorm

from kappadata.utils.is_deterministic_transform import (
    is_deterministic_transform,
    is_randomly_seeded_transform,
    has_stochastic_transform_with_seed,
)

class TestIsDeterministicTransform(unittest.TestCase):
    def test_deterministic_transform(self):
        self.assertTrue(is_deterministic_transform(KDImageNetNorm()).is_deterministic)
        self.assertTrue(is_deterministic_transform([KDImageNetNorm()]).is_deterministic)

    def test_stochastic_transform(self):
        self.assertFalse(is_deterministic_transform(KDRandomGrayscale(p=0.2)).is_deterministic)
        self.assertFalse(is_deterministic_transform([KDRandomGrayscale(p=0.2)]).is_deterministic)
        self.assertTrue(is_deterministic_transform(KDRandomGrayscale(p=0.2, seed=5)).is_deterministic)
        self.assertTrue(is_deterministic_transform([KDRandomGrayscale(p=0.2, seed=5)]).is_deterministic)

    def test_is_randomly_seeded(self):
        self.assertTrue(is_randomly_seeded_transform(KDRandomGrayscale(p=0.2, seed=5)))
        self.assertTrue(is_randomly_seeded_transform([
            KDRandomGrayscale(p=0.2, seed=5),
            KDRandomGrayscale(p=0.2, seed=6),
        ]))
        self.assertFalse(is_randomly_seeded_transform([
            KDRandomGrayscale(p=0.2, seed=5),
            KDRandomGrayscale(p=0.2, seed=5),
        ]))

    def test_mix1(self):
        determinstic = KDRandomGrayscale(p=0.2, seed=5)
        non_deterministic0 = KDRandomGrayscale(p=0.2)
        non_deterministic1 = KDRandomGrayscale(p=0.2)
        unknown = ToTensor()
        result = is_deterministic_transform([determinstic, non_deterministic0, non_deterministic1, unknown])
        self.assertFalse(result.is_deterministic)
        self.assertFalse(result.all_kd_transforms_are_deterministic)
        self.assertTrue(result.is_randomly_seeded)

    def test_mix2(self):
        determinstic0 = KDRandomGrayscale(p=0.2, seed=5)
        deterministic1 = KDRandomGrayscale(p=0.2, seed=6)
        deterministic2 = KDRandomGrayscale(p=0.2, seed=8)
        unknown = ToTensor()
        result = is_deterministic_transform([determinstic0, deterministic1, deterministic2, unknown])
        self.assertFalse(result.is_deterministic)
        self.assertTrue(result.all_kd_transforms_are_deterministic)
        self.assertTrue(result.is_randomly_seeded)

    def test_has_stochastic_transform_with_seed(self):
        self.assertTrue(has_stochastic_transform_with_seed(KDRandomGrayscale(p=0.2, seed=5)))
        self.assertFalse(has_stochastic_transform_with_seed(KDRandomGrayscale(p=0.2)))
        self.assertTrue(has_stochastic_transform_with_seed([KDRandomGrayscale(p=0.2, seed=5)]))
        self.assertFalse(has_stochastic_transform_with_seed([KDRandomGrayscale(p=0.2)]))
        self.assertTrue(has_stochastic_transform_with_seed([
            KDRandomGrayscale(p=0.2, seed=5),
            KDRandomGrayscale(p=0.2),
            KDRandomGrayscale(p=0.2),
        ]))
        self.assertFalse(has_stochastic_transform_with_seed([
            KDRandomGrayscale(p=0.2),
            KDRandomGrayscale(p=0.2),
            KDRandomGrayscale(p=0.2),
        ]))