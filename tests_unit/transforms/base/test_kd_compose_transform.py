import unittest

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale


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
