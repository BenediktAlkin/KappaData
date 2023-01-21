import unittest

import torch
from torchvision.transforms import Compose, Normalize

from kappadata.common.transforms.norm.kd_image_net_norm import KDImageNetNorm
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.norm.kd_image_range_norm import KDImageRangeNorm
from kappadata.utils.transform_utils import (
    flatten_transform,
    get_denorm_transform,
    get_norm_transform,
    get_x_transform,
)
from tests_util.datasets import XDataset
from kappadata import XTransformWrapper, KDRandomHorizontalFlip


class TestTransformUtils(unittest.TestCase):
    def _test_flatten_transforms(self, compose_ctor):
        transforms = [lambda x: x for _ in range(10)]
        transform = compose_ctor([
            transforms[0],
            compose_ctor(transforms[1:5]),
            transforms[5],
            transforms[6],
            compose_ctor(transforms[7:10]),
        ])
        flat = flatten_transform(transform)
        self.assertEqual(transforms, flat)

    def test_flatten_transforms_tv(self):
        self._test_flatten_transforms(Compose)

    def test_flatten_transforms_kd(self):
        self._test_flatten_transforms(KDComposeTransform)

    def test_get_norm_transform(self):
        norm = Normalize(mean=(0.3, 0.5), std=(0.1, 0.2))
        self.assertEqual(norm, get_norm_transform(norm))
        self.assertEqual(norm, get_norm_transform(Compose([norm])))
        self.assertEqual(norm, get_norm_transform(Compose([Compose([norm])])))
        self.assertEqual(norm, get_norm_transform(KDComposeTransform([norm])))

    def test_get_denorm_transform_kdnorm(self):
        norm0 = KDImageNetNorm()
        self.assertEqual(norm0.denormalize, get_denorm_transform(norm0).func)
        norm1 = KDImageRangeNorm()
        self.assertEqual(norm1.denormalize, get_denorm_transform(norm1).func)

        norm2 = KDImageNetNorm()
        transform = KDComposeTransform([norm2])
        self.assertEqual(norm2.denormalize, get_denorm_transform(transform).func)
        norm3 = KDImageRangeNorm()
        transform = KDComposeTransform([norm3])
        self.assertEqual(norm3.denormalize, get_denorm_transform(transform).func)

    def test_get_denorm_transform_tv(self):
        x = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(5))
        norm = Normalize(mean=(0.3, 0.5, 0.25), std=(0.05, 0.1, 0.2))
        denorm = get_denorm_transform(norm)
        self.assertTrue(torch.allclose(x, denorm(norm(x))))

    def test_get_norm_empty(self):
        self.assertIsNone(get_norm_transform(None))
        self.assertIsNone(get_norm_transform(lambda x: x))
        self.assertIsNone(get_norm_transform(Compose([lambda x: x])))
        self.assertIsNone(get_norm_transform(KDComposeTransform([lambda x: x])))

    def test_get_denorm_empty(self):
        self.assertIsNone(get_denorm_transform(None))
        self.assertIsNone(get_denorm_transform(lambda x: x))
        self.assertIsNone(get_denorm_transform(Compose([lambda x: x])))
        self.assertIsNone(get_denorm_transform(KDComposeTransform([lambda x: x])))

    def test_get_x_transform(self):
        self.assertIsNone(get_x_transform(None))
        hflip = KDRandomHorizontalFlip()
        ds = XDataset(x=None, transform=hflip)
        self.assertEqual(hflip, get_x_transform(ds))
