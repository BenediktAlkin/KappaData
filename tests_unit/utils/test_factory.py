import unittest

from torchvision.transforms import ToTensor

from kappadata.factory import register_transform, object_to_transform
from kappadata.transforms import KDComposeTransform


class TestFactory(unittest.TestCase):
    class MultiplyTransform:
        def __init__(self, factor):
            self.factor = factor

        def forward(self, x):
            return x * self.factor

    def test_object_to_transform_registered_transform_compose(self):
        register_transform("multiply_transform", self.MultiplyTransform)
        transform = object_to_transform(
            [
                dict(kind="multiply_transform", factor=2),
                dict(kind="to_tensor"),
            ]
        )
        self.assertIsInstance(transform, KDComposeTransform)
        self.assertEqual(2, len(transform.transforms))
        self.assertIsInstance(transform.transforms[0], self.MultiplyTransform)
        self.assertEqual(2, transform.transforms[0].factor)
        self.assertIsInstance(transform.transforms[1], ToTensor)

    def test_object_to_transform_registered_transform_compose_single(self):
        register_transform("multiply_transform", self.MultiplyTransform)
        transform = object_to_transform(dict(kind="multiply_transform", factor=2))
        self.assertIsInstance(transform, self.MultiplyTransform)
        self.assertEqual(2, transform.factor)
