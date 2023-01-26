import unittest

from kappadata import object_to_transform, KDComposeTransform, KDScheduledTransform, KDRandomHorizontalFlip


class TestFactory(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(object_to_transform(None))

    def test_already_created(self):
        self.assertIsInstance(object_to_transform(KDRandomHorizontalFlip()), KDRandomHorizontalFlip)

    def test_implicit_compose(self):
        t = object_to_transform([
            dict(kind="kd_random_horizontal_flip")
        ])
        self.assertIsInstance(t, KDComposeTransform)
        self.assertIsInstance(t.transforms[0], KDRandomHorizontalFlip)

    def test_explicit_compose(self):
        t = object_to_transform(dict(
            kind="kd_compose_transform",
            transforms=[
                dict(kind="kd_random_horizontal_flip")
            ],
        ))
        self.assertIsInstance(t, KDComposeTransform)
        self.assertIsInstance(t.transforms[0], KDRandomHorizontalFlip)

    def test_scheduled_single(self):
        t = object_to_transform(dict(
            kind="kd_scheduled_transform",
            transform=dict(kind="kd_random_horizontal_flip"),
        ))
        self.assertIsInstance(t, KDScheduledTransform)
        self.assertIsInstance(t.transform, KDRandomHorizontalFlip)

    def test_scheduled_compose(self):
        t = object_to_transform(dict(
            kind="kd_scheduled_transform",
            transform=[
                dict(kind="kd_random_horizontal_flip")
            ],
        ))
        self.assertIsInstance(t, KDScheduledTransform)
        self.assertIsInstance(t.transform, KDComposeTransform)
        self.assertIsInstance(t.transform.transforms[0], KDRandomHorizontalFlip)
