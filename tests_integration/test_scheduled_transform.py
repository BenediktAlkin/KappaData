import unittest
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.transforms.base.kd_scheduled_transform import KDScheduledTransform, KDTransform
from tests_util.transforms.strength_transform import StrengthTransform
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper
from functools import partial
from tests_util.datasets.x_dataset import XDataset
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform

class TestScheduledTransform(unittest.TestCase):
    def _test(self, transform):
        batch_size = 2
        updates = 5
        transform = KDScheduledTransform(transform=transform)
        dataset = XDataset(x=torch.zeros(batch_size * updates))
        dataset = XTransformWrapper(dataset=dataset, transform=transform)
        dataset = ModeWrapper(dataset=dataset, mode="x")

        worker_init_fn = partial(dataset.worker_init_fn, batch_size=batch_size, updates=updates)
        loader = DataLoader(dataset, batch_size=batch_size, worker_init_fn=worker_init_fn, num_workers=2)

        strengths = []
        for x in loader:
            strengths.append(x.clone())
        self.assertEqual([0, 0, 2.5, 2.5, 5, 5, 7.5, 7.5, 10, 10], torch.concat(strengths).tolist())

    def test_single(self):
        self._test(StrengthTransform(strength=10.))

    def test_compose(self):
        self._test(KDComposeTransform([StrengthTransform(strength=1.), StrengthTransform(strength=10.)]))