import unittest
from functools import partial

import torch
from torch.utils.data import DataLoader

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.base.kd_scheduled_transform import KDScheduledTransform
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper
from tests_util.datasets.x_dataset import XDataset
from tests_util.transforms.strength_transform import StrengthTransform
from kappadata.samplers import InterleavedSampler, InterleavedSamplerConfig
from torch.utils.data import RandomSampler, SequentialSampler


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

        strengths = torch.concat([x.clone() for x in loader])
        self.assertEqual([0, 0, 2.5, 2.5, 5, 5, 7.5, 7.5, 10, 10], strengths.tolist())

    def test_single(self):
        self._test(StrengthTransform(strength=10.))

    def test_compose(self):
        self._test(KDComposeTransform([StrengthTransform(strength=1.), StrengthTransform(strength=10.)]))

    def test_interleaved_sampler(self):
        train_dataset = ModeWrapper(
            dataset=XTransformWrapper(
                dataset=XDataset(x=torch.zeros(11)),
                transform=KDScheduledTransform(StrengthTransform(strength=1.)),
            ),
            mode="x",
        )
        test_dataset = ModeWrapper(
            dataset=XTransformWrapper(
                dataset=XDataset(x=torch.ones(8)),
                transform=KDScheduledTransform(StrengthTransform(strength=2.)),
            ),
            mode="x",
        )
        sampler = InterleavedSampler(
            main_sampler=SequentialSampler(train_dataset),
            configs=[
                InterleavedSamplerConfig(
                    sampler=SequentialSampler(test_dataset),
                    every_n_epochs=1,
                ),
            ],
            batch_size=2,
            drop_last=True,
            epochs=1,
        )
        loader = sampler.get_data_loader(num_workers=2)
        strengths = torch.concat([x.clone() for x in loader])
        expected_train = [0., 0., 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1., 1.]
        expected_test = [0., 0., 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0]
        self.assertEqual(10 + len(test_dataset), len(strengths))
        self.assertEqual(expected_train + expected_test, strengths.tolist())
