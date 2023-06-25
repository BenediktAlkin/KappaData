# TODO the problem is that since interleaved sampler is used, the assumption that the first batch is handled
#  by the dataloader process with rank0 is violated -> results in errornous behavior
#  I think the best way to solve this is to calculate the rank that loads the first batch and pass it via init_workers_fn
#  and have a global counter run in KDScheduledTransform
# TODO test with different number of workers
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

def main():
    train_dataset = ModeWrapper(
        dataset=XTransformWrapper(
            dataset=XDataset(x=torch.zeros(5)),
            transform=KDScheduledTransform(StrengthTransform(strength=1.)),
        ),
        mode="x",
    )
    test_dataset = ModeWrapper(
        dataset=XDataset(x=torch.full(size=(4,), fill_value=-1)),
        # dataset=XTransformWrapper(
        #     dataset=XDataset(x=torch.ones(4)),
        #     #transform=KDScheduledTransform(StrengthTransform(strength=2.)),
        # ),
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
        epochs=3,
    )
    loader = sampler.get_data_loader(num_workers=5)
    strengths = torch.concat([x.clone() for x in loader])
    expected_train = [0., 0., 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1., 1.]
    expected_test = [0., 0., 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0]
    print(strengths.tolist())
    print("fin")


if __name__ == '__main__':
    main()