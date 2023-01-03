import unittest

import torch
from torch.utils.data import DataLoader, RandomSampler
from kappadata.batch_samplers.infinite_batch_sampler import InfiniteBatchSampler
from kappadata.batch_samplers.infinite_batch_sampler_iterator import InfiniteBatchSamplerIterator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.index_dataset import IndexDataset


class TestInfiniteBatchSampler(unittest.TestCase):
    def test(self):
        n_epochs = 20
        for dataset_size in [10]:
            for batch_size in [1, 2, 4]:
                for drop_last in [True, False]:
                    ds = ModeWrapper(dataset=IndexDataset(size=dataset_size), mode="x")
                    infinite_sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(5))
                    batch_sampler = InfiniteBatchSampler(
                        sampler=infinite_sampler,
                        batch_size=batch_size,
                        drop_last=drop_last,
                    )
                    infinite_loader = InfiniteBatchSamplerIterator(DataLoader(ds, batch_sampler=batch_sampler))
                    finite_sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(5))
                    finite_loader = DataLoader(ds, batch_size=batch_size, drop_last=drop_last, sampler=finite_sampler)

                    infinite_iter = iter(infinite_loader)
                    for expected_epoch_counter in range(n_epochs):
                        for expected_batch_counter, expected_batch in enumerate(finite_loader):
                            actual_epoch_counter, actual_batch_counter, actual_batch = next(infinite_iter)
                            self.assertEqual(expected_epoch_counter, actual_epoch_counter)
                            self.assertEqual(expected_batch_counter, actual_batch_counter)
                            self.assertTrue(torch.all(expected_batch == actual_batch))