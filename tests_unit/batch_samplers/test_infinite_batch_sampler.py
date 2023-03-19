import unittest
from functools import partial

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from kappadata.samplers.batch_samplers.infinite_batch_sampler import InfiniteBatchSampler
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.datasets.index_dataset import IndexDataset
from tests_util.infinite_batch_sampler_iterator import InfiniteBatchSamplerIterator


class TestInfiniteBatchSampler(unittest.TestCase):
    @staticmethod
    def sequential_sampler_ctor(dataset):
        return SequentialSampler(dataset)

    @staticmethod
    def random_sampler_ctor(dataset, seed):
        return RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))

    # very slow unittest and it is basically already tested below
    # def test_prefetch_next_epoch(self):
    #     ds = ModeWrapper(dataset=IndexDataset(size=8), mode="x")
    #     batch_sampler = InfiniteBatchSampler(
    #         sampler=self.sequential_sampler_ctor(dataset=ds),
    #         batch_size=8,
    #         drop_last=False,
    #     )
    #     infinite_loader = InfiniteBatchSamplerIterator(DataLoader(ds, batch_sampler=batch_sampler, num_workers=1))
    #     iterator = iter(infinite_loader)
    #     sleep(0.1)
    #     _, _, batch0 = next(iterator)
    #     _, _, batch1 = next(iterator)
    #     self.assertEqual(list(range(8)), batch0.tolist())
    #     self.assertEqual(batch0.tolist(), batch1.tolist())

    def test(self):
        n_epochs = 20
        for seed in [5]:
            for sampler_ctor in [
                partial(self.random_sampler_ctor, seed=seed),
                self.sequential_sampler_ctor,
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=0, shuffle=False, drop_last=False),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=0, shuffle=False, drop_last=True),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=0, shuffle=True, drop_last=False),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=0, shuffle=True, drop_last=True),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=1, shuffle=False, drop_last=False),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=1, shuffle=False, drop_last=True),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=1, shuffle=True, drop_last=False),
                partial(DistributedSampler, seed=seed, num_replicas=4, rank=1, shuffle=True, drop_last=True),
            ]:
                for dataset_size in [10, 100]:
                    for batch_size in [1, 2, 4, 8]:
                        for drop_last in [True, False]:
                            ds = ModeWrapper(dataset=IndexDataset(size=dataset_size), mode="x")
                            batch_sampler = InfiniteBatchSampler(
                                sampler=sampler_ctor(dataset=ds),
                                batch_size=batch_size,
                                drop_last=drop_last,
                            )
                            infinite_loader = InfiniteBatchSamplerIterator(DataLoader(ds, batch_sampler=batch_sampler))
                            finite_loader = DataLoader(
                                ds,
                                batch_size=batch_size,
                                drop_last=drop_last,
                                sampler=sampler_ctor(dataset=ds),
                            )

                            infinite_iter = iter(infinite_loader)
                            for expected_epoch_counter in range(n_epochs):
                                if isinstance(finite_loader.sampler, DistributedSampler):
                                    finite_loader.sampler.set_epoch(expected_epoch_counter)
                                for expected_batch_counter, expected_batch in enumerate(finite_loader):
                                    actual_epoch_counter, actual_batch_counter, actual_batch = next(infinite_iter)
                                    self.assertEqual(expected_epoch_counter, actual_epoch_counter)
                                    self.assertEqual(expected_batch_counter, actual_batch_counter)
                                    self.assertTrue(torch.all(expected_batch == actual_batch))
