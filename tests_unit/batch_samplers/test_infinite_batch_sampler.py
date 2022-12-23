import unittest

import torch
from torch.utils.data import DataLoader, RandomSampler
from kappadata.batch_samplers.infinite_batch_sampler import InfiniteBatchSampler
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.index_dataset import IndexDataset


class TestInfiniteBatchSampler(unittest.TestCase):
    def test(self):
        n_epochs = 20
        for dataset_size in [10]:
            for batch_size in [1, 2, 4]:
                for drop_last in [True, False]:
                    ds = ModeWrapper(dataset=IndexDataset(size=dataset_size), mode="x")
                    sampler = RandomSampler(ds)
                    batch_sampler = InfiniteBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
                    loader = DataLoader(ds, batch_sampler=batch_sampler)

                    if drop_last:
                        batches_per_epoch = dataset_size // batch_size
                    else:
                        batches_per_epoch = (dataset_size + batch_size - 1) // batch_size

                    samples = []
                    iterator = iter(loader)
                    for i in range(n_epochs):
                        for j in range(batches_per_epoch):
                            samples.append(next(iterator).clone())
                    samples = torch.concat(samples)
                    if not drop_last:
                        uniques, counts = samples.unique(return_counts=True)
                        self.assertTrue(torch.all(counts == counts[0]))
                        self.assertEqual(uniques.tolist(), list(range(dataset_size)))
                    else:
                        for i in range(n_epochs):
                            start = batches_per_epoch * batch_size * i
                            end = batches_per_epoch * batch_size * (i + 1)
                            self.assertTrue(len(samples[start:end].unique()) == len(samples[start:end]))

