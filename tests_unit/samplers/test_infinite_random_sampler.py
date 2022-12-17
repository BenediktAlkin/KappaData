import torch
import unittest
from tests_util.index_dataset import IndexDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.samplers.infinite_random_sampler import InfiniteRandomSampler


class TestInfiniteRandomSampler(unittest.TestCase):
    @staticmethod
    def init(size, n_epochs, batch_size):
        ds = ModeWrapper(dataset=IndexDataset(size=size), mode="x")
        batches_per_epoch = int(ds.size / batch_size)
        n_batches = n_epochs * batches_per_epoch
        n_samples = n_batches * batch_size
        samples_per_batch = batch_size * batches_per_epoch
        return ds, n_epochs, n_samples, n_batches, samples_per_batch

    def test_single(self):
        infinite_rng = torch.Generator().manual_seed(5)
        finite_rng = torch.Generator().manual_seed(5)
        batch_size = 2
        ds, n_epochs, n_samples, n_batches, samples_per_batch = self.init(size=2, n_epochs=5, batch_size=batch_size)
        sampler = InfiniteRandomSampler(data_source=ds, generator=infinite_rng)
        self.assertEquals(ds.size, len(sampler))

        infinite_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, generator=infinite_rng)
        finite_loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, generator=finite_rng)

        infinite_samples = []
        infinite_iter = iter(infinite_loader)
        for i in range(n_batches):
            infinite_samples.append(next(infinite_iter).clone())
        infinite_samples = torch.concat(infinite_samples)
        finite_samples = []
        for i in range(n_epochs):
            for x in finite_loader:
                finite_samples.append(x.clone())
        finite_samples = torch.concat(finite_samples)
        self.assertEquals(finite_samples.tolist(), infinite_samples.tolist())

    def test_distributed(self):
        raise NotImplementedError
