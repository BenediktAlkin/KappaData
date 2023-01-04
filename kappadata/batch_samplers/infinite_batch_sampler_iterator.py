from .infinite_batch_sampler import InfiniteBatchSampler


class InfiniteBatchSamplerIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        assert isinstance(self.dataloader.batch_sampler, InfiniteBatchSampler)
        # adapted from torch.utils.data.sampler.BatchSampler.__len__
        sampler = self.dataloader.batch_sampler
        if sampler.drop_last:
            batches_per_epoch = len(sampler.sampler) // sampler.batch_size
        else:
            batches_per_epoch = (len(sampler.sampler) + sampler.batch_size - 1) // sampler.batch_size
        for batch_idx, batch in enumerate(self.dataloader):
            epoch_counter = batch_idx // batches_per_epoch
            batch_counter = batch_idx % batches_per_epoch
            yield epoch_counter, batch_counter, batch
