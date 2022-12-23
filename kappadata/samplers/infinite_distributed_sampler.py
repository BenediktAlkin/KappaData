from torch.utils.data.distributed import DistributedSampler

class InfiniteDistributedSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, drop_last_batch, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch

    def __iter__(self):
        ds_len = len(self.dataset) // self.num_replicas
        if self.drop_last_batch:
            samples_per_batch = ds_len // self.batch_size * self.batch_size
        else:
            samples_per_batch = ds_len
        while True:
            idxs = list(super().__iter__())
            assert len(idxs) == len(self.dataset) // self.num_replicas
            yield from idxs[:samples_per_batch]