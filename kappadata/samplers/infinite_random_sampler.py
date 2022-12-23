from torch.utils.data.sampler import RandomSampler

class InfiniteRandomSampler(RandomSampler):
    def __init__(self, data_source, batch_size, drop_last, generator=None):
        super().__init__(data_source=data_source, generator=generator)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        if self.drop_last:
            samples_per_batch = len(self.data_source) // self.batch_size * self.batch_size
        else:
            samples_per_batch = len(self.data_source)
        while True:
            idxs = list(super().__iter__())
            assert len(idxs) == len(self.data_source)
            yield from idxs[:samples_per_batch]
