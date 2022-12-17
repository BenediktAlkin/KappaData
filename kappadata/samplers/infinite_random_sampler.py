from torch.utils.data.sampler import RandomSampler

class InfiniteRandomSampler(RandomSampler):
    def __iter__(self):
        while True:
            yield from super().__iter__()