from torch.utils.data.sampler import BatchSampler


class InfiniteBatchSampler(BatchSampler):
    """
    BatchSampler that keeps fetching batches across epoch boundaries.
    This allows a DataLoader to prefetch batches from the next epoch.
    This is especially useful for small datasets as the default
    pytorch BatchSampler will always have to wait for the first batch
    of EVERY epoch to be ready.
    In the extreme case (batch_size == len(dataset)) the dataloading
    will be fully synchronous (with BatchSampler), independent of how
    many workers are used.
    With InfiniteBatchSampler this downtime is avoided.
    TODO code example
    """

    def __iter__(self):
        while True:
            yield from super().__iter__()

    def __len__(self):
        raise NotImplementedError
