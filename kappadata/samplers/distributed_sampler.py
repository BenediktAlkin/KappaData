import torch
from torch.utils.data import DistributedSampler as TorchDistributedSampler


class DistributedSampler(TorchDistributedSampler):
    """ torch.utils.data.DistributedSampler but with support for RepeatedAugmentation """

    def __init__(self, *args, num_repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert 1 <= num_repeats
        self.num_repeats = num_repeats

    @property
    def effective_length(self):
        return len(self.dataset)

    def __iter__(self):
        if self.num_repeats == 1:
            yield from super().__iter__()
            return

        assert self.shuffle
        indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed + self.epoch))
        indices = indices.repeat_interleave(repeats=self.num_repeats)[:len(self.dataset)].tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        yield from indices
