import torch
from torch.utils.data import RandomSampler as TorchRandomSampler


class RandomSampler(TorchRandomSampler):
    """ torch.utils.data.RandomSampler but with support for RepeatedAugmentation """

    def __init__(self, *args, num_repeats=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert 1 <= num_repeats
        self.num_repeats = num_repeats

    @property
    def effective_length(self):
        return self.num_samples

    def __iter__(self):
        if self.num_repeats == 1:
            yield from super().__iter__()
            return

        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            idxs = torch.randint(high=n, size=(n,), dtype=torch.int64, generator=generator)
        else:
            idxs = torch.randperm(n, generator=generator)
        yield from idxs.repeat_interleave(repeats=self.num_repeats)[:n].tolist()
