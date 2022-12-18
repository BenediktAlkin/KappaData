import torch

class InfiniteRandomSampler:
    def __init__(self, dataset_size, batch_size, drop_last=False, generator=None):
        super().__init__()
        self.dataset_size = dataset_size
        if drop_last:
            self.sampler_size = dataset_size // batch_size * batch_size
        else:
            self.sampler_size = dataset_size
        self.generator = generator

    def __iter__(self):
        while True:
            yield from torch.randperm(self.dataset_size, generator=self.generator)[:self.sampler_size].tolist()

    def __len__(self) -> int:
        return self.sampler_size