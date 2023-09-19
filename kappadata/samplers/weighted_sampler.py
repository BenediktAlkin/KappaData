import torch

from kappadata.utils.distributed import get_rank, get_world_size
from kappadata.utils.getall_as_tensor import getall_as_tensor


class WeightedSampler:
    def __init__(self, dataset, weights, size=None, seed=0, rank=None, world_size=None):
        super().__init__()
        assert len(dataset) == len(weights)
        self.dataset = dataset
        self.weights = weights
        self.size = size
        self.seed = seed
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0

    @property
    def effective_length(self):
        if self.size is None:
            return len(self.dataset)
        assert len(self.dataset) >= self.size, f"{len(self.dataset)} < {self.size}"
        return self.size

    def __len__(self):
        # adjust to length-per-device and cutoff trailing samples for distributed
        return self.effective_length // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # draw indices for current epoch (same for all ranks)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.effective_length, replacement=False, generator=generator)
        # distribute among ranks
        indices = indices[self.rank:self.effective_length:self.world_size].tolist()
        # drop last
        indices = indices[:len(self)]
        yield from indices
