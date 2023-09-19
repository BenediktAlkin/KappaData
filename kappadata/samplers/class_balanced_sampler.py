import torch

from kappadata.utils.distributed import get_rank, get_world_size
from kappadata.utils.getall_as_tensor import getall_as_tensor


class ClassBalancedSampler:
    def __init__(
            self,
            dataset,
            shuffle=True,
            samples_per_class=None,
            getall_item="class",
            seed=0,
            rank=None,
            world_size=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0

        # load/check all classes
        self.num_classes = max(2, dataset.getdim_class())
        classes = getall_as_tensor(self.dataset, item=getall_item)
        unique, counts = classes.unique(return_counts=True)
        assert classes.ndim == 1
        assert len(unique) == self.num_classes

        # calculate indices per class
        self.indices_per_class = [(classes == i).nonzero().squeeze(1) for i in range(self.num_classes)]
        self.samples_per_class = samples_per_class or counts.max().item()

    @property
    def effective_length(self):
        return self.num_classes * self.samples_per_class

    def __len__(self):
        # adjust to length-per-device and cutoff trailing samples for distributed
        return self.effective_length // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # draw indices for current epoch (same for all ranks)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = []
        for indices_per_class in self.indices_per_class:
            remaining_indices = self.samples_per_class
            while remaining_indices > 0:
                if self.shuffle:
                    perm = torch.randperm(len(indices_per_class), generator=generator)
                else:
                    perm = torch.arange(len(indices_per_class))
                perm = perm[:remaining_indices]
                indices.append(indices_per_class[perm])
                remaining_indices -= len(perm)
        indices = torch.concat(indices)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices), generator=generator)]
        # distribute among ranks
        indices = indices[self.rank:self.effective_length:self.world_size].tolist()
        # drop last
        indices = indices[:len(self)]
        yield from indices
