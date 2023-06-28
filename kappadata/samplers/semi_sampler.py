import torch
from kappadata.utils.getall_class_as_tensor import getall_class_as_tensor
from kappadata.utils.distributed import get_rank, get_world_size

class SemiSampler:
    """
    generates indices such that
    for _ in range(num_labeled):
      yield sample_labeled_index()
    for _ in range(num_unlabeled):
      yield sample_unlabeled_index()

    if batch_size % (num_labeled + num_unlabeled) == 0:
        each batch has num_labeled / (num_labeled + num_unlabeled) labeled samples
        each batch has num_unlabeled / (num_labeled + num_unlabeled) unlabeled samples

    distributed sampling is implemented via:
    - shuffling with a different seed per device
    - dividing total length by world size (same as DistributedSampler with drop_last=True)
    """

    def __init__(self, dataset, num_labeled=1, num_unlabeled=1, rank=None, world_size=None, seed=0):
        super().__init__()
        assert 1 <= num_labeled
        assert 1 <= num_unlabeled
        self.dataset = dataset
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0
        self.seed = seed

        self.classes = getall_class_as_tensor(dataset)
        is_unlabeled = self.classes == -1
        self.labeled_idxs = (~is_unlabeled).nonzero().squeeze(1).tolist()
        self.unlabeled_idxs = is_unlabeled.nonzero().squeeze(1).tolist()
        assert len(self.labeled_idxs) > 0 and len(self.unlabeled_idxs) > 0

    @property
    def effective_length(self):
        # one epoch is when all labeled samples are returned once
        return len(self.labeled_idxs)

    def __len__(self):
        # one epoch is when all labeled samples are returned once -> pad with number of unlabeled samples
        num_chunks = len(self.labeled_idxs) // self.num_labeled
        length = num_chunks * (self.num_labeled + self.num_unlabeled)
        # adjust to length-per-device and cutoff trailing samples for distributed
        length //= self.world_size
        return length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # generate random numbers to avoid self.seed + self.epoch + self.rank
        # (epoch 0 of rank 1 would have same seed as epoch 1 of rank 0)
        rank_seed = torch.empty((), dtype=torch.int32).random_(generator=torch.Generator().manual_seed(self.rank))
        epoch_seed = torch.empty((), dtype=torch.int32).random_(generator=torch.Generator().manual_seed(self.epoch))
        generator = torch.Generator().manual_seed(self.seed + rank_seed.item() + epoch_seed.item())

        def _iterator(idxs):
            while True:
                yield from torch.randperm(len(idxs), generator=generator).tolist()
        labeled_iterator = _iterator(self.labeled_idxs)
        unlabeled_iterator = _iterator(self.unlabeled_idxs)
        for i in range(len(self)):
            if i % (self.num_labeled + self.num_unlabeled) < self.num_labeled:
                yield self.labeled_idxs[next(labeled_iterator)]
            else:
                yield self.unlabeled_idxs[next(unlabeled_iterator)]
