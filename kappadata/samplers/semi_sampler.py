import torch
from kappadata.utils.getall_class_as_tensor import getall_class_as_tensor


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
    """

    def __init__(self, dataset, num_labeled=1, num_unlabeled=1, shuffle=True, generator=None):
        super().__init__()
        assert 1 <= num_labeled
        assert 1 <= num_unlabeled
        self.dataset = dataset
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.shuffle = shuffle
        self.generator = generator

        self.classes = getall_class_as_tensor(dataset)
        is_unlabeled = self.classes == -1
        self.labeled_idxs = (~is_unlabeled).nonzero().squeeze(1).tolist()
        self.unlabeled_idxs = is_unlabeled.nonzero().squeeze(1).tolist()
        assert len(self.labeled_idxs) > 0 and len(self.unlabeled_idxs) > 0

    def __iter__(self):
        if self.generator is None and self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        def _iterator(idxs):
            while True:
                if self.shuffle:
                    yield from torch.randperm(len(idxs), generator=generator).tolist()
                else:
                    yield from range(len(idxs))
        labeled_iterator = _iterator(self.labeled_idxs)
        unlabeled_iterator = _iterator(self.unlabeled_idxs)
        for i in range(len(self.labeled_idxs) + len(self.unlabeled_idxs)):
            if i % (self.num_labeled + self.num_unlabeled) < self.num_labeled:
                yield self.labeled_idxs[next(labeled_iterator)]
            else:
                yield self.unlabeled_idxs[next(unlabeled_iterator)]

