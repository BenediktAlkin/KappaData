import torch
from kappadata.utils.getall_class_as_tensor import getall_class_as_tensor


class SemiSampler:
    """
    generates indices such that
    for _ in range(n_labeled):
      yield sample_labeled_index()
    for _ in range(n_unlabeled):
      yield sample_unlabeled_index()

    if batch_size % (n_labeled + n_unlabeled) == 0:
        each batch has n_labeled / (n_labeled + n_unlabeled) labeled samples
        each batch has n_unlabeled / (n_labeled + n_unlabeled) unlabeled samples
    """

    def __init__(self, dataset, n_labeled=1, n_unlabeled=1, generator=None):
        super().__init__()
        assert 1 <= n_labeled
        assert 1 <= n_unlabeled
        self.dataset = dataset
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.generator = generator

        self.classes = getall_class_as_tensor(dataset)
        is_unlabeled = self.classes == -1
        self.labeled_idxs = (~is_unlabeled).nonzero().squeeze(1).tolist()
        self.unlabeled_idxs = is_unlabeled.nonzero().squeeze(1).tolist()
        assert len(self.labeled_idxs) > 0 and len(self.unlabeled_idxs) > 0

    def __iter__(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        def _iterator(idxs):
            while True:
                yield from torch.randperm(len(idxs), generator=generator).tolist()
        labeled_iterator = _iterator(self.labeled_idxs)
        unlabeled_iterator = _iterator(self.unlabeled_idxs)
        for i in range(len(self.labeled_idxs) + len(self.unlabeled_idxs)):
            if i % (self.n_labeled + self.n_unlabeled) < self.n_labeled:
                yield self.labeled_idxs[next(labeled_iterator)]
            else:
                yield self.unlabeled_idxs[next(unlabeled_iterator)]

