import logging
from torch.utils.data import Dataset

class CachedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.logger = logging.getLogger(type(self).__name__)
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = self._getitem_impl(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def _getitem_impl(self, index):
        raise NotImplementedError

    def dispose(self):
        raise NotImplementedError