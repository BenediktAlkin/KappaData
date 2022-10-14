from torch.utils.data import Dataset

class CachedDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def dispose(self):
        raise NotImplementedError

    def __getattr__(self, item):
        return getattr(self.dataset, item)