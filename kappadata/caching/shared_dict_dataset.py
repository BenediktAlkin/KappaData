# adapted from https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py
from multiprocessing import Manager
from .cached_dataset import CachedDataset


class SharedDictDataset(CachedDataset):
    def __init__(self, dataset, transform=None):
        manager = Manager()
        self.shared_dict = manager.dict()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        if idx not in self.shared_dict:
            sample = self.dataset[idx]
            self.shared_dict[idx] = sample
        else:
            sample = self.shared_dict[idx]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        return getattr(self.dataset, item)

    def dispose(self):
        self.shared_dict.clear()