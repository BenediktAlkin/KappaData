# adapted from https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py
from multiprocessing import Manager
from .cached_dataset import CachedDataset


class SharedDictDataset(CachedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manager = Manager()
        self.shared_dict = manager.dict()

    def _getitem_impl(self, idx):
        if idx not in self.shared_dict:
            sample = self.dataset[idx]
            self.shared_dict[idx] = sample
        else:
            sample = self.shared_dict[idx]
        return sample

    def dispose(self):
        self.shared_dict.clear()