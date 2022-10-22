from torch.utils.data import Subset
from functools import partial
from kappadata.errors import UseModeWrapperException

class KDSubset(Subset):
    def __getattr__(self, item):
        if item.startswith("getitem_"):
            # all methods starting with getitem_ are called with self.indices[idx]
            func = getattr(self.dataset, item)
            return partial(self._call_getitem, func)
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def _call_getitem(self, func, idx, *args, **kwargs):
        return func(self.indices[idx], *args, **kwargs)

    @property
    def root_dataset(self):
        return self.dataset.root_dataset

    def __getitem__(self, idx):
        raise UseModeWrapperException
