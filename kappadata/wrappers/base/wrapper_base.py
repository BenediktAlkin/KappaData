from torch.utils.data import Subset
from functools import partial

class WrapperBase(Subset):
    def __getattr__(self, item):
        if item.startswith("getitem_"):
            # all methods starting with getitem_ are called with self.indices[idx]
            func = getattr(self.dataset, item)
            return partial(self._call_with_nested_index, func)
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    @property
    def root_dataset(self):
        return self.dataset.root_dataset

    def _call_with_nested_index(self, func, idx):
        return func(self.indices[idx])
