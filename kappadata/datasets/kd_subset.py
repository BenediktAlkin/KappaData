from functools import partial

from torch.utils.data import Subset

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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    @property
    def root_dataset(self):
        return self.dataset.root_dataset

    def has_wrapper(self, wrapper):
        if self == wrapper:
            return True
        return self.dataset.has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if type(self) == wrapper_type:
            return True
        return self.dataset.has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        return [self] + self.dataset.all_wrappers

    @property
    def all_wrapper_types(self):
        return [type(self)] + self.dataset.all_wrapper_types

    def __getitem__(self, idx):
        raise UseModeWrapperException
