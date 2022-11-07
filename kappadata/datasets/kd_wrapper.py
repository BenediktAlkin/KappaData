from kappadata.errors import UseModeWrapperException
from .kd_dataset import KDDataset


class KDWrapper(KDDataset):
    def __init__(self, dataset: KDDataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def dispose(self):
        self.dataset.dispose()

    @property
    def root_dataset(self):
        # KDDataset implements root_dataset -> __getitem__ doesn't trigger
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
