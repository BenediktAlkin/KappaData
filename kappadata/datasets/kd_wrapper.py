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