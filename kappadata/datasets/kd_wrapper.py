from kappadata.errors import UseModeWrapperException
from .kd_dataset import KDDataset

# TODO test and publish
# TODO better naming conventions between wrapper that modifies which samples are in the dataset and wrappers which
#  modifiy the sample directly
# TODO maybe better naming is not really needed as wrappers can be nested arbitrarily
# TODO should probably rename wrappers into dataset_wrappers and call the other ones sample_wrappers and make another
#  package for them
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
