import logging

from torch.utils.data import Dataset

from kappadata.errors import UseModeWrapperException


class KDDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(type(self).__name__)

    def __len__(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    def dispose(self):
        """ release resources occupied by dataset (e.g. filehandles) """
        pass

    @property
    def root_dataset(self):
        return self

    def __getitem__(self, idx):
        raise UseModeWrapperException
