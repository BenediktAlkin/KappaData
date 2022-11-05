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

    @staticmethod
    def has_wrapper(wrapper):
        return False

    @staticmethod
    def has_wrapper_type(wrapper_type):
        return False

    @property
    def all_wrappers(self):
        return []

    @property
    def all_wrapper_types(self):
        return []

    def __getitem__(self, idx):
        raise UseModeWrapperException
