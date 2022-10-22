from torch.utils.data import Dataset
import logging
from kappadata.errors import UseModeWrapperException

class KDDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(type(self).__name__)

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def __len__(self):
        raise NotImplementedError

    @property
    def root_dataset(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    def dispose(self):
        """  """
        pass