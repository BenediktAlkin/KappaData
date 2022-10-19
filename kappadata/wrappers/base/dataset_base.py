from torch.utils.data import Dataset
import logging

class DatasetBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(type(self).__name__)

    def __getitem__(self, index):
        raise RuntimeError("use kappadata.wrappers.ModeWrapper instead of __getitem__ directly")

    def __len__(self):
        raise NotImplementedError

    @property
    def root_dataset(self):
        return self