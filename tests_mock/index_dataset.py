from kappadata.wrappers.base.dataset_base import DatasetBase

class IndexDataset(DatasetBase):
    def __init__(self, size):
        super().__init__()
        self.size = size

    @staticmethod
    def getitem_x(idx, _=None):
        return idx

    def __len__(self):
        return self.size