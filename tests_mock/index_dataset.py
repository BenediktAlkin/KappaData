from kappadata.wrappers.base.dataset_base import DatasetBase

class IndexDataset(DatasetBase):
    def __init__(self, size):
        super().__init__()
        self.size = size

    @staticmethod
    def idxget_x(idx, ctx=None):
        return idx

    @staticmethod
    def idxget_class(idx, ctx=None):
        return idx

    def __len__(self):
        return self.size