from kappadata.datasets.kd_dataset import KDDataset


class IndexDataset(KDDataset):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.indices = list(range(size))
        self.disposed = False

    def getitem_x(self, idx, _=None):
        return self.indices[idx]

    def __len__(self):
        return self.size

    def dispose(self):
        self.disposed = True
