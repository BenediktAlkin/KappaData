from kappadata.datasets.kd_dataset import KDDataset

class IndexDataset(KDDataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    @staticmethod
    def getitem_x(idx, _=None):
        return idx

    def __len__(self):
        return self.size