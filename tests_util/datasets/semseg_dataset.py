import torch

from kappadata.datasets.kd_dataset import KDDataset


class SemsegDataset(KDDataset):
    def __init__(self, x, semseg):
        super().__init__()
        assert len(x) == len(semseg)
        self.x = x
        self.semseg = semseg

    # noinspection PyUnusedLocal
    def getitem_x(self, idx, ctx=None):
        x = self.x[idx]
        if torch.is_tensor(x):
            x = x.clone()
        return x

    # noinspection PyUnusedLocal
    def getitem_semseg(self, idx, ctx=None):
        return self.semseg[idx].clone()

    def __len__(self):
        return len(self.x)
