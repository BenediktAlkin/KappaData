import torch

from kappadata.datasets.kd_dataset import KDDataset


class XDataset(KDDataset):
    def __init__(self, x, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.transform = transform

    def getitem_x(self, idx, ctx=None):
        x = self.x[idx]
        if torch.is_tensor(x):
            x = x.clone()
        if self.transform is not None:
            x = self.transform(x, ctx)
        return x

    def __len__(self):
        return len(self.x)
