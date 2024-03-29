import torch

from kappadata.datasets.kd_dataset import KDDataset


class ClassificationDataset(KDDataset):
    def __init__(self, x, classes):
        super().__init__()
        assert len(x) == len(classes)
        self.x = x
        self.classes = classes

    # noinspection PyUnusedLocal
    def getitem_x(self, idx, ctx=None):
        x = self.x[idx]
        if torch.is_tensor(x):
            x = x.clone()
        return x

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.classes[idx]

    def getshape_class(self):
        max_class = max(self.classes)
        if torch.is_tensor(max_class):
            max_class = max_class.item()
        return max_class + 1,

    def __len__(self):
        return len(self.classes)
