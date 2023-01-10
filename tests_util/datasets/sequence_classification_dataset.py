import torch

from kappadata.datasets.kd_dataset import KDDataset


class SequenceClassificationDataset(KDDataset):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.lengths = [len(c) for c in classes]

    def getitem_x(self, idx, _=None):
        return torch.arange(idx, idx + self.lengths[idx])

    def getitem_classes(self, idx, _=None):
        return self.classes[idx]

    def getitem_seqlen(self, idx, _=None):
        return self.lengths[idx]

    def __len__(self):
        return len(self.lengths)
