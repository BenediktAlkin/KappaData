import torch

from kappadata.datasets.kd_dataset import KDDataset


class SequenceIndexDataset(KDDataset):
    def __init__(self, lengths):
        super().__init__()
        self.lengths = lengths

    def getitem_x(self, idx, _=None):
        return torch.arange(idx, idx + self.lengths[idx])

    def __len__(self):
        return len(self.lengths)
