import torch

from torch.utils.data import Dataset

class ClassificationDatasetTorch(Dataset):
    def __init__(self, x, classes):
        super().__init__()
        assert len(x) == len(classes)
        self.x = x
        self.classes = classes

    def __getitem__(self, idx):
        x = self.x[idx]
        if torch.is_tensor(x):
            x = x.clone()
        y = self.classes[idx]
        return x, y

    def __len__(self):
        return len(self.classes)
