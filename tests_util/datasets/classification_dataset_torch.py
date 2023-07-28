import torch

from torch.utils.data import Dataset

class ClassificationDatasetTorch(Dataset):
    def __init__(self, x, classes, transform=None):
        super().__init__()
        assert len(x) == len(classes)
        self.x = x
        self.classes = classes
        self.transform = transform

    def __getitem__(self, idx):
        x = self.x[idx]
        if torch.is_tensor(x):
            x = x.clone()
        if self.transform is not None:
            x = self.transform(x)
        y = self.classes[idx]
        return x, y

    def __len__(self):
        return len(self.classes)
