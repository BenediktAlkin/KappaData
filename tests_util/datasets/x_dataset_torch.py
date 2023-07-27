import torch

from torch.utils.data import Dataset

class XDatasetTorch(Dataset):
    def __init__(self, x, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.transform = transform

    def __getitem__(self, idx):
        x = self.x[idx]
        if torch.is_tensor(x):
            x = x.clone()
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.x)
