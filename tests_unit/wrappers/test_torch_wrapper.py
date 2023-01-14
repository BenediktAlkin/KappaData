import unittest

import torch
from torch.utils.data import Dataset
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.wrappers.torch_wrapper import TorchWrapper


class TestTorchWrapper(unittest.TestCase):
    class TorchDataset(Dataset):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

        def __len__(self):
            return len(self.x)

    def test(self):
        rng = torch.Generator().manual_seed(5)
        x = torch.randn(10, 5, generator=rng)
        y = torch.randint(15, size=(10,))
        ds = TorchWrapper(dataset=self.TorchDataset(x=x, y=y), mode="x class")
        ds = ModeWrapper(dataset=ds, mode="x", return_ctx=False)
        self.assertEqual(x.tolist(), torch.stack(ds[:]).tolist())
