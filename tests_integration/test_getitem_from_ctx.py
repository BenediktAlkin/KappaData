import unittest

import torch
from torch.utils.data import DataLoader

from kappadata.transforms.save_state_to_context_transform import SaveStateToContextTransform
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.datasets.x_dataset import XDataset


class TestGetitemFromCtx(unittest.TestCase):
    def test_save_state_to_context_transform(self):
        ds = XDataset(x=torch.randn(8, 3, 4, 4), transform=SaveStateToContextTransform(state_name="og"))
        wrapper = ModeWrapper(dataset=ds, mode="x ctx.og", return_ctx=False)
        x, og = next(iter(DataLoader(wrapper, batch_size=len(wrapper))))
        self.assertTrue(torch.all(x == og))
