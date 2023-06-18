import unittest

import numpy as np
import torch

from kappadata.wrappers.sample_wrappers import XRepeatWrapper
from tests_util.datasets.x_dataset import XDataset


class TestXRepeatWrapper(unittest.TestCase):
    def test_clone(self):
        ds = XDataset(x=torch.randn(10, 5, generator=torch.Generator().manual_seed(5)))
        ds = XRepeatWrapper(dataset=ds, num_repeats=2)
        x0 = ds.getitem_x(0)
        self.assertEqual(2, len(x0))
        self.assertTrue(torch.all(x0[0] == x0[1]))
        # inplace operation should only change one entry
        x0[0].add_(torch.ones_like(x0[0]))
        self.assertFalse(torch.all(x0[0] == x0[1]))
