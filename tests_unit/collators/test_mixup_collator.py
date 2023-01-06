import unittest

import torch
from torch.utils.data import DataLoader

from kappadata.collators.base.kd_compose_collator import KDComposeCollator
from kappadata.collators.mixup_collator import MixupCollator
from kappadata.functional.onehot import to_onehot_matrix
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.datasets import create_image_classification_dataset


class TestMixupCollator(unittest.TestCase):
    def test_automatic_p1(self):
        ds = create_image_classification_dataset(size=16, seed=19521, channels=1, resolution=8, n_classes=4)
        ds_mode = "x class"
        ds = ModeWrapper(dataset=ds, mode=ds_mode, return_ctx=True)

        # mixup
        mix_collator = MixupCollator(alpha=1., p=1., seed=3, n_classes=ds.n_classes)
        collator = KDComposeCollator(collators=[mix_collator], dataset_mode=ds_mode, return_ctx=True)
        dl = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        (x, y), ctx = next(iter(dl))
        lamb = ctx["mixup_lambda"]
        lamb_x = lamb.view(-1, 1, 1, 1)
        lamb_y = lamb.view(-1, 1)

        # check x
        expected_x = ds.x * lamb_x + ds.x.roll(shifts=1, dims=0) * (1. - lamb_x)
        self.assertTrue(torch.all(x == expected_x))

        # check y
        og_y = to_onehot_matrix(ds.classes, n_classes=ds.n_classes)
        expected_y = og_y * lamb_y + og_y.roll(shifts=1, dims=0) * (1. - lamb_y)
        self.assertTrue(torch.all(y == expected_y))

        # check y has probability mass at most 2 entries and it sums to 1
        n_probabilty_masses = (y > 0).sum(dim=1)
        self.assertTrue(torch.all(torch.logical_or(n_probabilty_masses == 1, n_probabilty_masses == 2)))
        self.assertTrue(torch.all(y.sum(dim=1) == 1.))
