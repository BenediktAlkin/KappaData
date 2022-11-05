import unittest

import torch
from torch.utils.data import DataLoader

from kappadata.collators.base.kd_collator import KDCollator
from kappadata.collators.cutmix_collator import CutmixCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.classification_dataset import ClassificationDataset


class TestCutmixCollator(unittest.TestCase):
    def test(self):
        rng = torch.Generator().manual_seed(1235)
        x = torch.randn(4, 1, 8, 8, generator=rng)
        n_classes = 4
        classes = torch.randint(n_classes, size=(len(x),), generator=rng)
        ds = ClassificationDataset(x=x, classes=classes)
        ds_mode = "x class"
        ds = ModeWrapper(dataset=ds, mode=ds_mode)

        mix_collator = CutmixCollator(alpha=1., p=1., seed=3, n_classes=n_classes, dataset_mode=ds_mode)
        collator = KDCollator(collators=[mix_collator])
        dl = DataLoader(ds, batch_size=len(x), collate_fn=collator)
        _ = next(iter(dl))
        # TODO
