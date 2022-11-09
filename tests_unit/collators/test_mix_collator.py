import unittest

import torch
from torch.utils.data import DataLoader

from kappadata.collators.base.kd_compose_collator import KDComposeCollator
from kappadata.collators.mix_collator import MixCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.classification_dataset import ClassificationDataset


class TestMixCollator(unittest.TestCase):
    def test(self):
        rng = torch.Generator().manual_seed(1235)
        x = torch.randn(4, 1, 8, 8, generator=rng)
        n_classes = 4
        classes = torch.randint(n_classes, size=(len(x),), generator=rng)
        ds = ClassificationDataset(x=x, classes=classes)
        ds_mode = "x class"
        ds = ModeWrapper(dataset=ds, mode=ds_mode)

        mix_collator = MixCollator(
            cutmix_alpha=1.,
            mixup_alpha=1.,
            cutmix_p=0.5,
            p=1.,
            mode="batch",
            seed=3,
            n_classes=n_classes,
        )
        collator = KDComposeCollator(collators=[mix_collator], dataset_mode=ds_mode)
        dl = DataLoader(ds, batch_size=len(x), collate_fn=collator)
        _ = next(iter(dl))
        # TODO
