import unittest

import torch
from torch.utils.data import DataLoader

from kappadata.collators.base.kd_compose_collator import KDComposeCollator
from kappadata.collators.mix_collator import MixCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from tests_util.datasets import create_image_classification_dataset


class TestMixCollator(unittest.TestCase):
    def test(self):
        ds = create_image_classification_dataset(size=16, seed=1235, channels=1, resolution=8, n_classes=4)
        ds_mode = "x class"
        ds = ModeWrapper(dataset=ds, mode=ds_mode)

        mix_collator = MixCollator(
            cutmix_alpha=1.,
            mixup_alpha=1.,
            cutmix_p=0.5,
            p=1.,
            mode="batch",
            seed=3,
            n_classes=ds.n_classes,
        )
        collator = KDComposeCollator(collators=[mix_collator], dataset_mode=ds_mode)
        dl = DataLoader(ds, batch_size=len(ds), collate_fn=collator)
        _ = next(iter(dl))
        # TODO
