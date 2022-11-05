import unittest
from torch.utils.data import DataLoader
from tests_util.classification_dataset import ClassificationDataset
import torch
from kappadata.collators.mixup_collator import MixupCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from kappadata.collators.custom_collator import CustomCollator

class TestMixupCollator(unittest.TestCase):
    def test(self):
        rng = torch.Generator().manual_seed(1235)
        x = torch.randn(4, 1, 8, 8, generator=rng)
        n_classes = 4
        classes = torch.randint(n_classes, size=(len(x),), generator=rng)
        ds = ClassificationDataset(x=x, classes=classes)
        ds = ModeWrapper(dataset=ds, mode="x class")

        collator = CustomCollator(collators=[MixupCollator(alpha=1., p=1., seed=3, n_classes=n_classes)])
        dl = DataLoader(ds, batch_size=len(x), collate_fn=collator)
        asdf = next(iter(dl))
        # TODO