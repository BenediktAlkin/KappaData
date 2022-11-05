import unittest

import torch

from kappadata.wrappers.sample_wrappers.cutmix_wrapper import CutmixWrapper
from kappadata.wrappers.sample_wrappers.mixup_wrapper import MixupWrapper
from tests_util.classification_dataset import ClassificationDataset


class TestCutmixMixupCombination(unittest.TestCase):
    def test_getitem_class_manual(self):
        rng = torch.Generator().manual_seed(42)
        data = torch.randn(4, 1, 16, 16, generator=rng)
        classes = torch.randint(4, size=(4,), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mix_ds = MixupWrapper(dataset=CutmixWrapper(dataset=ds, alpha=1., p=1., seed=101), alpha=1., p=1., seed=1304)

        y0 = mix_ds.getitem_class(0)
        y1 = mix_ds.getitem_class(1)
        self.assertTrue(torch.allclose(torch.tensor([0., 0.9577113986, 0.0006607597, 0.0416278653]), y0))
        self.assertTrue(torch.allclose(torch.tensor([0., 0.9795139432, 0.0204860158, 0.]), y1))

    def test_getitem_class_automatic_noctx(self):
        rng = torch.Generator().manual_seed(42)
        data = torch.randn(16, 1, 8, 8, generator=rng)
        classes = torch.randint(4, size=(len(data),), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mix_ds = MixupWrapper(dataset=CutmixWrapper(dataset=ds, alpha=1., p=1., seed=101), alpha=1., p=1., seed=1304)

        max_nonzero_class_prob_count = 0
        for i in range(len(data)):
            y = mix_ds.getitem_class(i)
            self.assertTrue(torch.allclose(torch.tensor(1.), y.sum()))
            nonzero_class_prob_count = (y != 0.).sum()
            max_nonzero_class_prob_count = max(max_nonzero_class_prob_count, nonzero_class_prob_count)
            self.assertLessEqual(nonzero_class_prob_count, 4)
        self.assertEqual(4, max_nonzero_class_prob_count)
