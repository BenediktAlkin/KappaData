import torch
import unittest
from tests_util.classification_dataset import ClassificationDataset
from kappadata.wrappers.sample_wrappers.mixup_wrapper import MixupWrapper
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.utils.data import DataLoader

class TestMixupWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha="a"))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=None))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=-0.1))
        _ = MixupWrapper(dataset=ClassificationDataset(x=torch.randn(2, 1), classes=list(range(2))), alpha=1.)

    def test_mixup(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(2, 4, generator=rng)
        classes = torch.randint(4, size=(2,), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mixup_ds = MixupWrapper(dataset=ds, alpha=1., seed=39)
        loader = DataLoader(ModeWrapper(mixup_ds, mode="x class", return_ctx=True), batch_size=len(mixup_ds))
        (x, y), ctxs = next(iter(loader))
        lamb = ctxs["mixup_lambda"]
        idx2 = ctxs["mixup_idx2"]
        self.assertTrue(torch.all(idx2 == torch.tensor([1, 1])))
        self.assertTrue(torch.all(lamb[0] * data[0] + (1. - lamb[0]) * data[1] == x[0]))
        # ds[1] is "interpolated" with itself (floating point errors occour)
        self.assertTrue(torch.allclose(data[1], x[1]))
        self.assertTrue(torch.all(lamb[1] * data[1] + (1. - lamb[1]) * data[1] == x[1]))