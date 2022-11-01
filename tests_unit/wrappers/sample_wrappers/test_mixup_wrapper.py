import torch
import unittest
from tests_util.classification_dataset import ClassificationDataset
from kappadata.wrappers.sample_wrappers.mixup_wrapper import MixupWrapper
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.utils.data import DataLoader

class TestMixupWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha="a", p=1.))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=None, p=1.))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=-0.1, p=1.))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=1., p="a"))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=1., p=None))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=1., p=0.))
        self.assertRaises(AssertionError, lambda: MixupWrapper(dataset=None, alpha=1., p=-2.))
        _ = MixupWrapper(dataset=ClassificationDataset(x=torch.randn(2, 1), classes=list(range(2))), alpha=1., p=1.)

    def test_getitem_x_class(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(2, 4, generator=rng)
        classes = torch.randint(4, size=(2,), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mixup_ds = MixupWrapper(dataset=ds, alpha=1., p=1., seed=103)
        loader = DataLoader(ModeWrapper(mixup_ds, mode="x class", return_ctx=True), batch_size=len(mixup_ds))
        (x, y), ctxs = next(iter(loader))
        lamb = ctxs["mixup_lambda"]
        idx2 = ctxs["mixup_idx2"]
        self.assertTrue(torch.all(idx2 == torch.tensor([1, 1])))
        self.assertTrue(torch.all(lamb[0] * data[0] + (1. - lamb[0]) * data[1] == x[0]))
        # ds[1] is "interpolated" with itself (floating point errors occour)
        self.assertTrue(torch.allclose(data[1], x[1]))
        self.assertTrue(torch.all(lamb[1] * data[1] + (1. - lamb[1]) * data[1] == x[1]))


    def test_getitem_class(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(2, 1, 4, 4, generator=rng)
        classes = torch.randint(4, size=(2,), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mixup_ds = MixupWrapper(dataset=ds, alpha=1., p=1., seed=621413)
        y0 = mixup_ds.getitem_class(0)
        y1 = mixup_ds.getitem_class(1)
        self.assertEquals([0, 1, 0], y0.tolist())
        self.assertTrue(torch.allclose(torch.tensor([0, 0.1406126618, 0.8593873382]), y1))

    def test_getitem_class_automatic_noctx(self):
        rng = torch.Generator().manual_seed(42)
        data = torch.randn(16, 1, 8, 8, generator=rng)
        classes = torch.randint(4, size=(len(data),), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        mixup_ds = MixupWrapper(dataset=ds, alpha=1., p=1., seed=101)

        max_nonzero_class_prob_count = 0
        for i in range(len(data)):
            y = mixup_ds.getitem_class(i)
            self.assertTrue(torch.allclose(torch.tensor(1.), y.sum()))
            nonzero_class_prob_count = (y != 0.).sum()
            max_nonzero_class_prob_count = max(max_nonzero_class_prob_count, nonzero_class_prob_count)
            self.assertLessEqual(nonzero_class_prob_count, 2)
        self.assertEqual(2, max_nonzero_class_prob_count)