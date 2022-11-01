import torch
import unittest
from tests_util.classification_dataset import ClassificationDataset
from kappadata.wrappers.sample_wrappers.cutmix_wrapper import CutmixWrapper
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.utils.data import DataLoader

class TestCutmixWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha="a", p=1.))
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha=None, p=1.))
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha=-0.1, p=1.))
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha=1., p="a"))
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha=1., p=None))
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha=1., p=0.))
        self.assertRaises(AssertionError, lambda: CutmixWrapper(dataset=None, alpha=1., p=-2.))
        _ = CutmixWrapper(dataset=ClassificationDataset(x=torch.randn(2, 1), classes=list(range(2))), alpha=1., p=1.)

    def test_getitem_x_class(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(2, 1, 4, 4, generator=rng)
        classes = torch.randint(4, size=(2,), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        cutmix_ds = CutmixWrapper(dataset=ds, alpha=1., p=1., seed=103)
        loader = DataLoader(ModeWrapper(cutmix_ds, mode="x class", return_ctx=True), batch_size=len(cutmix_ds))
        (x, y), ctxs = next(iter(loader))
        lamb = ctxs["cutmix_lambda"]
        idx2 = ctxs["cutmix_idx2"]
        top, left, bot, right = ctxs["cutmix_bbox"]
        self.assertTrue(torch.all(idx2 == torch.tensor([1, 1])))

        # check replaced patch
        replaced_expected = data[1, :, top[0]:bot[0], left[0]:right[0]]
        replaced_actual = x[0, :, top[0]:bot[0], left[0]:right[0]]
        self.assertTrue(torch.all(replaced_expected == replaced_actual))
        # check original patch
        self.assertTrue(torch.all(data[0, :, :top[0], :] == x[0, :, :top[0], :]))
        self.assertTrue(torch.all(data[0, :, bot[0]:, :] == x[0, :, bot[0]:, :]))
        self.assertTrue(torch.all(data[0, :, :, :left[0]] == x[0, :, :, :left[0]]))
        self.assertTrue(torch.all(data[0, :, :, right[0]:] == x[0, :, :, right[0]:]))
        # ds[1] is "interpolated" with itself
        self.assertTrue(torch.all(data[1] == x[1]))
        self.assertTrue(torch.all(lamb[1] * data[1] + (1. - lamb[1]) * data[1] == x[1]))

    def test_getitem_class(self):
        rng = torch.Generator().manual_seed(63)
        data = torch.randn(2, 1, 4, 4, generator=rng)
        classes = torch.randint(4, size=(2,), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        cutmix_ds = CutmixWrapper(dataset=ds, alpha=1., p=1., seed=1073)
        y0 = cutmix_ds.getitem_class(0)
        y1 = cutmix_ds.getitem_class(1)
        self.assertEquals([0, 1, 0], y0.tolist())
        self.assertEquals([0, 0.125, 0.875], y1.tolist())

    def test_getitem_class_automatic_noctx(self):
        rng = torch.Generator().manual_seed(42)
        data = torch.randn(16, 1, 8, 8, generator=rng)
        classes = torch.randint(4, size=(len(data),), generator=rng)
        ds = ClassificationDataset(x=data, classes=classes)
        cutmix_ds = CutmixWrapper(dataset=ds, alpha=1., p=1., seed=101)

        max_nonzero_class_prob_count = 0
        for i in range(len(data)):
            y = cutmix_ds.getitem_class(i)
            self.assertTrue(torch.allclose(torch.tensor(1.), y.sum()))
            nonzero_class_prob_count = (y != 0.).sum()
            max_nonzero_class_prob_count = max(max_nonzero_class_prob_count, nonzero_class_prob_count)
            self.assertLessEqual(nonzero_class_prob_count, 2)
        self.assertEqual(2, max_nonzero_class_prob_count)