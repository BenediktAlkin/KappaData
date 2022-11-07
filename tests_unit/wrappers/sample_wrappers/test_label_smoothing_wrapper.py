import unittest

import torch

from kappadata.wrappers.sample_wrappers.label_smoothing_wrapper import LabelSmoothingWrapper
from tests_util.class_dataset import ClassDataset


class TestLabelSmoothingWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing="a"))
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing=None))
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing=-0.1))
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing=1.))
        _ = LabelSmoothingWrapper(dataset=ClassDataset(classes=list(range(2))), smoothing=0.5)

    def test_getitem_class_automatic(self):
        rng = torch.Generator().manual_seed(42)
        n_classes = 10
        classes = torch.randint(n_classes, size=(100,), generator=rng)
        ds = ClassDataset(classes=classes)
        smoothing = .1
        ds = LabelSmoothingWrapper(dataset=ds, smoothing=smoothing)
        for i in range(len(ds)):
            y = ds.getitem_class(i)
            self.assertTrue(torch.allclose(torch.tensor(1.), y.sum()))
            unique = y.unique()
            self.assertEqual(2., len(unique))
            self.assertTrue(torch.allclose(torch.tensor(1. - smoothing / n_classes * (n_classes - 1)), unique.max()))
            self.assertTrue(torch.allclose(torch.tensor(smoothing / n_classes), unique.min()))
            self.assertEqual(y.argmax(), classes[i])
