import unittest

import torch

from kappadata.wrappers.sample_wrappers.label_smoothing_wrapper import LabelSmoothingWrapper
from tests_util.datasets import create_class_dataset
from tests_util.datasets.class_dataset import ClassDataset


class TestLabelSmoothingWrapper(unittest.TestCase):
    def test_ctor_arg_checks(self):
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing="a"))
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing=None))
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing=-0.1))
        self.assertRaises(AssertionError, lambda: LabelSmoothingWrapper(dataset=None, smoothing=1.))
        _ = LabelSmoothingWrapper(dataset=ClassDataset(classes=list(range(2))), smoothing=0.5)

    def test_getitem_class_automatic(self):
        smoothing = .1
        ds = create_class_dataset(size=100, n_classes=10, seed=42)
        smoothed_ds = LabelSmoothingWrapper(dataset=ds, smoothing=smoothing)
        for i in range(len(ds)):
            y = smoothed_ds.getitem_class(i)
            self.assertTrue(torch.allclose(torch.tensor(1.), y.sum()))
            unique = y.unique()
            self.assertEqual(2., len(unique))
            on_value = torch.tensor(1. - smoothing / ds.n_classes * (ds.n_classes - 1))
            self.assertTrue(torch.allclose(on_value, unique.max()))
            self.assertTrue(torch.allclose(torch.tensor(smoothing / ds.n_classes), unique.min()))
            self.assertEqual(y.argmax(), ds.getitem_class(i))
