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

    def test_getitem_class_semisupervised_float(self):
        ds = LabelSmoothingWrapper(dataset=ClassDataset(classes=[0, -1, 1, 2, 3]), smoothing=.1)
        self.assertEqual(torch.float32, ds.getitem_class(1).dtype)

    def test_getitem_class_semisupvervised(self):
        ds = LabelSmoothingWrapper(dataset=ClassDataset(classes=[0, -1, 1, -1, 2, 3, -1]), smoothing=.1)
        expected = [
            [0.925, 0.025, 0.025, 0.025],
            [-1, -1, -1, -1],
            [0.025, 0.925, 0.025, 0.025],
            [-1, -1, -1, -1],
            [0.025, 0.025, 0.925, 0.025],
            [0.025, 0.025, 0.025, 0.925],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ]
        for i in range(len(ds)):
            # use allclose because of floating point precision errors
            self.assertTrue(torch.allclose(torch.tensor(expected[i]), ds.getitem_class(i)))


    def test_getitem_class_automatic(self):
        smoothing = .1
        ds = create_class_dataset(size=100, n_classes=10, seed=42)
        smoothed_ds = LabelSmoothingWrapper(dataset=ds, smoothing=smoothing)
        for i in range(len(ds)):
            y = smoothed_ds.getitem_class(i)
            self.assertTrue(torch.allclose(torch.tensor(1.), y.sum()))
            unique = y.unique()
            self.assertEqual(2., len(unique))
            on_value = torch.tensor(1. - smoothing / ds.getdim_class() * (ds.getdim_class() - 1))
            self.assertTrue(torch.allclose(on_value, unique.max()))
            self.assertTrue(torch.allclose(torch.tensor(smoothing / ds.getdim_class()), unique.min()))
            self.assertEqual(y.argmax(), ds.getitem_class(i))