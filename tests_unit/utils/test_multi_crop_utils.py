import unittest

import torch
import torch.nn as nn

from kappadata.utils.multi_crop_utils import (
    split_forward,
    SplitForwardModule,
    concat_same_shape_inputs,
    split_same_shape_inputs,
)
from tests_util.modules.memorize_shape_module import MemorizeShapeModule


class TestMultiCropUtils(unittest.TestCase):
    class MultiCropModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = MemorizeShapeModule(nn.Linear(4, 8, bias=False))
            self.layer1 = MemorizeShapeModule(nn.Linear(8, 6, bias=False))

        def forward(self, x, batch_size):
            x = self.layer0(x)
            x = split_forward(self.layer1, x, batch_size=batch_size)
            return x

    def test_tensor_2views(self):
        model = self.MultiCropModule()
        model(torch.ones(10, 4), batch_size=5)
        self.assertEqual((10, 4), model.layer0.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[1])

    def test_list_2views(self):
        model = self.MultiCropModule()
        model(torch.ones(10, 4), batch_size=5)
        self.assertEqual((10, 4), model.layer0.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[1])

    def test_concat_same_shape_inputs_noop(self):
        actual, batch_size = concat_same_shape_inputs(torch.randn(5, 10))
        self.assertEqual(5, batch_size)
        self.assertEqual((5, 10), actual[0].shape)

    def test_split_same_shape_inputs_noop(self):
        actual = split_same_shape_inputs(torch.randn(15, 10), batch_size=5)
        self.assertEqual(1, len(actual))
        self.assertEqual((15, 10), actual[0].shape)

    def test_concat_same_shape_inputs(self):
        x = [
            torch.randn(5, 10),
            torch.randn(5, 10),
            torch.randn(5, 8),
            torch.randn(5, 10),
            torch.randn(5, 12),
            torch.randn(5, 8),
        ]
        actual, batch_size = concat_same_shape_inputs(x)
        self.assertEqual(3, len(actual))
        self.assertEqual(5, batch_size)
        self.assertEqual((15, 10), actual[0].shape)
        self.assertEqual((10, 8), actual[1].shape)
        self.assertEqual((5, 12), actual[2].shape)

    def test_split_same_shape_inputs(self):
        actual = split_same_shape_inputs([torch.randn(15, 10), torch.randn(10, 8)], batch_size=5)
        self.assertEqual(5, len(actual))
        self.assertEqual((5, 10), actual[0].shape)
        self.assertEqual((5, 10), actual[1].shape)
        self.assertEqual((5, 10), actual[2].shape)
        self.assertEqual((5, 8), actual[3].shape)
        self.assertEqual((5, 8), actual[4].shape)