import unittest

import torch
import torch.nn as nn

from kappadata.utils.multi_crop_utils import (
    multi_crop_split_forward,
    multi_crop_joint_forward,
    MultiCropSplitForwardModule,
    MultiCropJointForwardModule,
)
from tests_util.modules.memorize_shape_module import MemorizeShapeModule


class TestMultiCropUtils(unittest.TestCase):
    class MultiCropModule(nn.Module):
        def __init__(self, n_views=None):
            super().__init__()
            self.n_views = n_views
            self.layer0 = MemorizeShapeModule(nn.Linear(4, 8, bias=False))
            self.layer1 = MemorizeShapeModule(nn.Linear(8, 6, bias=False))

        def forward(self, x):
            x = multi_crop_joint_forward(self.layer0, x)
            x = multi_crop_split_forward(self.layer1, x, n_chunks=self.n_views)
            return x

    def test_tensor_2views(self):
        model = self.MultiCropModule(n_views=2)
        model(torch.ones(10, 4))
        self.assertEqual((10, 4), model.layer0.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[1])

    def test_list_2views(self):
        model = self.MultiCropModule()
        model(torch.ones(10, 4).chunk(2))
        self.assertEqual((10, 4), model.layer0.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[1])

    def test_list_2views_sequential(self):
        model = nn.Sequential(
            MultiCropJointForwardModule(MemorizeShapeModule(nn.Linear(4, 8, bias=False))),
            MultiCropSplitForwardModule(MemorizeShapeModule(nn.Linear(8, 6, bias=False))),
        )
        model(torch.ones(10, 4).chunk(2))
        self.assertEqual((10, 4), model[0].module.shapes[0])
        self.assertEqual((5, 8), model[1].module.shapes[0])
        self.assertEqual((5, 8), model[1].module.shapes[1])
