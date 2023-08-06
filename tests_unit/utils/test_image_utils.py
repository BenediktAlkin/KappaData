import unittest

import torch
from torchvision.transforms.functional import to_pil_image

from kappadata.utils.image_utils import get_dimensions


class TestImageUtils(unittest.TestCase):
    def test_pil_eqals_torch_rgb(self):
        shape = (3, 16, 32)
        tensor = torch.ones(shape)
        pil = to_pil_image(tensor)
        self.assertEqual(shape, get_dimensions(tensor))
        self.assertEqual(shape, get_dimensions(pil))

    def test_pil_eqals_torch_greyscale(self):
        shape = (1, 16, 32)
        tensor = torch.ones(shape)
        pil = to_pil_image(tensor)
        self.assertEqual(shape, get_dimensions(tensor))
        self.assertEqual(shape, get_dimensions(pil))
