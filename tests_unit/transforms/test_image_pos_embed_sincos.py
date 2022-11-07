import unittest

import torch
from kappadata.transforms.image_pos_embed_sincos import ImagePosEmbedSincos

class TestImagePosEmbedSincos(unittest.TestCase):
    def test(self):
        rng = torch.Generator().manual_seed(5)
        x = torch.rand(3, 32, 32, generator=rng)

        transform = ImagePosEmbedSincos(dim=4)
        y = transform(x)
        self.assertEqual((7, 32, 32), y.shape)
