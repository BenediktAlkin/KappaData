import unittest

import torch

from kappadata.transforms.image_pos_embed_grid import ImagePosEmbedGrid


class TestImagePosEmbedSincos(unittest.TestCase):
    def test_shape(self):
        rng = torch.Generator().manual_seed(5)
        x = torch.rand(3, 32, 32, generator=rng)

        transform = ImagePosEmbedGrid()
        y = transform(x)
        self.assertEqual((5, 32, 32), y.shape)

    def test_minmax(self):
        rng = torch.Generator().manual_seed(5)
        x = torch.rand(3, 32, 32, generator=rng)

        transform = ImagePosEmbedGrid()
        y = transform(x)
        self.assertEqual(-1., y[3].min())
        self.assertEqual(1., y[3].max())
        self.assertEqual(-1., y[4].min())
        self.assertEqual(1., y[4].max())
