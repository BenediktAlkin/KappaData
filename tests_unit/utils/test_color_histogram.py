import unittest

import torch

from kappadata.utils.color_histogram import color_histogram


class TestColorHistogram(unittest.TestCase):
    def test_uniform_bins(self):
        img = torch.arange(0, 256).reshape(1, 1, 16, 16)
        for bins in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            counts = color_histogram(img, bins=bins, density=False)
            self.assertTrue(all(counts[0] == count for count in counts[1:]))

    def test_uniform_bins_manual(self):
        img = torch.tensor([31.99, 32, 223.99, 224, 255])
        counts = color_histogram(img.reshape(1, 1, 1, len(img)), bins=8, density=False)
        self.assertEqual([1, 1, 0, 0, 0, 0, 1, 2], counts.squeeze().tolist())
