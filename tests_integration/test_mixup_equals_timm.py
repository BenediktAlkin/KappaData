import torch
import unittest
from timm.data.mixup import Mixup
from tests_util.datasets import create_image_classification_dataset

class TestMixupEqualsTimm(unittest.TestCase):
    def test(self):
        dataset = create_image_classification_dataset(seed=5, size=100, channels=3, resolution=32, n_classes=10)
        mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=dataset.n_classes,
        )