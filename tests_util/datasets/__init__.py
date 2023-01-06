import torch

from .classification_dataset import ClassificationDataset


def create_image_classification_dataset(size, seed, channels=3, resolution=32, n_classes=10) -> ClassificationDataset:
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand(size, channels, resolution, resolution, generator=rng)
    classes = torch.randint(n_classes, size=(size,), generator=rng)
    return ClassificationDataset(x=x, classes=classes)
