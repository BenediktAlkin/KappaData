import torch

from .class_dataset import ClassDataset
from .classification_dataset import ClassificationDataset


def create_image_classification_dataset(size, seed, channels=3, resolution=32, n_classes=10) -> ClassificationDataset:
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand(size, channels, resolution, resolution, generator=rng)
    classes = torch.randint(n_classes, size=(size,), generator=rng)
    return ClassificationDataset(x=x, classes=classes)


def create_class_dataset(size, n_classes, seed) -> ClassDataset:
    rng = torch.Generator().manual_seed(seed)
    classes = torch.randint(n_classes, size=(size,), generator=rng)
    return ClassDataset(classes=classes, n_classes=n_classes)
