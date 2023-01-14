from kappadata.datasets.kd_dataset import KDDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

class KDImageFolder(KDDataset):
    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = ImageFolder(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    # noinspection PyUnusedLocal
    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        return x

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.dataset.targets[idx]

    def __len__(self):
        return len(self.dataset)
