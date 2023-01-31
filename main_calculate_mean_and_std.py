import einops
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from kappadata import TorchWrapper, ModeWrapper, KDDataset
import torch
from torch.utils.data import DataLoader

def calculate_mean_and_std(dataset, batch_size=64, num_workers=0):
    assert isinstance(dataset, KDDataset), "use KDDataset with getitem_x"
    x0 = dataset.getitem_x(0)
    assert torch.is_tensor(x0) and x0.ndim == 3, "dataset should return a 3D tensor (image tensor)"
    loader = DataLoader(ModeWrapper(dataset=dataset, mode="x"), batch_size=batch_size, num_workers=num_workers)
    count = 0
    mean = torch.zeros(3)
    for x in loader:
        # https://stackoverflow.com/a/23493727/13253318
        count += len(x)
        mean = mean * (count - len(x)) / count + x.mean(dim=[2, 3]).sum(dim=0) / count

    var = torch.zeros(3)
    for x in loader:
        var += (((x - mean.view(1, -1, *(1,) * (x.ndim - 2))) ** 2).mean(dim=[2, 3]) / len(dataset)).sum(dim=0)
    std = var ** 0.5

    return mean, std


def main():
    dataset = CIFAR100(root="~/Documents/data/cifar10", download=True, transform=ToTensor(), train=True)
    dataset = TorchWrapper(dataset=dataset, mode="x class")
    mean, std = calculate_mean_and_std(dataset)
    print(mean, std)


if __name__ == "__main__":
    main()