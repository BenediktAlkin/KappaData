# TODO include as testcase
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

from kappadata.utils.color_histogram import color_histogram


def main():
    root = Path("~/Documents/data").expanduser()
    images = [
        root / "ILSVRC2012_val_00046145.JPEG",
    ]
    images = torch.stack([to_tensor(default_loader(image)) for image in images]) * 255
    images = images.to(torch.device("cuda"))
    images = images.repeat(100, 1, 1, 1)

    hists = color_histogram(images, bins=128, density=True, batch_size=10)
    for i in range(3):
        plt.plot(range(len(hists[0][i])), hists[0][i].cpu())
        plt.show()


if __name__ == "__main__":
    main()
