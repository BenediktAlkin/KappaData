# TODO include as testcase
import torch
from kappadata.utils.color_histogram import color_histogram
from torchvision.datasets.folder import default_loader
from pathlib import Path
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

def main():
    root = Path("~/Documents/data").expanduser()
    images = [
        root / "ILSVRC2012_val_00046145.JPEG",
    ]
    images = torch.stack([to_tensor(default_loader(image)) for image in images]) * 255


    hists = color_histogram(images, bins=32, density=True)
    for i in range(3):
        plt.plot(range(len(hists[0][i])), hists[0][i])
        plt.show()


if __name__ == "__main__":
    main()
