from torchvision.datasets import ImageFolder
from argparse import ArgumentParser
from pathlib import Path
import kappadata as kd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    return vars(parser.parse_args())

def main(folder):
    folder = Path(folder).expanduser()
    assert folder.exists()
    transform = kd.KDComposeTransform([
        kd.KDRandomResizedCrop(size=224, interpolation="bicubic"),
        kd.KDRandomHorizontalFlip(),
        kd.KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        kd.KDRandomGaussianBlurPIL(p=0.5, sigma=[0.1, 2.0]),
    ])
    ds = ImageFolder(root=folder, transform=transform)
    for i in range(25):
        x, y = ds[i]
        x.save(f"img_{i}.png")


if __name__ == "__main__":
    main(**parse_args())