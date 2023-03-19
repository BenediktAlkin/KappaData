from pathlib import Path

from torchvision.datasets.folder import default_loader

import kappadata as kd


def main():
    root = Path("~/Documents/data/imagenet1k/val").expanduser()
    paths = [
        root / "n02056570" / "ILSVRC2012_val_00043795.JPEG",  # penguins
        root / "n02128757" / "ILSVRC2012_val_00010472.JPEG",  # leopard
        root / "n02690373" / "ILSVRC2012_val_00001952.JPEG",  # plane
        root / "n02690373" / "ILSVRC2012_val_00023626.JPEG",  # orange
    ]
    resize = kd.KDResize(size=[256, 256], interpolation="bicubic")
    imgs = [resize(default_loader(path)) for path in paths]
    raise NotImplementedError


if __name__ == "__main__":
    main()
