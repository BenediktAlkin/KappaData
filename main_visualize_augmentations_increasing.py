import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Resize, ToPILImage
from torchvision.transforms.functional import to_tensor, to_pil_image

import kappadata as kd
from kappadata.utils.save_image import concat_images_square




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
