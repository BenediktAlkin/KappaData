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
    for root in [
        Path("~/Documents/data/ILSVRC2012_val_00046145.JPEG").expanduser(),
        Path("~/Documents/data/imagenet1k/val/n02087394/ILSVRC2012_val_00019209.JPEG").expanduser(),
        Path("~/Documents/data/imagenet1k/val/n02087394/ILSVRC2012_val_00000102.JPEG").expanduser(),
    ]:
        img = default_loader(root)
        img = kd.KDResize(size=[256, 256], interpolation="bicubic")(img)
        for name, transform in [
            ("original", lambda x: x),
        ]:
            transform(img).save(f"temp/{name}.{root.name}.png")


if __name__ == "__main__":
    main()
