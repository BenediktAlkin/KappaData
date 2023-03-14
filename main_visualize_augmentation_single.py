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
    img = default_loader(Path("~/Documents/data/ILSVRC2012_val_00046145.JPEG").expanduser())
    img = kd.KDResize(size=[256, 256], interpolation="bicubic")(img)
    for name, transform in [
        ("original", lambda x: x),
        ("blur", kd.KDGaussianBlurPIL(sigma=2.)),
        ("greyscale", kd.KDRandomGrayscale(p=1.)),
        ("jitter", kd.KDColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)),
        ("solarize", kd.KDRandomSolarize(p=1., threshold=128)),
        ("all", kd.KDComposeTransform([
            kd.KDColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            kd.KDGaussianBlurPIL(sigma=2.),
            kd.KDRandomSolarize(p=1., threshold=128),
            #kd.KDRandomGrayscale(p=1.),
        ]))
    ]:
        transform(img).save(f"temp/{name}.png")


if __name__ == "__main__":
    main()
