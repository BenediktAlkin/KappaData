import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToPILImage
from torchvision.transforms.functional import to_tensor, to_pil_image

import kappadata as kd
from kappadata.utils.save_image import concat_images_square

from kappadata.transforms.audio import KDSpecAugment
import platform

def main():
    if platform.uname().node == "DESKTOP-QVRSC79":
        import matplotlib
        matplotlib.use('TkAgg')

    x = torch.randn(1, 128, 1024, generator=torch.Generator().manual_seed(0)) * 10
    t = KDSpecAugment(frequency_masking=48, time_masking=192, shape_mode="cft").set_rng(np.random.default_rng(seed=0))
    for _ in range(10):
        y = t(x)
        plt.imshow(y.squeeze(0))
        plt.show()

if __name__ == "__main__":
    main()
