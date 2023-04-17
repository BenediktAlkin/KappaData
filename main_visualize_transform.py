import matplotlib.pyplot as plt
import torch
from kappadata.visualization.visualize_transform import visualize_transform
from pathlib import Path
from torchvision.datasets.folder import default_loader
from argparse import ArgumentParser
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_gaussian_blur_pil import KDGaussianBlurPIL
from kappadata.transforms.kd_random_color_jitter import KDRandomColorJitter
from kappadata.transforms.kd_random_gaussian_blur_pil import KDRandomGaussianBlurPIL
from kappadata.transforms.kd_random_grayscale import KDRandomGrayscale
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from kappadata.transforms.kd_random_solarize import KDRandomSolarize

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--repeat", default=100, type=int)
    return vars(parser.parse_args())


def main(root, repeat):
    # load image
    root = Path(root).expanduser()
    img = default_loader(root)

    size = 300
    # solarize
    # transform = KDComposeTransform([
    #     KDRandomSolarize(p=1., threshold=128),
    # ])
    # byol0
    # transform = KDComposeTransform([
    #     KDRandomHorizontalFlip(),
    #     KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    #     KDGaussianBlurPIL(sigma=(0.1, 2.0)),
    #     KDRandomGrayscale(p=0.2),
    # ])
    # byol1
    transform = KDComposeTransform([
        KDRandomHorizontalFlip(),
        KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        KDRandomGaussianBlurPIL(p=0.1, sigma=(0.1, 2.0)),
        KDRandomGrayscale(p=0.2),
        KDRandomSolarize(p=0.2, threshold=128),
    ])

    temp_dir = Path("temp/Transform")
    temp_dir.mkdir(exist_ok=True, parents=True)
    for i in range(repeat):
        # visualize
        transformed = visualize_transform(img, transform, size=size, seed=i)
        transformed.save(temp_dir / f"{root.name}_{i}.png")


if __name__ == "__main__":
    main(**parse_args())