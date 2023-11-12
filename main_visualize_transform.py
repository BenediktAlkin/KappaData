from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets.folder import default_loader

from kappadata.transforms import (
    KDComposeTransform,
    KDRandomColorJitter,
    KDRandomGaussianBlurPIL,
    KDRandomGrayscale,
    KDRandomHorizontalFlip,
    KDRandomSolarize,
    PatchwiseTransform,
    KDGaussianBlurPIL,
)
from kappadata.visualization.visualize_transform import visualize_transform


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--repeat", default=100, type=int)
    parser.add_argument("--transform", required=True, type=str)
    return vars(parser.parse_args())


def main(root, repeat, transform):
    # load image
    root = Path(root).expanduser()
    img = default_loader(root)
    temp_dir = Path(f"temp/visualize_transform/{transform}")
    temp_dir.mkdir(exist_ok=True, parents=True)

    size = 300
    if transform == "solarize":
        transform = KDRandomSolarize(p=1., threshold=128)
    elif transform == "byol0":
        transform = KDComposeTransform([
            KDRandomHorizontalFlip(),
            KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            KDGaussianBlurPIL(sigma=(0.1, 2.0)),
            KDRandomGrayscale(p=0.2),
        ])
    elif transform == "byol1":
        transform = KDComposeTransform([
            KDRandomHorizontalFlip(),
            KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            KDRandomGaussianBlurPIL(p=0.1, sigma=(0.1, 2.0)),
            KDRandomGrayscale(p=0.2),
            KDRandomSolarize(p=0.2, threshold=128),
        ])
    elif transform == "patchwise-byol0":
        transform = PatchwiseTransform(
            patch_size=25,
            transform=KDComposeTransform([
                KDRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                KDGaussianBlurPIL(sigma=(0.1, 2.0)),
                KDRandomGrayscale(p=0.2),
            ]),
        )
    else:
        raise NotImplementedError

    # store
    for i in range(repeat):
        # visualize
        transformed = visualize_transform(img, transform, size=size, seed=i)
        transformed.save(temp_dir / f"{root.name}_{i}.png")


if __name__ == "__main__":
    main(**parse_args())
