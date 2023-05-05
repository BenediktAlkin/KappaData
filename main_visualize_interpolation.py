import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torchvision.datasets.folder import default_loader
from argparse import ArgumentParser
from kappadata.visualization.visualize_interpolation import visualize_interpolation

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    return vars(parser.parse_args())


def main(root):
    # load image
    root = Path(root).expanduser()
    img = default_loader(root)

    size = 300

    temp_dir = Path("temp/interpolation")
    temp_dir.mkdir(exist_ok=True, parents=True)
    img, diff = visualize_interpolation(img, size=size, border=2)
    img.save(temp_dir / f"{root.name}.png")
    diff.save(temp_dir / f"{root.name}_diff.png")


if __name__ == "__main__":
    main(**parse_args())